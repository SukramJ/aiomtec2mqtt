"""
MQTT client wrapper for publishing and subscribing to an MQTT broker.

This module encapsulates connection management (connect, disconnect, loop),
publication (with optional retain), and topic subscription bookkeeping. It is
used by the coordinator and, when enabled, integrates with Home Assistant by
subscribing to its status topic so discovery can be coordinated.

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import logging
import threading
from typing import Any, Final

from paho.mqtt import client as mqtt

from aiomtec2mqtt import hass_int
from aiomtec2mqtt.const import CLIENT_ID, Config
from aiomtec2mqtt.exceptions import MqttConnectionError, MqttPublishError, MqttSubscribeError
from aiomtec2mqtt.health import HealthCheck
from aiomtec2mqtt.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    ConnectionState,
    ConnectionStateMachine,
)

DEFAULT_RETAIN: bool = False
_LOGGER: Final = logging.getLogger(__name__)


class MqttClient:
    """Client for mqtt."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        on_mqtt_message: Callable[[mqtt.Client, Any, mqtt.MQTTMessage], None],
        hass: hass_int.HassIntegration | None = None,
        health_check: HealthCheck | None = None,
    ) -> None:
        """
        Init the mqtt client.

        Args:
            config: Configuration dictionary
            on_mqtt_message: Callback for received messages
            hass: Optional Home Assistant integration
            health_check: Optional health check manager for monitoring

        """
        self._on_mqtt_message = on_mqtt_message
        self._hass = hass
        self._username: Final[str] = config[Config.MQTT_LOGIN]
        self._password: Final[str] = config[Config.MQTT_PASSWORD]
        self._hostname: Final[str] = config[Config.MQTT_SERVER]
        self._port: Final[int] = config[Config.MQTT_PORT]
        self._hass_status_topic: Final[str] = f"{config[Config.HASS_BASE_TOPIC]}/status"
        # Initialize state BEFORE starting the paho client, because the connect callback
        # may fire immediately during loop_start in some environments (e.g. tests).
        self._subscribed_topics: set[str] = set()
        self._connected: bool = False
        self._lock: Final = threading.RLock()

        # Resilience patterns
        self._circuit_breaker: Final = CircuitBreaker(
            name="mqtt",
            config=CircuitBreakerConfig(
                failure_threshold=5,  # Open circuit after 5 failures
                success_threshold=2,  # Close after 2 successes in HALF_OPEN
                timeout=30.0,  # Try recovery after 30s
            ),
        )
        self._state_machine: Final = ConnectionStateMachine(name="mqtt")
        self._health_check = health_check

        # Register with health check if provided
        if self._health_check:
            self._health_check.register_component(name="mqtt")

        self._client = self._initialize_client()

    def publish(self, *, topic: str, payload: str, retain: bool = DEFAULT_RETAIN) -> None:
        """
        Publish mqtt message.

        Args:
            topic: MQTT topic
            payload: Message payload
            retain: Whether to retain message

        Note:
            Paho will queue messages while offline, so publish won't fail immediately.
            The circuit breaker protects against persistent publishing failures.

        """
        _LOGGER.debug("Publishing to %s: %s", topic, str(payload))

        def _do_publish() -> None:
            """Perform the publish operation."""
            try:
                # paho will queue messages (including QoS0) while offline due to our configuration
                result = self._client.publish(topic=topic, payload=payload, qos=0, retain=retain)
                # Check if publish was accepted (handle both real paho and test fake)
                if (
                    result is not None
                    and hasattr(result, "rc")
                    and result.rc != mqtt.MQTT_ERR_SUCCESS
                ):
                    error_msg = f"Publish failed for topic {topic}: rc={result.rc}"
                    _LOGGER.error(error_msg)
                    if self._health_check:
                        self._health_check.record_failure(name="mqtt", error=error_msg)
                    raise MqttPublishError(
                        message=error_msg,
                        topic=topic,
                        details={"payload_length": len(payload), "rc": result.rc},
                    )
                if self._health_check:
                    self._health_check.record_success(name="mqtt")
            except MqttPublishError:
                raise
            except Exception as ex:
                error_msg = f"Exception during publish to {topic}: {ex}"
                _LOGGER.error(error_msg)
                if self._health_check:
                    self._health_check.record_failure(name="mqtt", error=error_msg)
                raise MqttPublishError(
                    message=error_msg,
                    topic=topic,
                    details={"payload_length": len(payload)},
                ) from ex

        try:
            # Use circuit breaker for publish operations
            self._circuit_breaker.call(_do_publish)
        except MqttPublishError:
            # Log but don't re-raise - coordinator expects publish to not throw
            pass
        except Exception as ex:
            _LOGGER.error("Unexpected error in publish: %s", ex)

    def stop(self) -> None:
        """Stop the MQTT client gracefully."""
        try:
            # Unsubscribe only if connected to avoid unnecessary errors
            with self._lock:
                topics = list(self._subscribed_topics)
            if self._connected:
                for topic in topics:
                    self.unsubscribe_from_topic(topic=topic)
            # Perform a graceful disconnect before stopping the loop
            with contextlib.suppress(Exception):
                self._client.disconnect()

            # Wait for network thread to stop cleanly
            self._client.loop_stop()

            # Drop callbacks to help GC and avoid accidental calls post-stop
            self._client.on_connect = None
            self._client.on_message = None
            self._client.on_subscribe = None
            self._client.on_disconnect = None

            self._state_machine.transition_to(new_state=ConnectionState.DISCONNECTED)
            _LOGGER.info("MQTT client stopped")
        except Exception as ex:
            error_msg = f"Error stopping MQTT client: {ex}"
            _LOGGER.warning(error_msg)
            if self._health_check:
                self._health_check.record_failure(name="mqtt", error=error_msg)

    def subscribe_to_topic(self, *, topic: str) -> None:
        """
        Subscribe to MQTT topic.

        Args:
            topic: Topic to subscribe to

        Raises:
            MqttSubscribeError: If subscription fails

        """
        _LOGGER.debug("Subscribing to topic: %s", topic)
        try:
            with self._lock:
                if topic in self._subscribed_topics:
                    _LOGGER.debug("Already subscribed to topic: %s", topic)
                    return
                self._subscribed_topics.add(topic)
            # Handle both real paho (returns tuple) and test fake (returns None)
            if (
                self._connected
                and (result_tuple := self._client.subscribe(topic=topic)) is not None
            ):
                result, _ = result_tuple
                if result != mqtt.MQTT_ERR_SUCCESS:
                    error_msg = f"Subscription failed for topic {topic}: rc={result}"
                    _LOGGER.error(error_msg)
                    if self._health_check:
                        self._health_check.record_failure(name="mqtt", error=error_msg)
                    raise MqttSubscribeError(
                        message=error_msg, topic=topic, details={"rc": result}
                    )
        except MqttSubscribeError:
            raise
        except Exception as ex:
            error_msg = f"Exception during subscribe to {topic}: {ex}"
            _LOGGER.error(error_msg)
            if self._health_check:
                self._health_check.record_failure(name="mqtt", error=error_msg)
            raise MqttSubscribeError(message=error_msg, topic=topic) from ex

    def unsubscribe_from_topic(self, *, topic: str) -> None:
        """
        Unsubscribe from MQTT topic.

        Args:
            topic: Topic to unsubscribe from

        Raises:
            MqttSubscribeError: If unsubscription fails

        """
        _LOGGER.debug("Unsubscribing from topic: %s", topic)
        try:
            with self._lock:
                if topic not in self._subscribed_topics:
                    _LOGGER.debug("Not subscribed to topic: %s", topic)
                    return
                self._subscribed_topics.remove(topic)
            # Handle both real paho (returns tuple) and test fake (returns None)
            if (
                self._connected
                and (result_tuple := self._client.unsubscribe(topic=topic)) is not None
            ):
                result, _ = result_tuple
                if result != mqtt.MQTT_ERR_SUCCESS:
                    error_msg = f"Unsubscription failed for topic {topic}: rc={result}"
                    _LOGGER.warning(error_msg)
                    if self._health_check:
                        self._health_check.record_failure(name="mqtt", error=error_msg)
        except Exception as ex:
            error_msg = f"Exception during unsubscribe from {topic}: {ex}"
            _LOGGER.warning(error_msg)
            if self._health_check:
                self._health_check.record_failure(name="mqtt", error=error_msg)

    def _get_connection_error_message(self, *, rc: int) -> str:
        """
        Convert MQTT return code to error message.

        Args:
            rc: MQTT return code

        Returns:
            Human-readable error message

        """
        error_messages = {
            1: "Connection refused - incorrect protocol version",
            2: "Connection refused - invalid client identifier",
            3: "Connection refused - server unavailable",
            4: "Connection refused - bad username or password",
            5: "Connection refused - not authorized",
        }
        return error_messages.get(rc, f"Connection error (rc={rc})")

    def _initialize_client(self) -> mqtt.Client:
        """
        Initialize and start the MQTT client (non-blocking, with auto-reconnect).

        Returns:
            Initialized MQTT client

        Raises:
            MqttConnectionError: If client initialization fails

        """
        self._state_machine.transition_to(new_state=ConnectionState.CONNECTING)

        try:
            client = mqtt.Client(client_id=CLIENT_ID, protocol=mqtt.MQTTv311, clean_session=True)
            client.username_pw_set(username=self._username, password=self._password)
            # Route paho internal logs into our logger (useful for debugging)
            with contextlib.suppress(Exception):
                client.enable_logger(_LOGGER)
            # Set handlers before connecting to avoid missing early events
            client.on_connect = self._on_mqtt_connect
            client.on_message = self._on_mqtt_message
            client.on_subscribe = self._on_mqtt_subscribe
            client.on_disconnect = self._on_mqtt_disconnect

            # Set a Last Will and Testament to signal unexpected offline state
            client.will_set(
                topic=f"{self._hass_status_topic}/lwt",
                payload="offline",
                retain=True,
            )

            # Configure exponential reconnect backoff to reduce tight retry loops
            # Note: Paho has built-in auto-reconnect, so we keep this instead of our own backoff
            client.reconnect_delay_set(min_delay=1, max_delay=120)

            # Optimize internal queues: allow QoS0 to be queued while offline
            with contextlib.suppress(Exception):
                client.max_inflight_messages_set(20)
                client.max_queued_messages_set(1000)
                client.queue_qos0_messages = True  # type: ignore[attr-defined]

            # Use async connect to avoid blocking and enable auto-reconnect in loop
            client.connect_async(host=self._hostname, port=self._port, keepalive=60)

            # Start network loop after initiating connection
            client.loop_start()
            _LOGGER.info("MQTT client started, connecting to %s:%s", self._hostname, self._port)
        except Exception as ex:
            error_msg = f"Failed to initialize MQTT client: {ex}"
            _LOGGER.exception(error_msg)
            self._state_machine.transition_to(new_state=ConnectionState.ERROR, error=error_msg)
            if self._health_check:
                self._health_check.record_failure(name="mqtt", error=error_msg)
            raise MqttConnectionError(
                message=error_msg,
                details={"host": self._hostname, "port": self._port},
            ) from ex
        else:
            return client

    def _on_mqtt_connect(  # kwonly: disable
        self, mqttclient: mqtt.Client, userdata: Any, flags: Any, rc: int
    ) -> None:
        """
        Handle mqtt connect.

        Args:
            mqttclient: MQTT client instance
            userdata: User data
            flags: Connection flags
            rc: Return code (0 = success)

        """
        if rc == 0:
            self._connected = True
            _LOGGER.info("Connected to MQTT broker %s:%s", self._hostname, self._port)
            self._state_machine.transition_to(new_state=ConnectionState.CONNECTED)
            if self._health_check:
                self._health_check.record_success(name="mqtt")

            # Subscribe to HA status topic and any user-requested topics
            try:
                if self._hass:
                    mqttclient.subscribe(topic=self._hass_status_topic)
                with self._lock:
                    for topic in list(self._subscribed_topics):
                        mqttclient.subscribe(topic=topic)
            except Exception as ex:  # defensive: avoid breaking network loop
                _LOGGER.warning("Post-connect subscription failed: %s", ex)
                if self._health_check:
                    self._health_check.record_failure(
                        name="mqtt", error=f"Subscription failed: {ex}"
                    )
        else:
            error_msg = self._get_connection_error_message(rc=rc)
            _LOGGER.error("Error while connecting to MQTT broker: %s", error_msg)
            self._state_machine.transition_to(new_state=ConnectionState.ERROR, error=error_msg)
            if self._health_check:
                self._health_check.record_failure(name="mqtt", error=error_msg)

    def _on_mqtt_disconnect(  # kwonly: disable
        self, mqttclient: mqtt.Client, userdata: Any, rc: int
    ) -> None:
        """
        Handle mqtt disconnect.

        Args:
            mqttclient: MQTT client instance
            userdata: User data
            rc: Return code (0 = clean disconnect)

        """
        self._connected = False
        if rc == 0:
            _LOGGER.info("MQTT broker disconnected cleanly")
            self._state_machine.transition_to(new_state=ConnectionState.DISCONNECTED)
        else:
            error_msg = f"MQTT broker disconnected unexpectedly: rc={rc}"
            _LOGGER.warning(error_msg)
            self._state_machine.transition_to(new_state=ConnectionState.RECONNECTING)
            if self._health_check:
                self._health_check.record_failure(name="mqtt", error=error_msg)

    def _on_mqtt_subscribe(  # kwonly: disable
        self, mqttclient: mqtt.Client, userdata: Any, mid: int, granted_qos: Any
    ) -> None:
        """
        Handle mqtt subscribe confirmation.

        Args:
            mqttclient: MQTT client instance
            userdata: User data
            mid: Message ID
            granted_qos: Granted QoS levels

        """
        _LOGGER.info("MQTT broker subscribed to mid %s", mid)
