"""
MQTT client wrapper for publishing and subscribing to an MQTT broker.

This module encapsulates connection management (connect, disconnect, loop),
publication (with optional retain), and topic subscription bookkeeping. It is
used by the coordinator and, when enabled, integrates with Home Assistant by
subscribing to its status topic so discovery can be coordinated.

(c) 2024 by Christian Rödel
(c) 2024 by SukramJ
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
from aiomtec2mqtt.exceptions import MtecException

DEFAULT_RETAIN: bool = False
_LOGGER: Final = logging.getLogger(__name__)


class MqttClient:
    """Client for mqtt."""

    def __init__(
        self,
        config: dict[str, Any],
        on_mqtt_message: Callable[[mqtt.Client, Any, mqtt.MQTTMessage], None],
        hass: hass_int.HassIntegration | None = None,
    ) -> None:
        """Init the mqtt client."""
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
        self._client = self._initialize_client()

    def _on_mqtt_connect(
        self, mqttclient: mqtt.Client, userdata: Any, flags: Any, rc: int
    ) -> None:
        """Handle mqtt connect."""
        if rc == 0:
            self._connected = True
            _LOGGER.info("Connected to MQTT broker")
            # Subscribe to HA status topic and any user-requested topics
            try:
                if self._hass:
                    mqttclient.subscribe(topic=self._hass_status_topic)
                with self._lock:
                    for topic in list(self._subscribed_topics):
                        mqttclient.subscribe(topic=topic)
            except Exception as ex:  # defensive: avoid breaking network loop
                _LOGGER.warning("Post-connect subscription failed: %s", ex)
        else:
            _LOGGER.error("Error while connecting to MQTT broker: rc=%s", rc)

    def _on_mqtt_disconnect(self, mqttclient: mqtt.Client, userdata: Any, rc: int) -> None:
        self._connected = False
        _LOGGER.warning("MQTT broker disconnected: rc=%s", rc)

    def _on_mqtt_subscribe(
        self, mqttclient: mqtt.Client, userdata: Any, mid: int, granted_qos: Any
    ) -> None:
        _LOGGER.info("MQTT broker subscribed to mid %s", mid)

    def _initialize_client(self) -> mqtt.Client:
        """Initialize and start the MQTT client (non-blocking, with auto-reconnect)."""
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
            _LOGGER.info("MQTT server started")
        except Exception as ex:
            msg = f"Couldn't start MQTT: {ex}"
            _LOGGER.warning(msg)
            raise MtecException(msg) from ex
        else:
            return client

    def stop(self) -> None:
        """Stop the MQTT client."""
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

            _LOGGER.info("MQTT server stopped")
        except Exception as ex:
            _LOGGER.warning("Couldn't stop MQTT: %s", ex)

    def publish(self, topic: str, payload: str, retain: bool = DEFAULT_RETAIN) -> None:
        """Publish mqtt message."""
        _LOGGER.debug("- %s: %s", topic, str(payload))
        try:
            # paho will queue messages (including QoS0) while offline due to our configuration
            self._client.publish(topic=topic, payload=payload, qos=0, retain=retain)
        except Exception as ex:
            _LOGGER.error("Couldn't send MQTT command: %s", ex)

    def subscribe_to_topic(self, topic: str) -> None:
        """Subscribe on topic."""
        _LOGGER.debug("subscribe on %s", topic)
        try:
            with self._lock:
                if topic in self._subscribed_topics:
                    return
                self._subscribed_topics.add(topic)
            if self._connected:
                self._client.subscribe(topic=topic)
        except Exception as ex:
            _LOGGER.error("Couldn't subscribe on MQTT topic: %s: %s", topic, ex)

    def unsubscribe_from_topic(self, topic: str) -> None:
        """Unsubscribe from topic."""
        _LOGGER.debug("unsubscribe from %s", topic)
        try:
            with self._lock:
                if topic not in self._subscribed_topics:
                    return
                self._subscribed_topics.remove(topic)
            if self._connected:
                self._client.unsubscribe(topic=topic)
        except Exception as ex:
            _LOGGER.error("Couldn't unsubscribe from MQTT topic: %s: %s", topic, ex)
