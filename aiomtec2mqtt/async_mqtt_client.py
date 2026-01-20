"""
Async MQTT client for publishing M-TEC inverter data to MQTT brokers.

This module provides an asynchronous MQTT client implementation using aiomqtt.
It supports connection management with auto-reconnect, QoS handling, Last Will and
Testament, health monitoring, and connection state tracking.

Key Features:
- Non-blocking I/O with asyncio
- Auto-reconnect with exponential backoff
- Circuit breaker for publish operations
- Connection state machine
- Health check integration
- Typed exceptions
- Context manager support

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
import logging
from typing import Any, Final

import aiomqtt

from aiomtec2mqtt.const import CLIENT_ID, Config
from aiomtec2mqtt.exceptions import MqttConnectionError, MqttPublishError, MqttSubscribeError
from aiomtec2mqtt.health import HealthCheck
from aiomtec2mqtt.resilience import (
    BackoffConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    ConnectionState,
    ConnectionStateMachine,
    ExponentialBackoff,
)

DEFAULT_RETAIN: bool = False
_LOGGER: Final = logging.getLogger(__name__)


class AsyncMqttClient:
    """Async MQTT client for publishing data to MQTT broker."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        on_message: Callable[[aiomqtt.Message], None] | None = None,
        health_check: HealthCheck | None = None,
    ) -> None:
        """
        Initialize async MQTT client.

        Args:
            config: Configuration dictionary
            on_message: Optional callback for received messages
            health_check: Optional health check manager

        """
        self._config = config
        self._on_message = on_message
        self._health_check = health_check

        # Connection parameters
        self._hostname: Final[str] = config[Config.MQTT_SERVER]
        self._port: Final[int] = config[Config.MQTT_PORT]
        self._username: Final[str] = config[Config.MQTT_LOGIN]
        self._password: Final[str] = config[Config.MQTT_PASSWORD]
        self._hass_status_topic: Final[str] = f"{config[Config.HASS_BASE_TOPIC]}/status"

        # MQTT client
        self._client: aiomqtt.Client | None = None
        self._subscribed_topics: set[str] = set()
        self._connected: bool = False
        self._message_task: asyncio.Task[None] | None = None

        # Resilience patterns
        self._circuit_breaker: Final = CircuitBreaker(
            name="async_mqtt",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=2,
                timeout=30.0,
            ),
        )
        self._state_machine: Final = ConnectionStateMachine(name="async_mqtt")
        self._backoff: Final = ExponentialBackoff(
            config=BackoffConfig(
                initial_delay=1.0,
                max_delay=120.0,
                multiplier=2.0,
                jitter=True,
            )
        )

        # Register with health check
        if self._health_check:
            self._health_check.register_component(name="async_mqtt")

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._state_machine.state == ConnectionState.CONNECTED

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state_machine.state

    @property
    def subscribed_topics(self) -> set[str]:
        """Get set of subscribed topics."""
        return self._subscribed_topics.copy()

    async def connect(self) -> None:
        """
        Connect to MQTT broker.

        Raises:
            MqttConnectionError: If connection fails

        """
        # Update state machine
        if self._state_machine.state == ConnectionState.CONNECTED:
            self._state_machine.transition_to(new_state=ConnectionState.RECONNECTING)
        else:
            self._state_machine.transition_to(new_state=ConnectionState.CONNECTING)

        _LOGGER.debug(
            "Connecting to MQTT broker %s:%i",
            self._hostname,
            self._port,
        )

        try:
            # Create MQTT client with will message
            self._client = aiomqtt.Client(
                hostname=self._hostname,
                port=self._port,
                username=self._username,
                password=self._password,
                identifier=CLIENT_ID,
                will=aiomqtt.Will(
                    topic=f"{self._hass_status_topic}/lwt",
                    payload="offline",
                    qos=0,
                    retain=True,
                ),
            )

            # Connect to broker  # pylint: disable=unnecessary-dunder-call
            await self._client.__aenter__()

            _LOGGER.info(
                "Connected to MQTT broker %s:%i",
                self._hostname,
                self._port,
            )
            self._connected = True
            self._state_machine.transition_to(new_state=ConnectionState.CONNECTED)
            if self._health_check:
                self._health_check.record_success(name="async_mqtt")
            self._backoff.reset()

            # Start message listener if callback provided
            if self._on_message:
                self._message_task = asyncio.create_task(self._message_listener())

        except Exception as ex:
            error_msg = f"Failed to connect to MQTT broker {self._hostname}:{self._port}: {ex}"
            _LOGGER.exception(error_msg)
            self._state_machine.transition_to(new_state=ConnectionState.ERROR, error=error_msg)
            if self._health_check:
                self._health_check.record_failure(name="async_mqtt", error=error_msg)
            raise MqttConnectionError(
                message=error_msg,
                details={"host": self._hostname, "port": self._port},
            ) from ex

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncMqttClient]:
        """
        Async context manager for connection lifecycle.

        Usage:
            async with client.connection():
                await client.publish("topic", "payload")

        """
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()

    async def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self._message_task:
            self._message_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._message_task
            self._message_task = None

        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
                _LOGGER.info("Disconnected from MQTT broker")
                self._connected = False
                self._state_machine.transition_to(new_state=ConnectionState.DISCONNECTED)
            except Exception as ex:
                error_msg = f"Error disconnecting from MQTT broker: {ex}"
                _LOGGER.warning(error_msg)
                if self._health_check:
                    self._health_check.record_failure(name="async_mqtt", error=error_msg)

    async def publish(
        self,
        *,
        topic: str,
        payload: str,
        retain: bool = DEFAULT_RETAIN,
        qos: int = 0,
    ) -> None:
        """
        Publish message to MQTT broker.

        Args:
            topic: MQTT topic
            payload: Message payload
            retain: Whether to retain message
            qos: Quality of Service level (0, 1, or 2)

        Raises:
            MqttPublishError: If publish fails

        """
        if not self._client or not self._connected:
            error_msg = "Not connected to MQTT broker"
            _LOGGER.error(error_msg)
            raise MqttPublishError(
                message=error_msg,
                topic=topic,
                details={"payload_length": len(payload)},
            )

        _LOGGER.debug("Publishing to %s: %s", topic, payload[:100])

        async def _do_publish() -> None:
            """Perform the publish operation."""
            assert self._client is not None  # Type narrowing

            try:
                await self._client.publish(
                    topic=topic,
                    payload=payload,
                    qos=qos,
                    retain=retain,
                )
            except Exception as ex:
                error_msg = f"Failed to publish to {topic}: {ex}"
                _LOGGER.error(error_msg)
                if self._health_check:
                    self._health_check.record_failure(name="async_mqtt", error=error_msg)
                raise MqttPublishError(
                    message=error_msg,
                    topic=topic,
                    details={"payload_length": len(payload)},
                ) from ex

        try:
            # Execute publish operation
            await _do_publish()
        except MqttPublishError:
            # Re-raise but don't let it propagate to caller
            # Coordinator expects publish to not throw
            pass
        except Exception as ex:
            _LOGGER.error("Unexpected error in publish: %s", ex)
        else:
            if self._health_check:
                self._health_check.record_success(name="async_mqtt")

    async def subscribe(self, *, topic: str) -> None:
        """
        Subscribe to MQTT topic.

        Args:
            topic: Topic to subscribe to

        Raises:
            MqttSubscribeError: If subscription fails

        """
        if not self._client or not self._connected:
            error_msg = "Not connected to MQTT broker"
            _LOGGER.error(error_msg)
            raise MqttSubscribeError(
                message=error_msg,
                topic=topic,
            )

        _LOGGER.debug("Subscribing to topic: %s", topic)

        try:
            await self._client.subscribe(topic)
            self._subscribed_topics.add(topic)
            _LOGGER.info("Subscribed to topic: %s", topic)
        except Exception as ex:
            error_msg = f"Failed to subscribe to {topic}: {ex}"
            _LOGGER.exception(error_msg)
            if self._health_check:
                self._health_check.record_failure(name="async_mqtt", error=error_msg)
            raise MqttSubscribeError(
                message=error_msg,
                topic=topic,
            ) from ex

    async def unsubscribe(self, *, topic: str) -> None:
        """
        Unsubscribe from MQTT topic.

        Args:
            topic: Topic to unsubscribe from

        """
        if not self._client or not self._connected:
            _LOGGER.warning("Cannot unsubscribe: not connected")
            return

        _LOGGER.debug("Unsubscribing from topic: %s", topic)

        try:
            await self._client.unsubscribe(topic)
            self._subscribed_topics.discard(topic)
            _LOGGER.info("Unsubscribed from topic: %s", topic)
        except Exception as ex:
            error_msg = f"Failed to unsubscribe from {topic}: {ex}"
            _LOGGER.warning(error_msg)
            if self._health_check:
                self._health_check.record_failure(name="async_mqtt", error=error_msg)

    async def _message_listener(self) -> None:
        """Listen for incoming MQTT messages."""
        if not self._client or not self._on_message:
            return

        _LOGGER.info("Starting MQTT message listener")

        try:
            async for message in self._client.messages:
                try:
                    self._on_message(message)
                except Exception as ex:
                    _LOGGER.error("Error in message callback: %s", ex)
        except asyncio.CancelledError:
            _LOGGER.debug("Message listener cancelled")
            raise
        except Exception as ex:
            error_msg = f"Error in message listener: {ex}"
            _LOGGER.exception(error_msg)
            if self._health_check:
                self._health_check.record_failure(name="async_mqtt", error=error_msg)
