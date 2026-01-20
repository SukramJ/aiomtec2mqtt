"""
Testing utilities for dependency injection and mocking.

This module provides fake implementations of protocols for testing purposes,
eliminating the need for complex mocking in tests.

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from aiomtec2mqtt.health import ComponentHealth, HealthStatus, SystemHealth
from aiomtec2mqtt.protocols import (
    ConfigProviderProtocol,
    HealthMonitorProtocol,
    ModbusClientProtocol,
    MqttClientProtocol,
)
from aiomtec2mqtt.resilience import ConnectionState


class FakeModbusClient:
    """Fake Modbus client for testing."""

    def __init__(
        self,
        *,
        register_data: dict[int, list[int]] | None = None,
        should_fail: bool = False,
    ) -> None:
        """
        Initialize fake Modbus client.

        Args:
            register_data: Dictionary of address -> register values
            should_fail: If True, operations will fail

        """
        self._register_data = register_data or {}
        self._should_fail = should_fail
        self._connected = False
        self._error_count = 0
        self._state = ConnectionState.DISCONNECTED

    @property
    def error_count(self) -> int:
        """Get error count."""
        return self._error_count

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    @property
    def state(self) -> ConnectionState:
        """Get connection state."""
        return self._state

    async def connect(self) -> bool:
        """Connect to Modbus server."""
        if self._should_fail:
            self._state = ConnectionState.ERROR
            self._error_count += 1
            return False

        self._connected = True
        self._state = ConnectionState.CONNECTED
        return True

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[ModbusClientProtocol]:
        """Async context manager for connection lifecycle."""
        await self.connect()
        try:
            yield self  # type: ignore[misc]
        finally:
            await self.disconnect()

    async def disconnect(self) -> None:
        """Disconnect from Modbus server."""
        self._connected = False
        self._state = ConnectionState.DISCONNECTED

    async def read_holding_registers(
        self,
        *,
        address: int,
        count: int,
    ) -> Any:
        """Read holding registers."""
        if not self._connected:
            msg = "Not connected"
            raise RuntimeError(msg)

        if self._should_fail:
            self._error_count += 1
            msg = "Read failed"
            raise RuntimeError(msg)

        # Return fake data
        values = self._register_data.get(address, [0] * count)
        return type("Response", (), {"registers": values, "isError": lambda: False})()

    async def read_register_group(
        self,
        *,
        group_name: str,
    ) -> dict[str, Any]:
        """Read all registers in a group."""
        if not self._connected:
            return {}

        if self._should_fail:
            self._error_count += 1
            return {}

        # Return fake data based on group
        return {
            "battery_soc": 50,
            "battery_voltage": 48.5,
            "grid_power": 1500,
        }


class FakeMqttClient:
    """Fake MQTT client for testing."""

    def __init__(self, *, should_fail: bool = False) -> None:
        """
        Initialize fake MQTT client.

        Args:
            should_fail: If True, operations will fail

        """
        self._should_fail = should_fail
        self._connected = False
        self._state = ConnectionState.DISCONNECTED
        self._subscribed_topics: set[str] = set()
        self._published_messages: list[tuple[str, str, bool, int]] = []

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    @property
    def published_messages(self) -> list[tuple[str, str, bool, int]]:
        """Get list of published messages."""
        return self._published_messages

    @property
    def state(self) -> ConnectionState:
        """Get connection state."""
        return self._state

    @property
    def subscribed_topics(self) -> set[str]:
        """Get subscribed topics."""
        return self._subscribed_topics

    def clear_published(self) -> None:
        """Clear published messages list."""
        self._published_messages.clear()

    async def connect(self) -> None:
        """Connect to MQTT broker."""
        if self._should_fail:
            self._state = ConnectionState.ERROR
            msg = "Connection failed"
            raise RuntimeError(msg)

        self._connected = True
        self._state = ConnectionState.CONNECTED

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[MqttClientProtocol]:
        """Async context manager for connection lifecycle."""
        await self.connect()
        try:
            yield self  # type: ignore[misc]
        finally:
            await self.disconnect()

    async def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        self._connected = False
        self._state = ConnectionState.DISCONNECTED

    async def publish(
        self,
        *,
        topic: str,
        payload: str,
        retain: bool = False,
        qos: int = 0,
    ) -> None:
        """Publish message."""
        if not self._connected:
            msg = "Not connected"
            raise RuntimeError(msg)

        if self._should_fail:
            msg = "Publish failed"
            raise RuntimeError(msg)

        self._published_messages.append((topic, payload, retain, qos))

    async def subscribe(self, *, topic: str) -> None:
        """Subscribe to topic."""
        if not self._connected:
            msg = "Not connected"
            raise RuntimeError(msg)

        self._subscribed_topics.add(topic)

    async def unsubscribe(self, *, topic: str) -> None:
        """Unsubscribe from topic."""
        self._subscribed_topics.discard(topic)


class FakeConfigProvider:
    """Fake configuration provider for testing."""

    def __init__(self, *, config_dict: dict[str, Any] | None = None) -> None:
        """
        Initialize fake config provider.

        Args:
            config_dict: Configuration dictionary

        """
        self._config = config_dict or {}

    def __contains__(self, key: str) -> bool:  # kwonly: disable
        """Check if configuration key exists."""
        return key in self._config

    def __getitem__(self, key: str) -> Any:  # kwonly: disable
        """Get configuration value using dict syntax."""
        return self._config[key]

    def get(self, *, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def items(self) -> Any:
        """Get all configuration items."""
        return self._config.items()

    def set(self, *, key: str, value: Any) -> None:
        """Set configuration value (test helper)."""
        self._config[key] = value


class FakeHealthMonitor:
    """Fake health monitor for testing."""

    def __init__(self) -> None:
        """Initialize fake health monitor."""
        self._components: dict[str, ComponentHealth] = {}

    def check_health(self) -> SystemHealth:
        """Check overall system health."""
        from datetime import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
            UTC,
            datetime,
        )

        all_healthy = all(c.status == HealthStatus.HEALTHY for c in self._components.values())
        return SystemHealth(
            status=HealthStatus.HEALTHY if all_healthy else HealthStatus.UNHEALTHY,
            components=self._components.copy(),
            check_time=datetime.now(UTC),
            message="Fake health check",
        )

    def get_component_health(self, *, component_name: str) -> ComponentHealth | None:
        """Get health status of specific component."""
        return self._components.get(component_name)

    def record_failure(
        self,
        *,
        component_name: str,
        error: str | None = None,
    ) -> None:
        """Record failed operation."""
        if component_name not in self._components:
            self.register_component(name=component_name)

        comp = self._components[component_name]
        comp.status = HealthStatus.UNHEALTHY
        comp.error_count += 1
        if error:
            comp.message = error

    def record_success(self, *, component_name: str) -> None:
        """Record successful operation."""
        if component_name not in self._components:
            self.register_component(name=component_name)

        comp = self._components[component_name]
        comp.status = HealthStatus.HEALTHY
        comp.error_count = 0  # Reset error count on success

    def register_component(self, *, name: str) -> None:
        """Register a component."""
        self._components[name] = ComponentHealth(
            name=name,
            status=HealthStatus.UNKNOWN,
            message="Registered",
        )

    def reset_component(self, *, component_name: str) -> None:
        """Reset component health state."""
        if component_name in self._components:
            comp = self._components[component_name]
            comp.status = HealthStatus.UNKNOWN
            comp.error_count = 0
            comp.message = None


def create_test_container() -> Any:
    """
    Create a service container configured for testing.

    Returns:
        ServiceContainer with fake services

    """
    from aiomtec2mqtt.container import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        ServiceContainer,
    )

    container = ServiceContainer()

    # Register fake services
    container.register_singleton(service_type=ModbusClientProtocol, instance=FakeModbusClient())
    container.register_singleton(service_type=MqttClientProtocol, instance=FakeMqttClient())
    container.register_singleton(
        service_type=ConfigProviderProtocol,  # type: ignore[type-abstract]
        instance=FakeConfigProvider(),
    )
    container.register_singleton(service_type=HealthMonitorProtocol, instance=FakeHealthMonitor())  # type: ignore[type-abstract]

    return container
