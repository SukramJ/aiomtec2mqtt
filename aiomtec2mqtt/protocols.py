"""
Protocol definitions for dependency injection.

This module defines runtime-checkable protocols for all major components,
enabling loose coupling and easy testing through dependency injection.

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""
# pylint: disable=unnecessary-ellipsis  # Ellipsis required for Protocol methods

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from aiomtec2mqtt.health import ComponentHealth, SystemHealth
from aiomtec2mqtt.resilience import ConnectionState


@runtime_checkable
class ModbusClientProtocol(Protocol):
    """Protocol for Modbus client implementations."""

    @property
    def error_count(self) -> int:
        """Get error count."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        ...

    @property
    def state(self) -> ConnectionState:
        """Get connection state."""
        ...

    async def connect(self) -> bool:
        """Connect to Modbus server."""
        ...

    def connection(self) -> AsyncIterator[ModbusClientProtocol]:
        """Async context manager for connection lifecycle."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from Modbus server."""
        ...

    async def read_holding_registers(
        self,
        *,
        address: int,
        count: int,
    ) -> Any:
        """Read holding registers."""
        ...

    async def read_register_group(
        self,
        *,
        group_name: str,
    ) -> dict[str, Any]:
        """Read all registers in a group."""
        ...


@runtime_checkable
class MqttClientProtocol(Protocol):
    """Protocol for MQTT client implementations."""

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        ...

    @property
    def state(self) -> ConnectionState:
        """Get connection state."""
        ...

    @property
    def subscribed_topics(self) -> set[str]:
        """Get subscribed topics."""
        ...

    async def connect(self) -> None:
        """Connect to MQTT broker."""
        ...

    def connection(self) -> AsyncIterator[MqttClientProtocol]:
        """Async context manager for connection lifecycle."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        ...

    async def publish(
        self,
        *,
        topic: str,
        payload: str,
        retain: bool = False,
        qos: int = 0,
    ) -> None:
        """Publish message to MQTT broker."""
        ...

    async def subscribe(self, *, topic: str) -> None:
        """Subscribe to MQTT topic."""
        ...

    async def unsubscribe(self, *, topic: str) -> None:
        """Unsubscribe from MQTT topic."""
        ...


@runtime_checkable
class ConfigProviderProtocol(Protocol):
    """Protocol for configuration providers."""

    def __contains__(self, key: str) -> bool:  # kwonly: disable
        """Check if configuration key exists."""
        ...

    def __getitem__(self, key: str) -> Any:  # kwonly: disable
        """Get configuration value using dict syntax."""
        ...

    def get(self, *, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...

    def items(self) -> Any:
        """Get all configuration items."""
        ...


@runtime_checkable
class HealthMonitorProtocol(Protocol):
    """Protocol for health monitoring."""

    def check_health(self) -> SystemHealth:
        """Check overall system health."""
        ...

    def get_component_health(self, *, component_name: str) -> ComponentHealth | None:
        """Get health status of specific component."""
        ...

    def record_failure(
        self,
        *,
        component_name: str,
        error: str | None = None,
    ) -> None:
        """Record failed operation."""
        ...

    def record_success(self, *, component_name: str) -> None:
        """Record successful operation."""
        ...

    def register_component(self, *, name: str) -> None:
        """Register a component for health monitoring."""
        ...

    def reset_component(self, *, component_name: str) -> None:
        """Reset component health state."""
        ...


@runtime_checkable
class RegisterProcessorProtocol(Protocol):
    """Protocol for register value processors."""

    def can_process(self, *, register_name: str, register_info: dict[str, Any]) -> bool:
        """Check if this processor can handle the register."""
        ...

    def process(
        self,
        *,
        register_name: str,
        raw_value: int | float,
        register_info: dict[str, Any],
    ) -> Any:
        """Process register value."""
        ...


@runtime_checkable
class FormulaEvaluatorProtocol(Protocol):
    """Protocol for formula evaluation."""

    def evaluate(
        self,
        *,
        formula: str,
        context: dict[str, Any],
    ) -> float | int:
        """Evaluate formula with given context."""
        ...

    def get_dependencies(self, *, formula: str) -> set[str]:
        """Extract variable dependencies from formula."""
        ...

    def validate_formula(self, *, formula: str) -> bool:
        """Validate formula syntax."""
        ...
