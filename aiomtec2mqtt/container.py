"""
Dependency Injection container for managing service instances.

This module provides a lightweight service container for dependency injection,
supporting both singleton and factory registrations.

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any, Final, TypeVar, cast

_LOGGER: Final = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceContainer:
    """
    Lightweight dependency injection container.

    Supports singleton and factory registrations with automatic resolution.
    """

    def __init__(self) -> None:
        """Initialize the service container."""
        self._singletons: dict[type, Any] = {}
        self._factories: dict[type, Callable[[], Any]] = {}
        self._instances: dict[type, Any] = {}

    def clear(self) -> None:
        """Clear all registrations and instances."""
        self._singletons.clear()
        self._factories.clear()
        self._instances.clear()
        _LOGGER.debug("Container cleared")

    def is_registered(self, *, service_type: type) -> bool:
        """
        Check if a service type is registered.

        Args:
            service_type: The type to check

        Returns:
            True if registered

        """
        return (
            service_type in self._singletons
            or service_type in self._factories
            or service_type in self._instances
        )

    def register_factory(
        self,
        *,
        service_type: type[T],
        factory: Callable[[], T],
    ) -> None:
        """
        Register a factory function for creating instances.

        Args:
            service_type: The type/interface to register
            factory: Factory function that creates instances

        """
        if service_type in self._factories:
            _LOGGER.warning(
                "Overwriting existing factory registration for %s",
                service_type.__name__,
            )
        self._factories[service_type] = factory
        _LOGGER.debug("Registered factory: %s", service_type.__name__)

    def register_singleton(self, *, service_type: type[T], instance: T) -> None:
        """
        Register a singleton instance.

        Args:
            service_type: The type/interface to register
            instance: The concrete instance

        """
        if service_type in self._singletons:
            _LOGGER.warning(
                "Overwriting existing singleton registration for %s",
                service_type.__name__,
            )
        self._singletons[service_type] = instance
        _LOGGER.debug("Registered singleton: %s", service_type.__name__)

    def reset_instances(self) -> None:
        """Reset factory-created instances (keeps registrations)."""
        self._instances.clear()
        _LOGGER.debug("Factory instances reset")

    def resolve(self, *, service_type: type[T]) -> T:
        """
        Resolve a service instance.

        Args:
            service_type: The type/interface to resolve

        Returns:
            Service instance

        Raises:
            KeyError: If service type is not registered

        """
        # Check singletons first
        if service_type in self._singletons:
            return cast(T, self._singletons[service_type])

        # Check if we already instantiated this type
        if service_type in self._instances:
            return cast(T, self._instances[service_type])

        # Check factories
        if service_type in self._factories:
            instance = self._factories[service_type]()
            self._instances[service_type] = instance
            _LOGGER.debug("Created instance from factory: %s", service_type.__name__)
            return cast(T, instance)

        # Not found
        msg = f"Service type not registered: {service_type.__name__}"
        raise KeyError(msg)


def create_container(
    *,
    config: dict[str, Any] | None = None,
    register_map: dict[str, dict[str, Any]] | None = None,
    register_groups: list[str] | None = None,
) -> ServiceContainer:
    """
    Create and configure a service container with default services.

    Args:
        config: Configuration dictionary
        register_map: Register map dictionary
        register_groups: List of register groups

    Returns:
        Configured ServiceContainer

    """
    from aiomtec2mqtt.async_modbus_client import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        AsyncModbusClient,
    )
    from aiomtec2mqtt.async_mqtt_client import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        AsyncMqttClient,
    )
    from aiomtec2mqtt.health import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        HealthCheck,
    )

    container = ServiceContainer()

    # Register health check as singleton
    health_check = HealthCheck()
    container.register_singleton(service_type=HealthCheck, instance=health_check)

    # Register clients if config is provided
    if config and register_map and register_groups:
        # Create Modbus client
        modbus_client = AsyncModbusClient(
            config=config,
            register_map=register_map,
            register_groups=register_groups,
            health_check=health_check,
        )
        container.register_singleton(service_type=AsyncModbusClient, instance=modbus_client)

        # Create MQTT client
        mqtt_client = AsyncMqttClient(
            config=config,
            health_check=health_check,
        )
        container.register_singleton(service_type=AsyncMqttClient, instance=mqtt_client)

        _LOGGER.info("Service container created with all services")
    else:
        _LOGGER.info("Service container created (health check only)")

    return container
