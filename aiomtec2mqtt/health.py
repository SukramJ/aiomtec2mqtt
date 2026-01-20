"""
Health check system for aiomtec2mqtt.

This module provides health monitoring for application components (Modbus, MQTT, etc.)
and overall system health status.

(c) 2026 by SukramJ
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
from typing import Any, Final

_LOGGER: Final = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """
    Health information for a component.

    Attributes:
        name: Component name
        status: Current health status
        last_success: Timestamp of last successful operation
        last_failure: Timestamp of last failure
        error_count: Number of consecutive errors
        error_rate: Recent error rate (errors per minute)
        message: Optional status message
        details: Additional health details

    """

    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_success: datetime | None = None
    last_failure: datetime | None = None
    error_count: int = 0
    error_rate: float = 0.0
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def is_degraded(self) -> bool:
        """
        Check if component is degraded.

        Returns:
            True if status is DEGRADED

        """
        return self.status == HealthStatus.DEGRADED

    def is_healthy(self) -> bool:
        """
        Check if component is healthy.

        Returns:
            True if status is HEALTHY

        """
        return self.status == HealthStatus.HEALTHY

    def is_unhealthy(self) -> bool:
        """
        Check if component is unhealthy.

        Returns:
            True if status is UNHEALTHY

        """
        return self.status == HealthStatus.UNHEALTHY


@dataclass
class SystemHealth:
    """
    Overall system health status.

    Attributes:
        status: Overall system status
        components: Health status of each component
        check_time: When health check was performed
        uptime: System uptime
        message: Optional system-level message

    """

    status: HealthStatus
    components: dict[str, ComponentHealth]
    check_time: datetime
    uptime: timedelta | None = None
    message: str | None = None

    def is_healthy(self) -> bool:
        """
        Check if system is healthy.

        Returns:
            True if overall status is HEALTHY

        """
        return self.status == HealthStatus.HEALTHY


class HealthCheck:
    """
    Health check manager for monitoring component health.

    This class manages health checks for multiple components and provides
    overall system health status.

    Example:
        health = HealthCheck()

        # Register components
        health.register_component("modbus", modbus_health_check)
        health.register_component("mqtt", mqtt_health_check)

        # Perform health check
        system_health = health.check_health()

        if system_health.is_healthy():
            logger.info("System healthy")
        else:
            logger.warning("System unhealthy: %s", system_health.message)

    """

    def __init__(self, *, stale_threshold: float = 300.0) -> None:
        """
        Initialize health check manager.

        Args:
            stale_threshold: Seconds after which component is considered stale

        """
        self._components: dict[str, ComponentHealth] = {}
        self._health_checks: dict[str, Callable[[], ComponentHealth]] = {}
        self._stale_threshold = stale_threshold
        self._start_time = datetime.now(UTC)
        _LOGGER.info("Health check manager initialized")

    def check_health(self) -> SystemHealth:
        """
        Perform health check on all components.

        Returns:
            System health status with component details

        """
        now = datetime.now(UTC)
        components = {}

        # Run health check functions
        for name, health_check in self._health_checks.items():
            try:
                component_health = health_check()
                self._components[name] = component_health
            except Exception as ex:  # noqa: BLE001
                _LOGGER.exception(
                    "Health check failed for %s",
                    name,
                )
                self.record_failure(name=name, error=f"Health check error: {ex}")

        # Check for stale components
        for name, component in self._components.items():
            if component.last_success is not None:
                time_since_success = (now - component.last_success).total_seconds()
                if time_since_success > self._stale_threshold:
                    component.status = HealthStatus.UNHEALTHY
                    component.message = f"Stale (no success for {time_since_success:.0f}s)"

            # Calculate error rate
            if component.last_failure is not None:
                time_since_failure = (now - component.last_failure).total_seconds()
                if 0 < time_since_failure < 60:  # Last minute
                    component.error_rate = component.error_count / (time_since_failure / 60)

            components[name] = component

        # Determine overall system status
        system_status = self._calculate_system_status(components=components)

        # Calculate uptime
        uptime = now - self._start_time

        # Generate system message
        unhealthy_components = [name for name, comp in components.items() if comp.is_unhealthy()]
        degraded_components = [name for name, comp in components.items() if comp.is_degraded()]

        if unhealthy_components:
            message = f"Unhealthy components: {', '.join(unhealthy_components)}"
        elif degraded_components:
            message = f"Degraded components: {', '.join(degraded_components)}"
        else:
            message = "All components healthy"

        return SystemHealth(
            status=system_status,
            components=components,
            check_time=now,
            uptime=uptime,
            message=message,
        )

    def get_component_health(self, *, name: str) -> ComponentHealth | None:
        """
        Get health status for a specific component.

        Args:
            name: Component name

        Returns:
            Component health or None if not found

        """
        return self._components.get(name)

    def get_uptime(self) -> timedelta:
        """
        Get system uptime.

        Returns:
            Time since health check manager was initialized

        """
        return datetime.now(UTC) - self._start_time

    def record_failure(self, *, name: str, error: str | None = None) -> None:
        """
        Record failed operation for component.

        Args:
            name: Component name
            error: Optional error message

        """
        if name not in self._components:
            self.register_component(name=name)

        component = self._components[name]
        component.last_failure = datetime.now(UTC)
        component.error_count += 1
        component.message = error

        # Update status based on error count
        if component.error_count >= 5:
            component.status = HealthStatus.UNHEALTHY
        elif component.error_count >= 2:
            component.status = HealthStatus.DEGRADED
        else:
            component.status = HealthStatus.HEALTHY

    def record_success(self, *, name: str) -> None:
        """
        Record successful operation for component.

        Args:
            name: Component name

        """
        if name not in self._components:
            self.register_component(name=name)

        component = self._components[name]
        component.last_success = datetime.now(UTC)
        component.error_count = 0  # Reset error count on success
        component.status = HealthStatus.HEALTHY
        component.message = None

    def register_component(
        self,
        *,
        name: str,
        health_check: Callable[[], ComponentHealth] | None = None,
    ) -> None:
        """
        Register a component for health monitoring.

        Args:
            name: Component name
            health_check: Optional health check function that returns ComponentHealth

        """
        if name not in self._components:
            self._components[name] = ComponentHealth(name=name)
            _LOGGER.debug("Registered component for health monitoring: %s", name)

        if health_check is not None:
            self._health_checks[name] = health_check
            _LOGGER.debug("Registered health check function for: %s", name)

    def reset_component(self, *, name: str) -> None:
        """
        Reset component health state.

        Args:
            name: Component name

        """
        if name in self._components:
            self._components[name] = ComponentHealth(name=name)
            _LOGGER.info("Reset health for component: %s", name)

    def update_component_health(
        self,
        *,
        name: str,
        status: HealthStatus,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Update component health status.

        Args:
            name: Component name
            status: New health status
            message: Optional status message
            details: Optional additional details

        """
        if name not in self._components:
            self.register_component(name=name)

        component = self._components[name]
        component.status = status
        component.message = message

        if details:
            component.details.update(details)

        _LOGGER.debug("Updated health for %s: %s", name, status.value)

    def _calculate_system_status(
        self,
        *,
        components: dict[str, ComponentHealth],
    ) -> HealthStatus:
        """
        Calculate overall system status from component statuses.

        Args:
            components: Component health status dict

        Returns:
            Overall system health status

        """
        if not components:
            return HealthStatus.UNKNOWN

        # System is unhealthy if any component is unhealthy
        if any(comp.is_unhealthy() for comp in components.values()):
            return HealthStatus.UNHEALTHY

        # System is degraded if any component is degraded
        if any(comp.is_degraded() for comp in components.values()):
            return HealthStatus.DEGRADED

        # System is healthy if all components are healthy
        if all(comp.is_healthy() for comp in components.values()):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN
