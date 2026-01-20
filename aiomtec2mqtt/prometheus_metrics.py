"""
Prometheus metrics for observability.

This module provides comprehensive metrics collection for monitoring system
health, performance, and behavior using Prometheus client library.

(c) 2026 by SukramJ
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import time
from typing import TYPE_CHECKING, Any, Final

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_LOGGER: Final = logging.getLogger(__name__)

# Metric constants
DEFAULT_BUCKETS: Final = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
)


class PrometheusMetrics:
    """
    Centralized metrics collection for Prometheus.

    Provides counters, gauges, and histograms for monitoring:
    - Modbus operations (reads, writes, errors, latency)
    - MQTT operations (publishes, errors, latency)
    - System health (connections, errors, uptime)
    - Register data (battery SOC, power values)
    """

    def __init__(
        self,
        *,
        enable_http_server: bool = True,
        port: int = 9090,
        registry: CollectorRegistry | None = None,
    ) -> None:
        """
        Initialize Prometheus metrics.

        Args:
            enable_http_server: Start HTTP server for /metrics endpoint
            port: Port for metrics HTTP server
            registry: Prometheus registry (uses global REGISTRY if None)

        """
        self._port = port
        self._http_server_started = False
        self._registry = registry or REGISTRY

        # Modbus metrics
        self.modbus_reads_total = Counter(
            "aiomtec2mqtt_modbus_reads_total",
            "Total number of Modbus read operations",
            ["group", "status"],
            registry=self._registry,
        )

        self.modbus_writes_total = Counter(
            "aiomtec2mqtt_modbus_writes_total",
            "Total number of Modbus write operations",
            ["status"],
            registry=self._registry,
        )

        self.modbus_read_duration_seconds = Histogram(
            "aiomtec2mqtt_modbus_read_duration_seconds",
            "Duration of Modbus read operations in seconds",
            ["group"],
            buckets=DEFAULT_BUCKETS,
            registry=self._registry,
        )

        self.modbus_connected = Gauge(
            "aiomtec2mqtt_modbus_connected",
            "Current Modbus connection status (1=connected, 0=disconnected)",
            registry=self._registry,
        )

        self.last_successful_modbus_read_timestamp = Gauge(
            "aiomtec2mqtt_last_successful_modbus_read_timestamp_seconds",
            "Timestamp of last successful Modbus read",
            registry=self._registry,
        )

        # MQTT metrics
        self.mqtt_publishes_total = Counter(
            "aiomtec2mqtt_mqtt_publishes_total",
            "Total number of MQTT publish operations",
            ["topic", "status"],
            registry=self._registry,
        )

        self.mqtt_publish_duration_seconds = Histogram(
            "aiomtec2mqtt_mqtt_publish_duration_seconds",
            "Duration of MQTT publish operations in seconds",
            ["topic"],
            buckets=DEFAULT_BUCKETS,
            registry=self._registry,
        )

        self.mqtt_connected = Gauge(
            "aiomtec2mqtt_mqtt_connected",
            "Current MQTT connection status (1=connected, 0=disconnected)",
            registry=self._registry,
        )

        # Error metrics
        self.errors_total = Counter(
            "aiomtec2mqtt_errors_total",
            "Total number of errors by component and type",
            ["component", "error_type"],
            registry=self._registry,
        )

        # System health metrics
        self.health_status = Gauge(
            "aiomtec2mqtt_health_status",
            "System health status (0=unknown, 1=healthy, 2=degraded, 3=unhealthy)",
            ["component"],
            registry=self._registry,
        )

        self.circuit_breaker_state = Gauge(
            "aiomtec2mqtt_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
            ["component"],
            registry=self._registry,
        )

        # Application metrics
        self.uptime_seconds = Gauge(
            "aiomtec2mqtt_uptime_seconds",
            "Application uptime in seconds",
            registry=self._registry,
        )

        self.start_time = time.time()

        # Register data metrics (optional - high cardinality)
        self.battery_soc_percent = Gauge(
            "aiomtec2mqtt_battery_soc_percent",
            "Battery state of charge percentage",
            registry=self._registry,
        )

        self.grid_power_watts = Gauge(
            "aiomtec2mqtt_grid_power_watts",
            "Grid power in watts (positive=import, negative=export)",
            registry=self._registry,
        )

        self.solar_power_watts = Gauge(
            "aiomtec2mqtt_solar_power_watts",
            "Solar PV power generation in watts",
            registry=self._registry,
        )

        self.battery_power_watts = Gauge(
            "aiomtec2mqtt_battery_power_watts",
            "Battery power in watts (positive=charge, negative=discharge)",
            registry=self._registry,
        )

        if enable_http_server:
            self.start_http_server()

        _LOGGER.info("Prometheus metrics initialized")

    def record_error(self, *, component: str, error_type: str) -> None:
        """
        Record error occurrence.

        Args:
            component: Component name (modbus, mqtt, coordinator, etc.)
            error_type: Error type or exception class name

        """
        self.errors_total.labels(component=component, error_type=error_type).inc()

    def record_modbus_read(
        self,
        *,
        group: str,
        success: bool,
        duration: float | None = None,
    ) -> None:
        """
        Record Modbus read operation.

        Args:
            group: Register group name
            success: Whether read was successful
            duration: Read duration in seconds

        """
        status = "success" if success else "error"
        self.modbus_reads_total.labels(group=group, status=status).inc()

        if duration is not None:
            self.modbus_read_duration_seconds.labels(group=group).observe(duration)

        if success:
            self.last_successful_modbus_read_timestamp.set(time.time())

    def record_modbus_write(self, *, success: bool) -> None:
        """
        Record Modbus write operation.

        Args:
            success: Whether write was successful

        """
        status = "success" if success else "error"
        self.modbus_writes_total.labels(status=status).inc()

    def record_mqtt_publish(
        self,
        *,
        topic: str,
        success: bool,
        duration: float | None = None,
    ) -> None:
        """
        Record MQTT publish operation.

        Args:
            topic: MQTT topic
            success: Whether publish was successful
            duration: Publish duration in seconds

        """
        status = "success" if success else "error"
        self.mqtt_publishes_total.labels(topic=topic, status=status).inc()

        if duration is not None:
            self.mqtt_publish_duration_seconds.labels(topic=topic).observe(duration)

    def set_circuit_breaker_state(self, *, component: str, state_value: int) -> None:
        """
        Set circuit breaker state.

        Args:
            component: Component name
            state_value: State (0=closed, 1=open, 2=half_open)

        """
        self.circuit_breaker_state.labels(component=component).set(state_value)

    def set_health_status(self, *, component: str, status_value: int) -> None:
        """
        Set component health status.

        Args:
            component: Component name
            status_value: Health status (0=unknown, 1=healthy, 2=degraded, 3=unhealthy)

        """
        self.health_status.labels(component=component).set(status_value)

    def set_modbus_connection(self, *, connected: bool) -> None:
        """
        Set Modbus connection status.

        Args:
            connected: Connection status

        """
        self.modbus_connected.set(1 if connected else 0)

    def set_mqtt_connection(self, *, connected: bool) -> None:
        """
        Set MQTT connection status.

        Args:
            connected: Connection status

        """
        self.mqtt_connected.set(1 if connected else 0)

    def start_http_server(self) -> None:
        """Start Prometheus HTTP server for /metrics endpoint."""
        if self._http_server_started:
            _LOGGER.warning("HTTP server already started")
            return

        try:
            start_http_server(self._port)
            self._http_server_started = True
            _LOGGER.info("Prometheus HTTP server started on port %d", self._port)
        except OSError as ex:
            _LOGGER.error("Failed to start Prometheus HTTP server: %s", ex)

    @contextmanager
    def time_modbus_read(self, *, group: str) -> Iterator[None]:
        """
        Context manager to time Modbus read operations.

        Args:
            group: Register group name

        Yields:
            None

        Example:
            with metrics.time_modbus_read("BASE"):
                data = client.read_register_group("BASE")

        """
        start = time.time()
        try:
            yield
            duration = time.time() - start
            self.record_modbus_read(group=group, success=True, duration=duration)
        except Exception:
            duration = time.time() - start
            self.record_modbus_read(group=group, success=False, duration=duration)
            raise

    @contextmanager
    def time_mqtt_publish(self, *, topic: str) -> Iterator[None]:
        """
        Context manager to time MQTT publish operations.

        Args:
            topic: MQTT topic

        Yields:
            None

        Example:
            with metrics.time_mqtt_publish("MTEC/123/data"):
                client.publish(topic, payload)

        """
        start = time.time()
        try:
            yield
            duration = time.time() - start
            self.record_mqtt_publish(topic=topic, success=True, duration=duration)
        except Exception:
            duration = time.time() - start
            self.record_mqtt_publish(topic=topic, success=False, duration=duration)
            raise

    def update_register_values(self, *, register_data: dict[str, Any]) -> None:
        """
        Update register value metrics.

        Args:
            register_data: Dictionary of register values

        """
        if "battery_soc" in register_data:
            self.battery_soc_percent.set(register_data["battery_soc"])

        if "grid_power" in register_data:
            self.grid_power_watts.set(register_data["grid_power"])

        if "solar_power" in register_data:
            self.solar_power_watts.set(register_data["solar_power"])

        if "battery_power" in register_data:
            self.battery_power_watts.set(register_data["battery_power"])

    def update_uptime(self) -> None:
        """Update application uptime metric."""
        self.uptime_seconds.set(time.time() - self.start_time)
