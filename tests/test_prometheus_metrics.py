"""Tests for prometheus metrics."""

from __future__ import annotations

import time

from prometheus_client import CollectorRegistry
import pytest

from aiomtec2mqtt.prometheus_metrics import PrometheusMetrics


class TestPrometheusMetrics:
    """Test PrometheusMetrics class."""

    def test_initialization(self) -> None:
        """Test metrics initialization."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        assert metrics._port == 9090
        assert metrics._http_server_started is False
        assert metrics.modbus_reads_total is not None
        assert metrics.mqtt_publishes_total is not None
        assert metrics.errors_total is not None

    def test_initialization_custom_port(self) -> None:
        """Test initialization with custom port."""
        metrics = PrometheusMetrics(
            enable_http_server=False, port=8080, registry=CollectorRegistry()
        )

        assert metrics._port == 8080

    def test_multiple_operations(self) -> None:
        """Test multiple operations recorded correctly."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        # Record multiple Modbus reads
        metrics.record_modbus_read(group="BASE", success=True)
        metrics.record_modbus_read(group="BASE", success=True)
        metrics.record_modbus_read(group="BASE", success=False)

        success_count = metrics.modbus_reads_total.labels(
            group="BASE", status="success"
        )._value._value
        error_count = metrics.modbus_reads_total.labels(group="BASE", status="error")._value._value

        assert success_count == 2.0
        assert error_count == 1.0

    def test_record_error(self) -> None:
        """Test recording errors."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        metrics.record_error(component="modbus", error_type="TimeoutError")

        counter_value = metrics.errors_total.labels(
            component="modbus", error_type="TimeoutError"
        )._value._value
        assert counter_value == 1.0

    def test_record_modbus_read_failure(self) -> None:
        """Test recording failed Modbus read."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        metrics.record_modbus_read(group="BASE", success=False, duration=0.5)

        counter_value = metrics.modbus_reads_total.labels(
            group="BASE", status="error"
        )._value._value
        assert counter_value == 1.0

    def test_record_modbus_read_success(self) -> None:
        """Test recording successful Modbus read."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        initial_time = metrics.last_successful_modbus_read_timestamp._value._value

        metrics.record_modbus_read(group="BASE", success=True, duration=0.5)

        # Check counter increased
        counter_value = metrics.modbus_reads_total.labels(
            group="BASE", status="success"
        )._value._value
        assert counter_value == 1.0

        # Check timestamp updated
        new_time = metrics.last_successful_modbus_read_timestamp._value._value
        assert new_time > initial_time

    def test_record_modbus_write_failure(self) -> None:
        """Test recording failed Modbus write."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        metrics.record_modbus_write(success=False)

        counter_value = metrics.modbus_writes_total.labels(status="error")._value._value
        assert counter_value == 1.0

    def test_record_modbus_write_success(self) -> None:
        """Test recording successful Modbus write."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        metrics.record_modbus_write(success=True)

        counter_value = metrics.modbus_writes_total.labels(status="success")._value._value
        assert counter_value == 1.0

    def test_record_mqtt_publish_failure(self) -> None:
        """Test recording failed MQTT publish."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        metrics.record_mqtt_publish(topic="test/topic", success=False)

        counter_value = metrics.mqtt_publishes_total.labels(
            topic="test/topic", status="error"
        )._value._value
        assert counter_value == 1.0

    def test_record_mqtt_publish_success(self) -> None:
        """Test recording successful MQTT publish."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        metrics.record_mqtt_publish(topic="test/topic", success=True, duration=0.1)

        counter_value = metrics.mqtt_publishes_total.labels(
            topic="test/topic", status="success"
        )._value._value
        assert counter_value == 1.0

    def test_set_circuit_breaker_state(self) -> None:
        """Test setting circuit breaker state."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        metrics.set_circuit_breaker_state(component="modbus", state_value=0)
        assert metrics.circuit_breaker_state.labels(component="modbus")._value._value == 0.0

        metrics.set_circuit_breaker_state(component="modbus", state_value=1)
        assert metrics.circuit_breaker_state.labels(component="modbus")._value._value == 1.0

    def test_set_health_status(self) -> None:
        """Test setting health status."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        metrics.set_health_status(component="modbus", status_value=1)
        assert metrics.health_status.labels(component="modbus")._value._value == 1.0

        metrics.set_health_status(component="modbus", status_value=3)
        assert metrics.health_status.labels(component="modbus")._value._value == 3.0

    def test_set_modbus_connection(self) -> None:
        """Test setting Modbus connection status."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        metrics.set_modbus_connection(connected=True)
        assert metrics.modbus_connected._value._value == 1.0

        metrics.set_modbus_connection(connected=False)
        assert metrics.modbus_connected._value._value == 0.0

    def test_set_mqtt_connection(self) -> None:
        """Test setting MQTT connection status."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        metrics.set_mqtt_connection(connected=True)
        assert metrics.mqtt_connected._value._value == 1.0

        metrics.set_mqtt_connection(connected=False)
        assert metrics.mqtt_connected._value._value == 0.0

    def test_time_modbus_read_context_manager_failure(self) -> None:
        """Test time_modbus_read context manager with failure."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        with pytest.raises(ValueError), metrics.time_modbus_read(group="BASE"):
            raise ValueError("Test error")

        counter_value = metrics.modbus_reads_total.labels(
            group="BASE", status="error"
        )._value._value
        assert counter_value == 1.0

    def test_time_modbus_read_context_manager_success(self) -> None:
        """Test time_modbus_read context manager with success."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        with metrics.time_modbus_read(group="BASE"):
            time.sleep(0.01)  # Simulate read operation

        counter_value = metrics.modbus_reads_total.labels(
            group="BASE", status="success"
        )._value._value
        assert counter_value == 1.0

    def test_time_mqtt_publish_context_manager_failure(self) -> None:
        """Test time_mqtt_publish context manager with failure."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        with pytest.raises(RuntimeError), metrics.time_mqtt_publish(topic="test/topic"):
            raise RuntimeError("Test error")

        counter_value = metrics.mqtt_publishes_total.labels(
            topic="test/topic", status="error"
        )._value._value
        assert counter_value == 1.0

    def test_time_mqtt_publish_context_manager_success(self) -> None:
        """Test time_mqtt_publish context manager with success."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        with metrics.time_mqtt_publish(topic="test/topic"):
            time.sleep(0.01)  # Simulate publish operation

        counter_value = metrics.mqtt_publishes_total.labels(
            topic="test/topic", status="success"
        )._value._value
        assert counter_value == 1.0

    def test_update_register_values(self) -> None:
        """Test updating register value metrics."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        register_data = {
            "battery_soc": 85.5,
            "grid_power": 1500,
            "solar_power": 3000,
            "battery_power": -500,
        }

        metrics.update_register_values(register_data=register_data)

        assert metrics.battery_soc_percent._value._value == 85.5
        assert metrics.grid_power_watts._value._value == 1500
        assert metrics.solar_power_watts._value._value == 3000
        assert metrics.battery_power_watts._value._value == -500

    def test_update_register_values_partial(self) -> None:
        """Test updating with partial register data."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        register_data = {
            "battery_soc": 50,
        }

        metrics.update_register_values(register_data=register_data)

        assert metrics.battery_soc_percent._value._value == 50

    def test_update_uptime(self) -> None:
        """Test updating uptime metric."""
        metrics = PrometheusMetrics(enable_http_server=False, registry=CollectorRegistry())

        # Sleep a bit to ensure uptime increases
        time.sleep(0.1)

        metrics.update_uptime()

        uptime = metrics.uptime_seconds._value._value
        assert uptime > 0
