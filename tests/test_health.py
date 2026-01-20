"""Tests for health check system."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import time

from aiomtec2mqtt.health import ComponentHealth, HealthCheck, HealthStatus, SystemHealth


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_initialization(self) -> None:
        """Component health should initialize with defaults."""
        component = ComponentHealth(name="test")

        assert component.name == "test"
        assert component.status == HealthStatus.UNKNOWN
        assert component.last_success is None
        assert component.last_failure is None
        assert component.error_count == 0
        assert component.error_rate == 0.0
        assert component.message is None
        assert component.details == {}

    def test_is_degraded(self) -> None:
        """is_degraded should check DEGRADED status."""
        component = ComponentHealth(name="test", status=HealthStatus.DEGRADED)
        assert component.is_degraded()

        component.status = HealthStatus.HEALTHY
        assert not component.is_degraded()

    def test_is_healthy(self) -> None:
        """is_healthy should check HEALTHY status."""
        component = ComponentHealth(name="test", status=HealthStatus.HEALTHY)
        assert component.is_healthy()

        component.status = HealthStatus.DEGRADED
        assert not component.is_healthy()

    def test_is_unhealthy(self) -> None:
        """is_unhealthy should check UNHEALTHY status."""
        component = ComponentHealth(name="test", status=HealthStatus.UNHEALTHY)
        assert component.is_unhealthy()

        component.status = HealthStatus.HEALTHY
        assert not component.is_unhealthy()


class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_is_healthy(self) -> None:
        """is_healthy should check HEALTHY status."""
        system = SystemHealth(
            status=HealthStatus.HEALTHY,
            components={},
            check_time=datetime.now(UTC),
        )
        assert system.is_healthy()

        system.status = HealthStatus.DEGRADED
        assert not system.is_healthy()


class TestHealthCheck:
    """Tests for HealthCheck manager."""

    def test_check_time_recorded(self) -> None:
        """Health check should record check time."""
        health = HealthCheck()

        before = datetime.now(UTC)
        system_health = health.check_health()
        after = datetime.now(UTC)

        assert before <= system_health.check_time <= after

    def test_failure_count_determines_status(self) -> None:
        """Status should degrade based on error count."""
        health = HealthCheck()

        # First failure - still healthy
        health.record_failure(name="modbus")
        component = health.get_component_health(name="modbus")
        assert component.status == HealthStatus.HEALTHY

        # Second failure - degraded
        health.record_failure(name="modbus")
        component = health.get_component_health(name="modbus")
        assert component.status == HealthStatus.DEGRADED

        # More failures - unhealthy
        for _ in range(3):
            health.record_failure(name="modbus")
        component = health.get_component_health(name="modbus")
        assert component.status == HealthStatus.UNHEALTHY

    def test_health_check_function_called(self) -> None:
        """Health check function should be called during check_health."""
        health = HealthCheck()

        check_called = []

        def modbus_health_check() -> ComponentHealth:
            check_called.append(True)
            return ComponentHealth(name="modbus", status=HealthStatus.HEALTHY)

        health.register_component(name="modbus", health_check=modbus_health_check)

        system_health = health.check_health()

        assert len(check_called) == 1
        assert system_health.components["modbus"].status == HealthStatus.HEALTHY

    def test_health_check_function_error_handled(self) -> None:
        """Error in health check function should be handled gracefully."""
        health = HealthCheck()

        def failing_health_check() -> ComponentHealth:
            raise ValueError("Health check error")

        health.register_component(name="modbus", health_check=failing_health_check)

        # Should not raise
        system_health = health.check_health()

        # Component should record the failure
        component = system_health.components["modbus"]
        assert component.error_count == 1
        assert component.last_failure is not None
        assert "Health check error" in component.message

        # After multiple failures, should be unhealthy
        for _ in range(5):
            health.check_health()

        system_health = health.check_health()
        assert system_health.components["modbus"].status == HealthStatus.UNHEALTHY

    def test_initialization(self) -> None:
        """Health check should initialize with empty components."""
        health = HealthCheck()

        system_health = health.check_health()

        assert system_health.status == HealthStatus.UNKNOWN
        assert len(system_health.components) == 0

    def test_record_failure(self) -> None:
        """Recording failure should update component status."""
        health = HealthCheck()

        health.record_failure(name="modbus", error="Connection timeout")

        component = health.get_component_health(name="modbus")
        assert component is not None
        assert component.last_failure is not None
        assert component.error_count == 1
        assert component.message == "Connection timeout"

    def test_record_success(self) -> None:
        """Recording success should update component status."""
        health = HealthCheck()

        health.record_success(name="modbus")

        component = health.get_component_health(name="modbus")
        assert component is not None
        assert component.status == HealthStatus.HEALTHY
        assert component.last_success is not None
        assert component.error_count == 0

    def test_register_component(self) -> None:
        """Should register component for monitoring."""
        health = HealthCheck()

        health.register_component(name="modbus")
        health.register_component(name="mqtt")

        system_health = health.check_health()

        assert "modbus" in system_health.components
        assert "mqtt" in system_health.components
        assert system_health.components["modbus"].status == HealthStatus.UNKNOWN

    def test_reset_component(self) -> None:
        """Reset should clear component state."""
        health = HealthCheck()

        # Set component to unhealthy
        for _ in range(5):
            health.record_failure(name="modbus")

        component = health.get_component_health(name="modbus")
        assert component.status == HealthStatus.UNHEALTHY

        # Reset
        health.reset_component(name="modbus")

        component = health.get_component_health(name="modbus")
        assert component.status == HealthStatus.UNKNOWN
        assert component.error_count == 0

    def test_stale_component_detection(self) -> None:
        """Components with no recent success should be marked unhealthy."""
        health = HealthCheck(stale_threshold=0.1)  # 100ms threshold

        # Record success
        health.record_success(name="modbus")

        # Wait for stale threshold
        time.sleep(0.15)

        # Check health
        system_health = health.check_health()

        component = system_health.components["modbus"]
        assert component.status == HealthStatus.UNHEALTHY
        assert "Stale" in component.message

    def test_success_resets_error_count(self) -> None:
        """Success should reset error count."""
        health = HealthCheck()

        # Record multiple failures
        for _ in range(5):
            health.record_failure(name="modbus")

        component = health.get_component_health(name="modbus")
        assert component.error_count == 5
        assert component.status == HealthStatus.UNHEALTHY

        # Record success
        health.record_success(name="modbus")

        component = health.get_component_health(name="modbus")
        assert component.error_count == 0
        assert component.status == HealthStatus.HEALTHY

    def test_system_health_includes_uptime(self) -> None:
        """System health should include uptime."""
        health = HealthCheck()

        time.sleep(0.05)

        system_health = health.check_health()

        assert system_health.uptime is not None
        assert system_health.uptime >= timedelta(seconds=0.05)

    def test_system_status_degraded_if_any_component_degraded(self) -> None:
        """System should be degraded if any component is degraded."""
        health = HealthCheck()

        health.update_component_health(name="modbus", status=HealthStatus.HEALTHY)
        health.update_component_health(name="mqtt", status=HealthStatus.DEGRADED)

        system_health = health.check_health()

        assert system_health.status == HealthStatus.DEGRADED
        assert "mqtt" in system_health.message

    def test_system_status_healthy_if_all_components_healthy(self) -> None:
        """System should be healthy if all components are healthy."""
        health = HealthCheck()

        health.update_component_health(name="modbus", status=HealthStatus.HEALTHY)
        health.update_component_health(name="mqtt", status=HealthStatus.HEALTHY)

        system_health = health.check_health()

        assert system_health.status == HealthStatus.HEALTHY
        assert "All components healthy" in system_health.message

    def test_system_status_unhealthy_if_any_component_unhealthy(self) -> None:
        """System should be unhealthy if any component is unhealthy."""
        health = HealthCheck()

        health.update_component_health(name="modbus", status=HealthStatus.HEALTHY)
        health.update_component_health(name="mqtt", status=HealthStatus.UNHEALTHY)

        system_health = health.check_health()

        assert system_health.status == HealthStatus.UNHEALTHY
        assert "mqtt" in system_health.message

    def test_update_component_health(self) -> None:
        """Should manually update component health."""
        health = HealthCheck()

        health.update_component_health(
            name="modbus",
            status=HealthStatus.DEGRADED,
            message="High latency",
            details={"latency_ms": 500},
        )

        component = health.get_component_health(name="modbus")
        assert component is not None
        assert component.status == HealthStatus.DEGRADED
        assert component.message == "High latency"
        assert component.details["latency_ms"] == 500

    def test_uptime_tracking(self) -> None:
        """Should track system uptime."""
        health = HealthCheck()

        time.sleep(0.1)

        uptime = health.get_uptime()
        assert uptime >= timedelta(seconds=0.1)
        assert uptime < timedelta(seconds=1.0)


class TestHealthCheckIntegration:
    """Integration tests for health check system."""

    def test_custom_health_check_functions(self) -> None:
        """Test using custom health check functions."""
        health = HealthCheck()

        # Simulate external state
        modbus_connected = True
        mqtt_connected = True

        def modbus_health_check() -> ComponentHealth:
            if modbus_connected:
                return ComponentHealth(name="modbus", status=HealthStatus.HEALTHY)
            return ComponentHealth(
                name="modbus",
                status=HealthStatus.UNHEALTHY,
                message="Disconnected",
            )

        def mqtt_health_check() -> ComponentHealth:
            if mqtt_connected:
                return ComponentHealth(name="mqtt", status=HealthStatus.HEALTHY)
            return ComponentHealth(
                name="mqtt",
                status=HealthStatus.UNHEALTHY,
                message="Disconnected",
            )

        health.register_component(name="modbus", health_check=modbus_health_check)
        health.register_component(name="mqtt", health_check=mqtt_health_check)

        # All healthy
        system_health = health.check_health()
        assert system_health.is_healthy()

        # Modbus disconnects
        modbus_connected = False
        system_health = health.check_health()
        assert system_health.status == HealthStatus.UNHEALTHY

        # Modbus reconnects
        modbus_connected = True
        system_health = health.check_health()
        assert system_health.is_healthy()

    def test_typical_monitoring_scenario(self) -> None:
        """Test typical component monitoring scenario."""
        health = HealthCheck()

        # Register components
        health.register_component(name="modbus")
        health.register_component(name="mqtt")

        # Record some successes
        health.record_success(name="modbus")
        health.record_success(name="mqtt")

        # Check health - should be healthy
        system_health = health.check_health()
        assert system_health.is_healthy()

        # Modbus starts failing
        for _ in range(5):
            health.record_failure(name="modbus")

        # Check health - should be unhealthy
        system_health = health.check_health()
        assert system_health.status == HealthStatus.UNHEALTHY
        assert "modbus" in system_health.message

        # Modbus recovers
        health.record_success(name="modbus")

        # Check health - should be healthy again
        system_health = health.check_health()
        assert system_health.is_healthy()
