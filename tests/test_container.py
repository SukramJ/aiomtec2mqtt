"""Tests for dependency injection container."""

from __future__ import annotations

import pytest

from aiomtec2mqtt.container import ServiceContainer, create_container
from aiomtec2mqtt.health import HealthCheck


class DummyService:
    """Dummy service for testing."""

    def __init__(self, value: int = 42) -> None:
        """Initialize dummy service."""
        self.value = value


class TestServiceContainer:
    """Test ServiceContainer class."""

    def test_clear_removes_all_services(self) -> None:
        """Test clear removes all registrations."""
        container = ServiceContainer()
        container.register_singleton(service_type=DummyService, instance=DummyService())

        container.clear()

        assert not container.is_registered(service_type=DummyService)

    def test_is_registered(self) -> None:
        """Test is_registered check."""
        container = ServiceContainer()

        assert not container.is_registered(service_type=DummyService)

        container.register_singleton(service_type=DummyService, instance=DummyService())

        assert container.is_registered(service_type=DummyService)

    def test_overwrite_factory_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test overwriting factory logs warning."""
        container = ServiceContainer()

        def factory1() -> DummyService:
            return DummyService(1)

        def factory2() -> DummyService:
            return DummyService(2)

        container.register_factory(service_type=DummyService, factory=factory1)
        container.register_factory(service_type=DummyService, factory=factory2)

        assert "Overwriting" in caplog.text

    def test_overwrite_singleton_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test overwriting singleton logs warning."""
        container = ServiceContainer()
        container.register_singleton(service_type=DummyService, instance=DummyService(1))
        container.register_singleton(service_type=DummyService, instance=DummyService(2))

        assert "Overwriting" in caplog.text

    def test_register_and_resolve_factory(self) -> None:
        """Test factory registration and resolution."""
        container = ServiceContainer()

        def factory() -> DummyService:
            return DummyService(200)

        container.register_factory(service_type=DummyService, factory=factory)
        resolved1 = container.resolve(service_type=DummyService)
        resolved2 = container.resolve(service_type=DummyService)

        # Factory creates one instance, then returns same instance
        assert resolved1 is resolved2
        assert resolved1.value == 200

    def test_register_and_resolve_singleton(self) -> None:
        """Test singleton registration and resolution."""
        container = ServiceContainer()
        service = DummyService(100)

        container.register_singleton(service_type=DummyService, instance=service)
        resolved = container.resolve(service_type=DummyService)

        assert resolved is service
        assert resolved.value == 100

    def test_reset_instances_keeps_registrations(self) -> None:
        """Test reset_instances keeps factory registrations."""
        container = ServiceContainer()

        def factory() -> DummyService:
            return DummyService(300)

        container.register_factory(service_type=DummyService, factory=factory)

        # Resolve to create instance
        first = container.resolve(service_type=DummyService)
        assert first.value == 300

        # Reset instances
        container.reset_instances()

        # Should still be registered
        assert container.is_registered(service_type=DummyService)

        # Resolve creates new instance
        second = container.resolve(service_type=DummyService)
        assert second is not first

    def test_resolve_unregistered_service_raises_error(self) -> None:
        """Test resolving unregistered service raises KeyError."""
        container = ServiceContainer()

        with pytest.raises(KeyError) as exc_info:
            container.resolve(service_type=DummyService)

        assert "not registered" in str(exc_info.value)


class TestCreateContainer:
    """Test create_container factory function."""

    def test_create_container_with_config(self) -> None:
        """Test creating container with full configuration."""
        from aiomtec2mqtt.async_modbus_client import AsyncModbusClient
        from aiomtec2mqtt.async_mqtt_client import AsyncMqttClient
        from aiomtec2mqtt.const import Config

        config = {
            Config.MODBUS_IP: "192.168.1.100",
            Config.MODBUS_PORT: 502,
            Config.MODBUS_SLAVE: 1,
            Config.MODBUS_TIMEOUT: 5,
            Config.MQTT_SERVER: "mqtt.example.com",
            Config.MQTT_PORT: 1883,
            Config.MQTT_LOGIN: "user",
            Config.MQTT_PASSWORD: "pass",
            Config.MQTT_TOPIC: "MTEC/12345",
            Config.MQTT_FLOAT_FORMAT: ".2f",
            Config.HASS_BASE_TOPIC: "homeassistant",
            Config.HASS_ENABLE: False,
            Config.HASS_BIRTH_GRACETIME: 0,
        }

        register_map = {
            "10100": {
                "name": "battery_soc",
                "unit": "%",
                "group": "BASE",
                "scale": 1,
            },
        }

        register_groups = ["BASE"]

        container = create_container(
            config=config, register_map=register_map, register_groups=register_groups
        )

        # Should have all services
        assert container.is_registered(service_type=HealthCheck)
        assert container.is_registered(service_type=AsyncModbusClient)
        assert container.is_registered(service_type=AsyncMqttClient)

    def test_create_container_without_config(self) -> None:
        """Test creating container without configuration."""
        container = create_container()

        # Should have health check
        assert container.is_registered(service_type=HealthCheck)
