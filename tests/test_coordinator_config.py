"""Tests for :mod:`aiomtec2mqtt.coordinator_config`."""

from __future__ import annotations

import pytest

from aiomtec2mqtt.config_schema import ConfigValidationError
from aiomtec2mqtt.const import Config
from aiomtec2mqtt.coordinator_config import CoordinatorConfig, CoordinatorConfigBuilder


class TestCoordinatorConfigBuilder:
    """Happy-path and validation behaviour for the fluent builder."""

    def test_builder_builds_full_config(self) -> None:
        cfg = (
            CoordinatorConfigBuilder()
            .with_modbus(ip="192.168.1.20", port=502, slave=1, timeout=5)
            .with_mqtt(server="localhost", port=1883, topic="MTEC")
            .with_home_assistant(enabled=True, base_topic="homeassistant")
            .with_refresh(now=5)
            .with_debug(enabled=True)
            .build()
        )
        assert isinstance(cfg, CoordinatorConfig)
        assert cfg.modbus_ip == "192.168.1.20"
        assert cfg.mqtt_topic == "MTEC"
        assert cfg.hass_enable is True
        assert cfg.refresh_now == 5
        assert cfg.debug is True
        # Defaults kicked in for fields the caller never touched:
        assert cfg.modbus_framer == "rtu"
        assert cfg.mqtt_float_format == ".3f"
        assert cfg.refresh_config == 30

    def test_builder_invalid_value_raises(self) -> None:
        builder = (
            CoordinatorConfigBuilder()
            .with_modbus(ip="10.0.0.1", port=999_999, slave=1, timeout=5)
            .with_mqtt(server="localhost", port=1883, topic="MTEC")
        )
        with pytest.raises(ConfigValidationError, match="MODBUS_PORT"):
            builder.build()

    def test_builder_missing_field_raises(self) -> None:
        builder = CoordinatorConfigBuilder().with_modbus(
            ip="10.0.0.1", port=502, slave=1, timeout=5
        )
        with pytest.raises(ConfigValidationError, match="MQTT_SERVER"):
            builder.build()

    def test_from_dict_seeds_and_overrides(self) -> None:
        seed = {
            "MODBUS_IP": "10.0.0.1",
            "MODBUS_PORT": 5743,
            "MODBUS_SLAVE": 1,
            "MODBUS_TIMEOUT": 5,
            "MQTT_SERVER": "broker.example.com",
            "MQTT_PORT": 1883,
            "MQTT_TOPIC": "MTEC",
        }
        cfg = (
            CoordinatorConfigBuilder.from_dict(seed)
            .with_modbus(port=502)  # override
            .build()
        )
        assert cfg.modbus_port == 502
        assert cfg.mqtt_server == "broker.example.com"

    def test_none_arguments_are_ignored(self) -> None:
        """``with_*`` calls with all-None kwargs must not clobber previous state."""
        cfg = (
            CoordinatorConfigBuilder()
            .with_modbus(ip="10.0.0.1", port=502, slave=1, timeout=5)
            .with_modbus(ip=None)  # must not clear previous ip
            .with_mqtt(server="localhost", port=1883, topic="MTEC")
            .build()
        )
        assert cfg.modbus_ip == "10.0.0.1"

    def test_roundtrip_via_as_dict(self) -> None:
        cfg = (
            CoordinatorConfigBuilder()
            .with_modbus(ip="10.0.0.1", port=502, slave=1, timeout=5)
            .with_mqtt(server="localhost", port=1883, topic="MTEC")
            .build()
        )
        as_dict = cfg.as_dict()
        assert as_dict[Config.MODBUS_IP] == "10.0.0.1"
        assert as_dict[Config.MQTT_TOPIC] == "MTEC"

        rebuilt = CoordinatorConfigBuilder.from_config(cfg).build()
        assert rebuilt == cfg
