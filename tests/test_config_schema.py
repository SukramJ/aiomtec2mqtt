"""Tests for :mod:`aiomtec2mqtt.config_schema`."""

from __future__ import annotations

from typing import Any

import pytest

from aiomtec2mqtt.config_schema import ConfigValidationError, validate_config


@pytest.fixture
def valid_config() -> dict[str, Any]:
    """Return a minimal valid configuration dict."""
    return {
        "MODBUS_IP": "192.168.1.100",
        "MODBUS_PORT": 502,
        "MODBUS_SLAVE": 1,
        "MODBUS_TIMEOUT": 5,
        "MQTT_SERVER": "localhost",
        "MQTT_PORT": 1883,
        "MQTT_TOPIC": "MTEC",
    }


class TestValidateConfig:
    """Validation happy-path and defaults."""

    def test_applies_defaults(self, valid_config: dict[str, Any]) -> None:
        result = validate_config(valid_config)
        assert result["MODBUS_FRAMER"] == "rtu"
        assert result["MODBUS_RETRIES"] == 3
        assert result["HASS_ENABLE"] is False
        assert result["REFRESH_NOW"] == 10

    def test_preserves_extra_keys(self, valid_config: dict[str, Any]) -> None:
        valid_config["CUSTOM_KEY"] = "custom_value"
        result = validate_config(valid_config)
        assert result["CUSTOM_KEY"] == "custom_value"

    def test_template_style_float_format(self, valid_config: dict[str, Any]) -> None:
        valid_config["MQTT_FLOAT_FORMAT"] = "{:.1f}"
        result = validate_config(valid_config)
        assert result["MQTT_FLOAT_FORMAT"] == "{:.1f}"


class TestValidationErrors:
    """Ensures invalid inputs produce aggregated validation errors."""

    def test_invalid_float_format(self, valid_config: dict[str, Any]) -> None:
        valid_config["MQTT_FLOAT_FORMAT"] = "not-a-spec!"
        with pytest.raises(ConfigValidationError, match="MQTT_FLOAT_FORMAT"):
            validate_config(valid_config)

    def test_invalid_framer(self, valid_config: dict[str, Any]) -> None:
        valid_config["MODBUS_FRAMER"] = "bogus"
        with pytest.raises(ConfigValidationError, match="MODBUS_FRAMER"):
            validate_config(valid_config)

    def test_missing_mandatory_field(self, valid_config: dict[str, Any]) -> None:
        del valid_config["MODBUS_IP"]
        with pytest.raises(ConfigValidationError, match="MODBUS_IP"):
            validate_config(valid_config)

    def test_negative_refresh(self, valid_config: dict[str, Any]) -> None:
        valid_config["REFRESH_NOW"] = -1
        with pytest.raises(ConfigValidationError, match="REFRESH_NOW"):
            validate_config(valid_config)

    def test_port_out_of_range(self, valid_config: dict[str, Any]) -> None:
        valid_config["MODBUS_PORT"] = 70_000
        with pytest.raises(ConfigValidationError, match="MODBUS_PORT"):
            validate_config(valid_config)
