"""Extended branch tests for :mod:`aiomtec2mqtt.coordinator_config`.

Complements :mod:`tests.test_coordinator_config` by covering edge cases of
each ``with_*`` method, defaulting behaviour, and the ``as_dict`` ↔
``from_dict`` round trip.
"""

from __future__ import annotations

import pytest

from aiomtec2mqtt.config_schema import ConfigValidationError
from aiomtec2mqtt.const import Config
from aiomtec2mqtt.coordinator_config import CoordinatorConfigBuilder


def _minimal_required() -> dict[str, object]:
    """The smallest dict that satisfies :class:`ConfigSchema`."""
    return {
        "MODBUS_IP": "10.0.0.1",
        "MODBUS_PORT": 502,
        "MODBUS_SLAVE": 1,
        "MODBUS_TIMEOUT": 5,
        "MQTT_SERVER": "localhost",
        "MQTT_PORT": 1883,
        "MQTT_TOPIC": "MTEC",
    }


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    """Schema defaults must propagate to the dataclass."""

    def test_debug_defaults_false(self) -> None:
        cfg = CoordinatorConfigBuilder.from_dict(_minimal_required()).build()
        assert cfg.debug is False

    def test_hass_disabled_by_default(self) -> None:
        cfg = CoordinatorConfigBuilder.from_dict(_minimal_required()).build()
        assert cfg.hass_enable is False
        assert cfg.hass_base_topic == "homeassistant"
        assert cfg.hass_birth_gracetime == 15

    def test_modbus_framer_defaults_to_rtu(self) -> None:
        cfg = CoordinatorConfigBuilder.from_dict(_minimal_required()).build()
        assert cfg.modbus_framer == "rtu"

    def test_modbus_retries_default(self) -> None:
        cfg = CoordinatorConfigBuilder.from_dict(_minimal_required()).build()
        assert cfg.modbus_retries == 3

    def test_mqtt_login_defaults_empty(self) -> None:
        cfg = CoordinatorConfigBuilder.from_dict(_minimal_required()).build()
        assert cfg.mqtt_login == ""
        assert cfg.mqtt_password == ""

    def test_refresh_intervals_defaults(self) -> None:
        cfg = CoordinatorConfigBuilder.from_dict(_minimal_required()).build()
        assert cfg.refresh_now == 10
        assert cfg.refresh_config == 30
        assert cfg.refresh_day == 300
        assert cfg.refresh_static == 3600
        assert cfg.refresh_total == 300


# ---------------------------------------------------------------------------
# Builder method behaviour
# ---------------------------------------------------------------------------


class TestBuilderModbus:
    """``with_modbus`` semantics."""

    def test_invalid_framer_rejected(self) -> None:
        builder = CoordinatorConfigBuilder.from_dict(_minimal_required()).with_modbus(
            framer="invalid"
        )
        with pytest.raises(ConfigValidationError, match="MODBUS_FRAMER"):
            builder.build()

    def test_partial_update_keeps_other_fields(self) -> None:
        cfg = (
            CoordinatorConfigBuilder.from_dict(_minimal_required())
            .with_modbus(framer="socket")
            .build()
        )
        assert cfg.modbus_framer == "socket"
        assert cfg.modbus_ip == "10.0.0.1"

    def test_retries_clamped_to_schema_limits(self) -> None:
        builder = CoordinatorConfigBuilder.from_dict(_minimal_required()).with_modbus(retries=21)
        with pytest.raises(ConfigValidationError, match="MODBUS_RETRIES"):
            builder.build()

    def test_slave_lower_bound(self) -> None:
        builder = CoordinatorConfigBuilder.from_dict(_minimal_required()).with_modbus(slave=-1)
        with pytest.raises(ConfigValidationError, match="MODBUS_SLAVE"):
            builder.build()

    def test_slave_upper_bound(self) -> None:
        builder = CoordinatorConfigBuilder.from_dict(_minimal_required()).with_modbus(slave=248)
        with pytest.raises(ConfigValidationError, match="MODBUS_SLAVE"):
            builder.build()


class TestBuilderMqtt:
    """``with_mqtt`` semantics."""

    def test_braced_float_format_accepted(self) -> None:
        cfg = (
            CoordinatorConfigBuilder.from_dict(_minimal_required())
            .with_mqtt(float_format="{:.4f}")
            .build()
        )
        assert cfg.mqtt_float_format == "{:.4f}"

    def test_credentials_propagate(self) -> None:
        cfg = (
            CoordinatorConfigBuilder.from_dict(_minimal_required())
            .with_mqtt(login="user", password="secret")
            .build()
        )
        assert cfg.mqtt_login == "user"
        assert cfg.mqtt_password == "secret"

    def test_invalid_float_format_rejected(self) -> None:
        builder = CoordinatorConfigBuilder.from_dict(_minimal_required()).with_mqtt(
            float_format=".XYZf"
        )
        with pytest.raises(ConfigValidationError, match="MQTT_FLOAT_FORMAT"):
            builder.build()

    def test_port_upper_bound(self) -> None:
        builder = CoordinatorConfigBuilder.from_dict(_minimal_required()).with_mqtt(port=65536)
        with pytest.raises(ConfigValidationError, match="MQTT_PORT"):
            builder.build()


class TestBuilderHassAndRefresh:
    """``with_home_assistant`` and ``with_refresh`` semantics."""

    def test_all_refresh_fields_settable_in_one_call(self) -> None:
        cfg = (
            CoordinatorConfigBuilder.from_dict(_minimal_required())
            .with_refresh(now=1, config=2, day=3, static=4, total=5)
            .build()
        )
        assert (
            cfg.refresh_now,
            cfg.refresh_config,
            cfg.refresh_day,
            cfg.refresh_static,
            cfg.refresh_total,
        ) == (1, 2, 3, 4, 5)

    def test_birth_gracetime_upper_bound(self) -> None:
        builder = CoordinatorConfigBuilder.from_dict(_minimal_required()).with_home_assistant(
            birth_gracetime=601
        )
        with pytest.raises(ConfigValidationError, match="HASS_BIRTH_GRACETIME"):
            builder.build()

    def test_refresh_now_lower_bound(self) -> None:
        builder = CoordinatorConfigBuilder.from_dict(_minimal_required()).with_refresh(now=0)
        with pytest.raises(ConfigValidationError, match="REFRESH_NOW"):
            builder.build()

    def test_refresh_static_upper_bound(self) -> None:
        builder = CoordinatorConfigBuilder.from_dict(_minimal_required()).with_refresh(
            static=86401
        )
        with pytest.raises(ConfigValidationError, match="REFRESH_STATIC"):
            builder.build()


# ---------------------------------------------------------------------------
# Immutability + equality
# ---------------------------------------------------------------------------


class TestImmutability:
    """``CoordinatorConfig`` is a frozen dataclass."""

    def test_assignment_is_blocked(self) -> None:
        cfg = CoordinatorConfigBuilder.from_dict(_minimal_required()).build()
        with pytest.raises((AttributeError, TypeError)):
            cfg.modbus_ip = "10.0.0.99"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        a = CoordinatorConfigBuilder.from_dict(_minimal_required()).build()
        b = CoordinatorConfigBuilder.from_dict(_minimal_required()).build()
        assert a == b
        assert hash(a) == hash(b)

    def test_inequality_with_different_values(self) -> None:
        a = CoordinatorConfigBuilder.from_dict(_minimal_required()).build()
        b = CoordinatorConfigBuilder.from_dict(_minimal_required()).with_mqtt(port=8883).build()
        assert a != b


# ---------------------------------------------------------------------------
# from_config / as_dict round trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """``as_dict`` ↔ ``from_dict`` should preserve every field."""

    def test_from_config_clones(self) -> None:
        original = CoordinatorConfigBuilder.from_dict(_minimal_required()).build()
        clone = CoordinatorConfigBuilder.from_config(original).build()
        assert clone == original
        assert clone is not original

    def test_from_config_then_override(self) -> None:
        original = CoordinatorConfigBuilder.from_dict(_minimal_required()).build()
        modified = CoordinatorConfigBuilder.from_config(original).with_debug(enabled=True).build()
        assert modified.debug is True
        assert original.debug is False

    def test_full_roundtrip_preserves_all_fields(self) -> None:
        original = (
            CoordinatorConfigBuilder.from_dict(_minimal_required())
            .with_modbus(framer="socket", retries=5)
            .with_mqtt(login="u", password="p", float_format="{:.4f}")
            .with_home_assistant(enabled=True, base_topic="ha", birth_gracetime=30)
            .with_refresh(now=2, config=4, day=8, static=16, total=32)
            .with_debug(enabled=True)
            .build()
        )
        as_dict = original.as_dict()
        # `as_dict()` keys are Config enum members, not strings — convert.
        rebuild_seed = {
            str(k.value if isinstance(k, Config) else k): v for k, v in as_dict.items()
        }
        rebuilt = CoordinatorConfigBuilder.from_dict(rebuild_seed).build()
        assert rebuilt == original


# ---------------------------------------------------------------------------
# Error aggregation
# ---------------------------------------------------------------------------


class TestErrorAggregation:
    """Multi-field errors are collected by Pydantic and surfaced together."""

    def test_multiple_errors_in_single_message(self) -> None:
        builder = CoordinatorConfigBuilder().with_modbus(
            ip="10.0.0.1", port=999_999, slave=300, timeout=5
        )
        with pytest.raises(ConfigValidationError) as excinfo:
            builder.build()
        message = str(excinfo.value)
        assert "MODBUS_PORT" in message
        assert "MODBUS_SLAVE" in message

    def test_unrelated_extra_keys_preserved(self) -> None:
        seed = {**_minimal_required(), "EXTRA_KEY": "value"}
        cfg = CoordinatorConfigBuilder.from_dict(seed).build()
        # Extra keys do not crash validation thanks to ``extra="allow"``.
        assert cfg.modbus_ip == "10.0.0.1"
