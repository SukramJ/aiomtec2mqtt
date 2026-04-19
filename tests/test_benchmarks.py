"""Performance benchmarks for hot paths.

These benchmarks pin the cost of the per-register code paths that run on
every Modbus poll cycle. They are skipped during normal ``pytest`` runs
(see ``addopts = -m 'not benchmark'``); enable them explicitly with::

    pytest -p no:xdist -m benchmark tests/test_benchmarks.py

``pytest-benchmark`` is incompatible with ``pytest-xdist`` — the
``-p no:xdist`` flag is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiomtec2mqtt import _json
from aiomtec2mqtt.const import Config, Register, RegisterGroup

pytestmark = pytest.mark.benchmark


def _config() -> dict[Config, Any]:
    return {
        Config.MODBUS_IP: "1.2.3.4",
        Config.MODBUS_PORT: 502,
        Config.MODBUS_SLAVE: 1,
        Config.MODBUS_TIMEOUT: 5,
        Config.MQTT_SERVER: "mqtt",
        Config.MQTT_PORT: 1883,
        Config.MQTT_LOGIN: "",
        Config.MQTT_PASSWORD: "",
        Config.MQTT_TOPIC: "MTEC",
        Config.MQTT_FLOAT_FORMAT: ".3f",
        Config.HASS_BASE_TOPIC: "homeassistant",
        Config.HASS_ENABLE: False,
        Config.HASS_BIRTH_GRACETIME: 0,
        Config.DEBUG: False,
        Config.REFRESH_NOW: 1,
        Config.REFRESH_DAY: 10,
        Config.REFRESH_STATIC: 3600,
        Config.REFRESH_CONFIG: 60,
        Config.REFRESH_TOTAL: 300,
    }


def _register_map() -> dict[str, dict[str, Any]]:
    return {
        "11000": {
            Register.NAME: "Grid power",
            Register.MQTT: "grid_power",
            Register.GROUP: RegisterGroup.BASE,
        },
        "11016": {
            Register.NAME: "Inverter AC power",
            Register.MQTT: "inverter_ac_power",
            Register.GROUP: RegisterGroup.BASE,
        },
        "12000": {
            Register.NAME: "Inverter status",
            Register.MQTT: "status",
            Register.GROUP: RegisterGroup.BASE,
            Register.DEVICE_CLASS: "enum",
            Register.VALUE_ITEMS: {0: "Idle", 1: "Running", 2: "Fault"},
        },
    }


@pytest.fixture
def coordinator():
    """A coordinator with mocked I/O — only pure helpers exercised."""
    from aiomtec2mqtt.async_coordinator import AsyncMtecCoordinator

    with (
        patch("aiomtec2mqtt.async_coordinator.init_config", return_value=_config()),
        patch(
            "aiomtec2mqtt.async_coordinator.init_register_map",
            return_value=(_register_map(), [RegisterGroup.BASE]),
        ),
    ):
        coord = AsyncMtecCoordinator()
    coord._topic_base = "MTEC/SN123"
    coord._serial_no = "SN123"
    coord._mqtt_client = MagicMock()
    coord._mqtt_client.publish = AsyncMock()
    coord._modbus_client = MagicMock()
    return coord


# ---------------------------------------------------------------------------
# Coordinator hot paths
# ---------------------------------------------------------------------------


def test_format_value_float(benchmark, coordinator) -> None:
    """Per-register float formatting — runs once per published value."""
    benchmark(coordinator._format_value, value=1234.5678)


def test_format_value_bool(benchmark, coordinator) -> None:
    benchmark(coordinator._format_value, value=True)


def test_format_value_int(benchmark, coordinator) -> None:
    benchmark(coordinator._format_value, value=42)


def test_process_register_value_passthrough(benchmark, coordinator) -> None:
    """Hot path for non-special registers (vast majority)."""
    benchmark(
        coordinator._process_register_value,
        register_addr="11000",
        value=42,
        reg_info={},
    )


def test_process_register_value_enum(benchmark, coordinator) -> None:
    """Enum conversion — every status register triggers this."""
    reg_info = {
        Register.DEVICE_CLASS: "enum",
        Register.VALUE_ITEMS: {0: "Idle", 1: "Running", 2: "Fault"},
    }
    benchmark(
        coordinator._process_register_value,
        register_addr="12000",
        value=1,
        reg_info=reg_info,
    )


def test_convert_code_int(benchmark, coordinator) -> None:
    benchmark(
        coordinator._convert_code,
        value=1,
        value_items={0: "Idle", 1: "Running", 2: "Fault"},
    )


def test_convert_code_binary_string(benchmark, coordinator) -> None:
    """Bitmask faults — used by alarm/status registers."""
    benchmark(
        coordinator._convert_code,
        value="10101010",
        value_items={i: f"Fault{i}" for i in range(8)},
    )


# ---------------------------------------------------------------------------
# JSON helper hot paths
# ---------------------------------------------------------------------------


def test_json_dumps_small(benchmark) -> None:
    payload = {"x": 1.5, "y": "ok", "z": True, "nested": {"a": 1, "b": 2}}
    benchmark(_json.dumps, payload)


def test_json_dumps_typical_payload(benchmark) -> None:
    """Mirror the shape of a now-base MQTT payload (12 keys, mixed types)."""
    payload = {
        "grid_power": -250.5,
        "inverter_ac_power": 1234.0,
        "battery_soc": 85.0,
        "battery_power": 100.5,
        "pv_power": 1500.5,
        "consumption": 800.25,
        "phase_a_voltage": 230.1,
        "phase_b_voltage": 230.5,
        "phase_c_voltage": 229.9,
        "frequency": 50.0,
        "status": "Running",
        "api_date": "2026-04-19 12:00:00",
    }
    benchmark(_json.dumps, payload)


def test_json_loads_small(benchmark) -> None:
    raw = b'{"x":1.5,"y":"ok","z":true,"nested":{"a":1,"b":2}}'
    benchmark(_json.loads, raw)
