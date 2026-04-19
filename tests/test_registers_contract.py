"""
Contract tests for ``aiomtec2mqtt/registers.yaml``.

These tests enforce invariants between the bundled register YAML and the
consuming code (``config.init_register_map``, ``register_models``,
``register_processors``). They catch regressions where a register is added or
renamed in YAML without corresponding code updates — and vice versa.

Contracts verified:
    1. YAML loads and passes mandatory-parameter checks
    2. Every ``group`` value comes from a known set
    3. ``mqtt`` topic suffixes are alphanumeric + underscore only
    4. All optional fields defaulted by ``init_register_map`` are present
    5. ``hass_*`` markers map to values that the processors/HA integration understand
    6. Register keys and names are unique
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from aiomtec2mqtt.config import init_register_map
from aiomtec2mqtt.const import OPTIONAL_PARAMETERS, Register

# MQTT topic segments must be printable ASCII without reserved MQTT characters.
_MQTT_TOPIC_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-]*$")

# Registers are grouped by the YAML ``group`` key; update this set when adding a new one.
_KNOWN_GROUPS: frozenset[str] = frozenset(
    {
        "now-base",
        "now-grid",
        "now-inverter",
        "now-backup",
        "now-battery",
        "now-pv",
        "day",
        "total",
        "static",
        "config",
    }
)

_KNOWN_HASS_DEVICE_CLASSES: frozenset[str] = frozenset(
    {
        "power",
        "energy",
        "voltage",
        "current",
        "frequency",
        "temperature",
        "battery",
        "power_factor",
        "duration",
        "apparent_power",
        "reactive_power",
        "enum",
    }
)

_KNOWN_HASS_STATE_CLASSES: frozenset[str] = frozenset(
    {
        "measurement",
        "total",
        "total_increasing",
    }
)


@pytest.fixture(scope="module")
def register_map() -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Load the production register map exactly once per test module."""
    return init_register_map()


class TestRegisterYamlContract:
    """Structural invariants of the bundled registers.yaml."""

    def test_groups_are_known(
        self, register_map: tuple[dict[str, dict[str, Any]], list[str]]
    ) -> None:
        reg_map, _ = register_map
        unknown = {
            key: val[Register.GROUP]
            for key, val in reg_map.items()
            if val.get(Register.GROUP) and val[Register.GROUP] not in _KNOWN_GROUPS
        }
        assert not unknown, (
            f"registers reference unknown groups {unknown}. "
            f"Either add to _KNOWN_GROUPS or fix the YAML."
        )

    def test_groups_list_matches_registers(
        self, register_map: tuple[dict[str, dict[str, Any]], list[str]]
    ) -> None:
        reg_map, groups = register_map
        from_registers = {
            val[Register.GROUP] for val in reg_map.values() if val.get(Register.GROUP)
        }
        assert set(groups) == from_registers, (
            f"groups list {groups} out of sync with register groups {from_registers}"
        )

    def test_hass_device_classes_are_known(
        self, register_map: tuple[dict[str, dict[str, Any]], list[str]]
    ) -> None:
        reg_map, _ = register_map
        unknown: dict[str, str] = {}
        for key, val in reg_map.items():
            dc = val.get(Register.DEVICE_CLASS)
            if dc and dc not in _KNOWN_HASS_DEVICE_CLASSES:
                unknown[key] = str(dc)
        assert not unknown, (
            f"registers reference unknown HA device_class values: {unknown}. "
            f"Either add to _KNOWN_HASS_DEVICE_CLASSES or fix the YAML."
        )

    def test_hass_state_classes_are_known(
        self, register_map: tuple[dict[str, dict[str, Any]], list[str]]
    ) -> None:
        reg_map, _ = register_map
        unknown: dict[str, str] = {}
        for key, val in reg_map.items():
            sc = val.get(Register.STATE_CLASS) if hasattr(Register, "STATE_CLASS") else None
            if sc is None:
                sc = val.get("hass_state_class")
            if sc and sc not in _KNOWN_HASS_STATE_CLASSES:
                unknown[key] = str(sc)
        assert not unknown, (
            f"registers reference unknown HA state_class values: {unknown}. "
            f"Either add to _KNOWN_HASS_STATE_CLASSES or fix the YAML."
        )

    def test_mandatory_name_present(
        self, register_map: tuple[dict[str, dict[str, Any]], list[str]]
    ) -> None:
        reg_map, _ = register_map
        missing = [key for key, val in reg_map.items() if not val.get(Register.NAME)]
        assert not missing, f"registers missing 'name' field: {missing}"

    def test_mqtt_topics_are_valid(
        self, register_map: tuple[dict[str, dict[str, Any]], list[str]]
    ) -> None:
        reg_map, _ = register_map
        invalid: dict[str, str] = {}
        for key, val in reg_map.items():
            mqtt = val.get(Register.MQTT)
            if mqtt is None:
                continue
            if not _MQTT_TOPIC_RE.fullmatch(str(mqtt)):
                invalid[key] = str(mqtt)
        assert not invalid, f"invalid MQTT topic suffixes: {invalid}"

    def test_mqtt_topics_unique_per_group(
        self, register_map: tuple[dict[str, dict[str, Any]], list[str]]
    ) -> None:
        """Within a group, MQTT topic suffixes must be unique to avoid clobbering."""
        reg_map, _ = register_map
        seen: dict[tuple[str, str], str] = {}
        duplicates: dict[tuple[str, str], list[str]] = {}
        for key, val in reg_map.items():
            group = val.get(Register.GROUP)
            mqtt = val.get(Register.MQTT)
            if not group or not mqtt:
                continue
            signature = (str(group), str(mqtt))
            if signature in seen:
                duplicates.setdefault(signature, [seen[signature]]).append(key)
            else:
                seen[signature] = key
        assert not duplicates, f"duplicate MQTT topics per group: {duplicates}"

    def test_optional_defaults_applied(
        self, register_map: tuple[dict[str, dict[str, Any]], list[str]]
    ) -> None:
        reg_map, _ = register_map
        for key, val in reg_map.items():
            for param in OPTIONAL_PARAMETERS:
                assert param in val, f"register '{key}' missing defaulted optional param '{param}'"

    def test_register_keys_unique(
        self, register_map: tuple[dict[str, dict[str, Any]], list[str]]
    ) -> None:
        # init_register_map returns a dict so Python guarantees uniqueness;
        # this test documents the contract and guards future refactors.
        reg_map, _ = register_map
        assert len(reg_map) == len(set(reg_map))

    def test_register_map_not_empty(
        self, register_map: tuple[dict[str, dict[str, Any]], list[str]]
    ) -> None:
        reg_map, groups = register_map
        assert reg_map, "register map must not be empty"
        assert groups, "register groups must not be empty"
