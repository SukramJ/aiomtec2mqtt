"""Extended branch + edge-case tests for :class:`HassIntegration`.

The original :mod:`tests.test_hass_int` is one happy-path smoke test. This
module adds focused tests for individual branches in ``hass_int.py`` so
regressions in the discovery payload structure are caught quickly.

Naming follows the pattern ``test_<entity>_<aspect>`` to keep failures
self-explanatory.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aiomtec2mqtt.const import HA, MTEC_PREFIX, HAPlatform, Register
from aiomtec2mqtt.hass_int import HassIntegration


class _RecordingMqtt:
    """MQTT stub that records publishes/subscribes via keyword-only args."""

    def __init__(self) -> None:
        self.published: list[tuple[str, str, bool]] = []
        self.subscribed: list[str] = []

    def publish(self, *, topic: str, payload: str, retain: bool = False) -> None:
        self.published.append((topic, payload, retain))

    def subscribe_to_topic(self, *, topic: str) -> None:
        self.subscribed.append(topic)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_integration(register_map: dict[str, dict[str, Any]]) -> HassIntegration:
    return HassIntegration(
        hass_base_topic="homeassistant",
        mqtt_topic="MTEC",
        register_map=register_map,
    )


def _initialise(
    register_map: dict[str, dict[str, Any]],
    mqtt: _RecordingMqtt | None = None,
) -> tuple[HassIntegration, _RecordingMqtt | None]:
    integration = _make_integration(register_map)
    integration.initialize(
        mqtt=mqtt,
        serial_no="SN-TEST",
        firmware_version="V99.0",
        equipment_info="MyEquipment",
    )
    return integration, mqtt


def _config_payload(integration: HassIntegration, topic_suffix: str) -> dict[str, Any]:
    """Return the deserialised payload of the entity whose topic ends with the suffix."""
    for topic, payload, _command in integration._devices_array:  # noqa: SLF001
        if topic.endswith(topic_suffix):
            return json.loads(payload)
    raise AssertionError(f"no entity with suffix {topic_suffix!r}")


# ---------------------------------------------------------------------------
# Initialisation lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Initialisation, idempotency and the is_initialized flag."""

    def test_initialize_clears_devices_array(self) -> None:
        # Two consecutive initialize calls must not duplicate entries.
        register_map = {
            "11000": {
                Register.NAME: "Power",
                Register.GROUP: "now-base",
                Register.MQTT: "power",
                Register.DEVICE_CLASS: "power",
            },
        }
        integration, _ = _initialise(register_map, mqtt=_RecordingMqtt())
        first_count = len(integration._devices_array)  # noqa: SLF001
        integration.initialize(
            mqtt=_RecordingMqtt(),
            serial_no="SN-TEST",
            firmware_version="V99.0",
            equipment_info="EQ",
        )
        assert len(integration._devices_array) == first_count  # noqa: SLF001

    def test_initialize_without_mqtt_does_not_publish(self) -> None:
        integration, mqtt = _initialise(
            {
                "11000": {
                    Register.NAME: "Power",
                    Register.GROUP: "now-base",
                    Register.MQTT: "power",
                    Register.DEVICE_CLASS: "power",
                },
            },
            mqtt=None,
        )
        assert mqtt is None
        # Devices array is built even without a client, so a later
        # send_discovery_info() call has something to push.
        assert len(integration._devices_array) == 1  # noqa: SLF001
        assert integration.is_initialized is True

    def test_is_initialized_after_initialize(self) -> None:
        integration, _ = _initialise({}, mqtt=_RecordingMqtt())
        assert integration.is_initialized is True

    def test_is_initialized_starts_false(self) -> None:
        integration = _make_integration({})
        assert integration.is_initialized is False

    def test_send_discovery_info_warns_without_mqtt(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        integration = _make_integration(
            {
                "11000": {
                    Register.NAME: "Power",
                    Register.GROUP: "now-base",
                    Register.MQTT: "power",
                    Register.DEVICE_CLASS: "power",
                },
            }
        )
        # Initialize without an mqtt client, then ask to publish.
        integration.initialize(
            mqtt=None, serial_no="SN", firmware_version="V1", equipment_info="EQ"
        )
        with caplog.at_level("WARNING"):
            integration.send_discovery_info()
        assert any("MQTT client is None" in r.message for r in caplog.records)

    def test_send_unregister_info_warns_without_mqtt(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        integration = _make_integration({})
        integration.initialize(
            mqtt=None, serial_no="SN", firmware_version="V1", equipment_info="EQ"
        )
        with caplog.at_level("WARNING"):
            integration.send_unregister_info()
        assert any("MQTT client is None" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Sensor branch
# ---------------------------------------------------------------------------


class TestSensorBranch:
    """Coverage for the plain sensor entity."""

    def test_sensor_minimal_payload_fields(self) -> None:
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Power",
                    Register.GROUP: "now-base",
                    Register.MQTT: "power",
                    Register.DEVICE_CLASS: "power",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        payload = _config_payload(integration, "/sensor/MTEC_power/config")
        assert payload[HA.NAME] == "Power"
        assert payload[HA.UNIQUE_ID] == f"{MTEC_PREFIX}power"
        assert payload[HA.STATE_TOPIC] == "MTEC/SN-TEST/now-base/power/state"
        assert payload[HA.ENABLED_BY_DEFAULT] is True

    def test_sensor_no_unit_defaults_to_empty_string(self) -> None:
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Counter",
                    Register.GROUP: "total",
                    Register.MQTT: "counter",
                    Register.STATE_CLASS: "measurement",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        payload = _config_payload(integration, "/sensor/MTEC_counter/config")
        assert payload[HA.UNIT_OF_MEASUREMENT] == ""

    def test_sensor_state_class_propagates(self) -> None:
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Energy",
                    Register.GROUP: "day",
                    Register.MQTT: "energy",
                    Register.DEVICE_CLASS: "energy",
                    Register.STATE_CLASS: "total_increasing",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        payload = _config_payload(integration, "/sensor/MTEC_energy/config")
        assert payload[HA.STATE_CLASS] == "total_increasing"

    def test_sensor_value_template_propagates(self) -> None:
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Mode",
                    Register.GROUP: "now-base",
                    Register.MQTT: "mode",
                    Register.STATE_CLASS: "measurement",
                    Register.VALUE_TEMPLATE: "{{ value_json.x }}",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        payload = _config_payload(integration, "/sensor/MTEC_mode/config")
        assert payload[HA.VALUE_TEMPLATE] == "{{ value_json.x }}"

    def test_sensor_without_device_class_omits_field(self) -> None:
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Generic",
                    Register.GROUP: "now-base",
                    Register.MQTT: "generic",
                    Register.STATE_CLASS: "measurement",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        payload = _config_payload(integration, "/sensor/MTEC_generic/config")
        assert HA.DEVICE_CLASS not in payload


# ---------------------------------------------------------------------------
# Binary sensor branch
# ---------------------------------------------------------------------------


class TestBinarySensorBranch:
    """Coverage for the binary sensor entity (also produced by switch)."""

    def test_binary_sensor_no_command_topic_recorded(self) -> None:
        integration, mqtt = _initialise(
            {
                "11000": {
                    Register.NAME: "Door",
                    Register.GROUP: "now-base",
                    Register.MQTT: "door",
                    Register.COMPONENT_TYPE: HAPlatform.BINARY_SENSOR,
                    Register.PAYLOAD_ON: "1",
                    Register.PAYLOAD_OFF: "0",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        assert mqtt is not None
        assert mqtt.subscribed == []
        # Device array entry has no command topic.
        for topic, _, command in integration._devices_array:  # noqa: SLF001
            if "binary_sensor" in topic:
                assert command is None

    def test_binary_sensor_payload_on_off(self) -> None:
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Door",
                    Register.GROUP: "now-base",
                    Register.MQTT: "door",
                    Register.COMPONENT_TYPE: HAPlatform.BINARY_SENSOR,
                    Register.PAYLOAD_ON: "1",
                    Register.PAYLOAD_OFF: "0",
                    Register.DEVICE_CLASS: "door",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        payload = _config_payload(integration, "/binary_sensor/MTEC_door/config")
        assert payload[HA.PAYLOAD_ON] == "1"
        assert payload[HA.PAYLOAD_OFF] == "0"
        assert payload[HA.DEVICE_CLASS] == "door"


# ---------------------------------------------------------------------------
# Number branch
# ---------------------------------------------------------------------------


class TestNumberBranch:
    """Number entities create both a number and a sensor entry."""

    @pytest.fixture
    def integration(self) -> tuple[HassIntegration, _RecordingMqtt]:
        mqtt = _RecordingMqtt()
        integration, _ = _initialise(
            {
                "21000": {
                    Register.NAME: "Setpoint",
                    Register.GROUP: "config",
                    Register.MQTT: "setpoint",
                    Register.UNIT: "W",
                    Register.COMPONENT_TYPE: HAPlatform.NUMBER,
                    Register.DEVICE_CLASS: "power",
                },
            },
            mqtt=mqtt,
        )
        return integration, mqtt

    def test_number_creates_number_and_sensor_entries(
        self, integration: tuple[HassIntegration, _RecordingMqtt]
    ) -> None:
        hass, _ = integration
        topics = [t for t, _, _ in hass._devices_array]  # noqa: SLF001
        assert any("/number/MTEC_setpoint/config" in t for t in topics)
        assert any("/sensor/MTEC_setpoint/config" in t for t in topics)

    def test_number_has_command_topic(
        self, integration: tuple[HassIntegration, _RecordingMqtt]
    ) -> None:
        hass, _ = integration
        payload = _config_payload(hass, "/number/MTEC_setpoint/config")
        assert payload[HA.COMMAND_TOPIC] == "MTEC/SN-TEST/config/setpoint/set"
        assert payload[HA.UNIT_OF_MEASUREMENT] == "W"
        assert payload[HA.MODE] == "box"
        assert payload[HA.ENABLED_BY_DEFAULT] is False

    def test_number_subscribes_to_command_topic(
        self, integration: tuple[HassIntegration, _RecordingMqtt]
    ) -> None:
        _, mqtt = integration
        assert "MTEC/SN-TEST/config/setpoint/set" in mqtt.subscribed


# ---------------------------------------------------------------------------
# Select branch
# ---------------------------------------------------------------------------


class TestSelectBranch:
    """Select entities create both a select and a sensor entry."""

    def test_select_creates_companion_sensor(self) -> None:
        integration, _ = _initialise(
            {
                "22000": {
                    Register.NAME: "Mode",
                    Register.GROUP: "config",
                    Register.MQTT: "mode",
                    Register.COMPONENT_TYPE: HAPlatform.SELECT,
                    Register.VALUE_ITEMS: {0: "A"},
                },
            },
            mqtt=_RecordingMqtt(),
        )
        topics = [t for t, _, _ in integration._devices_array]  # noqa: SLF001
        assert any("/select/MTEC_mode/config" in t for t in topics)
        assert any("/sensor/MTEC_mode/config" in t for t in topics)

    def test_select_options_propagated(self) -> None:
        integration, _ = _initialise(
            {
                "22000": {
                    Register.NAME: "Mode",
                    Register.GROUP: "config",
                    Register.MQTT: "mode",
                    Register.COMPONENT_TYPE: HAPlatform.SELECT,
                    Register.VALUE_ITEMS: {0: "Auto", 1: "Manual", 2: "Off"},
                },
            },
            mqtt=_RecordingMqtt(),
        )
        payload = _config_payload(integration, "/select/MTEC_mode/config")
        assert payload[HA.OPTIONS] == ["Auto", "Manual", "Off"]
        assert payload[HA.COMMAND_TOPIC] == "MTEC/SN-TEST/config/mode/set"


# ---------------------------------------------------------------------------
# Switch branch
# ---------------------------------------------------------------------------


class TestSwitchBranch:
    """Switch entities create both a switch and a binary_sensor entry."""

    def test_switch_creates_companion_binary_sensor(self) -> None:
        integration, _ = _initialise(
            {
                "23000": {
                    Register.NAME: "Output",
                    Register.GROUP: "config",
                    Register.MQTT: "out",
                    Register.COMPONENT_TYPE: HAPlatform.SWITCH,
                    Register.PAYLOAD_ON: "ON",
                    Register.PAYLOAD_OFF: "OFF",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        topics = [t for t, _, _ in integration._devices_array]  # noqa: SLF001
        assert any("/switch/MTEC_out/config" in t for t in topics)
        assert any("/binary_sensor/MTEC_out/config" in t for t in topics)

    def test_switch_with_payload_on_off(self) -> None:
        integration, _ = _initialise(
            {
                "23000": {
                    Register.NAME: "Output",
                    Register.GROUP: "config",
                    Register.MQTT: "out",
                    Register.COMPONENT_TYPE: HAPlatform.SWITCH,
                    Register.PAYLOAD_ON: "ON",
                    Register.PAYLOAD_OFF: "OFF",
                    Register.DEVICE_CLASS: "switch",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        switch = _config_payload(integration, "/switch/MTEC_out/config")
        assert switch[HA.PAYLOAD_ON] == "ON"
        assert switch[HA.PAYLOAD_OFF] == "OFF"
        assert switch[HA.COMMAND_TOPIC] == "MTEC/SN-TEST/config/out/set"
        assert switch[HA.DEVICE_CLASS] == "switch"


# ---------------------------------------------------------------------------
# Filtering — registers that should NOT produce a discovery entry
# ---------------------------------------------------------------------------


class TestRegisterFiltering:
    """Registers without group or HA-marker keys must be skipped."""

    def test_payload_on_alone_is_enough_to_register(self) -> None:
        # Even without DEVICE_CLASS, the presence of any HA-marker key
        # triggers registration as a sensor (default component_type).
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Flag",
                    Register.GROUP: "now-base",
                    Register.MQTT: "flag",
                    Register.PAYLOAD_ON: "1",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        assert len(integration._devices_array) == 1  # noqa: SLF001

    def test_register_without_group_skipped(self) -> None:
        integration, _ = _initialise(
            {
                "99999": {
                    Register.NAME: "Internal",
                    Register.GROUP: "",
                    Register.MQTT: "internal",
                    Register.DEVICE_CLASS: "power",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        assert integration._devices_array == []  # noqa: SLF001

    def test_register_without_ha_keys_skipped(self) -> None:
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Plain",
                    Register.GROUP: "now-base",
                    Register.MQTT: "plain",
                    # No DEVICE_CLASS, STATE_CLASS, COMPONENT_TYPE, etc.
                },
            },
            mqtt=_RecordingMqtt(),
        )
        assert integration._devices_array == []  # noqa: SLF001


# ---------------------------------------------------------------------------
# Device info structure
# ---------------------------------------------------------------------------


class TestDeviceInfo:
    """Every payload must contain the canonical ``device`` block."""

    def test_device_info_complete(self) -> None:
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Power",
                    Register.GROUP: "now-base",
                    Register.MQTT: "power",
                    Register.DEVICE_CLASS: "power",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        payload = _config_payload(integration, "/sensor/MTEC_power/config")
        device = payload[HA.DEVICE]
        assert device[HA.IDENTIFIERS] == ["SN-TEST"]
        assert device[HA.MANUFACTURER] == "M-TEC"
        assert device[HA.MODEL] == "Energy-Butler"
        assert device[HA.MODEL_ID] == "MyEquipment"
        assert device[HA.NAME] == "MTEC EnergyButler"
        assert device[HA.SERIAL_NUMBER] == "SN-TEST"
        assert device[HA.SW_VERSION] == "V99.0"

    def test_device_info_shared_across_entities(self) -> None:
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Power",
                    Register.GROUP: "now-base",
                    Register.MQTT: "power",
                    Register.DEVICE_CLASS: "power",
                },
                "12000": {
                    Register.NAME: "Voltage",
                    Register.GROUP: "now-grid",
                    Register.MQTT: "voltage",
                    Register.DEVICE_CLASS: "voltage",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        first = _config_payload(integration, "/sensor/MTEC_power/config")
        second = _config_payload(integration, "/sensor/MTEC_voltage/config")
        assert first[HA.DEVICE] == second[HA.DEVICE]


# ---------------------------------------------------------------------------
# Publish + unregister behaviour
# ---------------------------------------------------------------------------


class TestPublishBehaviour:
    """``send_discovery_info`` and ``send_unregister_info`` semantics."""

    def test_discovery_publishes_with_retain_true(self) -> None:
        mqtt = _RecordingMqtt()
        _initialise(
            {
                "11000": {
                    Register.NAME: "Power",
                    Register.GROUP: "now-base",
                    Register.MQTT: "power",
                    Register.DEVICE_CLASS: "power",
                },
            },
            mqtt=mqtt,
        )
        assert all(retain for _, _, retain in mqtt.published)

    def test_send_discovery_idempotent(self) -> None:
        mqtt = _RecordingMqtt()
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Power",
                    Register.GROUP: "now-base",
                    Register.MQTT: "power",
                    Register.DEVICE_CLASS: "power",
                },
            },
            mqtt=mqtt,
        )
        first_count = len(mqtt.published)
        integration.send_discovery_info()
        # A second send_discovery_info() should publish exactly the same
        # number of messages again — no growth, no de-duplication.
        assert len(mqtt.published) == first_count * 2

    def test_send_unregister_emits_empty_payload_per_topic(self) -> None:
        mqtt = _RecordingMqtt()
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Power",
                    Register.GROUP: "now-base",
                    Register.MQTT: "power",
                    Register.DEVICE_CLASS: "power",
                },
                "12000": {
                    Register.NAME: "Voltage",
                    Register.GROUP: "now-grid",
                    Register.MQTT: "voltage",
                    Register.DEVICE_CLASS: "voltage",
                },
            },
            mqtt=mqtt,
        )
        before = len(mqtt.published)
        integration.send_unregister_info()
        new_publishes = mqtt.published[before:]
        assert len(new_publishes) == 2
        assert all(payload == "" for _, payload, _ in new_publishes)


# ---------------------------------------------------------------------------
# Topic naming
# ---------------------------------------------------------------------------


class TestTopicNaming:
    """Topic-naming invariants relied upon by HA and external dashboards."""

    def test_config_topic_is_under_hass_base_topic(self) -> None:
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Power",
                    Register.GROUP: "now-base",
                    Register.MQTT: "power",
                    Register.DEVICE_CLASS: "power",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        for topic, _, _ in integration._devices_array:  # noqa: SLF001
            assert topic.startswith("homeassistant/")

    def test_state_topic_uses_serial_and_group(self) -> None:
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Power",
                    Register.GROUP: "now-base",
                    Register.MQTT: "power",
                    Register.DEVICE_CLASS: "power",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        payload = _config_payload(integration, "/sensor/MTEC_power/config")
        assert payload[HA.STATE_TOPIC].startswith("MTEC/SN-TEST/now-base/")

    def test_unique_id_uses_mtec_prefix(self) -> None:
        integration, _ = _initialise(
            {
                "11000": {
                    Register.NAME: "Power",
                    Register.GROUP: "now-base",
                    Register.MQTT: "power",
                    Register.DEVICE_CLASS: "power",
                },
            },
            mqtt=_RecordingMqtt(),
        )
        payload = _config_payload(integration, "/sensor/MTEC_power/config")
        assert payload[HA.UNIQUE_ID].startswith(MTEC_PREFIX)
