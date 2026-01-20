"""Tests for Home Assistant discovery integration builder."""

from __future__ import annotations

from aiomtec2mqtt.const import HAPlatform, Register
from aiomtec2mqtt.hass_int import HassIntegration


class DummyMqtt:
    """Minimal MQTT adapter that records publications and subscriptions."""

    def __init__(self) -> None:
        """Initialize."""
        self.published: list[tuple[str, str, bool]] = []
        self.subscribed: list[str] = []

    def publish(self, topic: str, payload: str, retain: bool = False) -> None:
        """Record publication."""
        self.published.append((topic, payload, retain))

    def subscribe_to_topic(self, topic: str) -> None:
        """Record subscription to topic."""
        self.subscribed.append(topic)


class TestHassIntegration:
    """Tests for Home Assistant integration and discovery."""

    def test_initialize_and_send_discovery(self) -> None:
        """HassIntegration should build discovery payloads and publish/subscribe appropriately."""
        # Minimal register exposing a sensor and a number/select/switch to exercise the branches
        reg_map = {
            "11000": {
                Register.NAME: "Grid export",
                Register.GROUP: "now-base",
                Register.MQTT: "grid_export",
                Register.DEVICE_CLASS: "power",
                Register.STATE_CLASS: "measurement",
            },
            "21000": {
                Register.NAME: "Setpoint",
                Register.GROUP: "config",
                Register.MQTT: "setpoint",
                Register.COMPONENT_TYPE: HAPlatform.NUMBER,
                Register.UNIT: "W",
                Register.DEVICE_CLASS: "power",
            },
            "22000": {
                Register.NAME: "Mode",
                Register.GROUP: "config",
                Register.MQTT: "mode",
                Register.COMPONENT_TYPE: HAPlatform.SELECT,
                Register.VALUE_ITEMS: {0: "A", 1: "B"},
            },
            "23000": {
                Register.NAME: "Output",
                Register.GROUP: "config",
                Register.MQTT: "out",
                Register.COMPONENT_TYPE: HAPlatform.SWITCH,
                Register.PAYLOAD_ON: "ON",
                Register.PAYLOAD_OFF: "OFF",
            },
        }

        hass = HassIntegration(hass_base_topic="homeassistant", register_map=reg_map)  # type: ignore[arg-type]
        mqtt = DummyMqtt()
        hass.initialize(mqtt=mqtt, serial_no="SN", firmware_version="V1", equipment_info="EQ")

        # There should be one config per entity built
        # sensor + number(sensor+number) + select(sensor+select) + switch(binary+switch) => 6
        topics = [t for t, _, _ in hass._devices_array]  # noqa: SLF001 - internal for test inspection
        assert len(topics) == 7

        # And after initialize, discovery published with retain flag and command topics subscribed
        retained = [r for _, _, r in mqtt.published]
        assert all(retained)
        assert any(t.endswith("/number/MTEC_setpoint/config") for t, _, _ in hass._devices_array)
        assert any(
            t.endswith("/sensor/MTEC_grid_export/config") for t, _, _ in hass._devices_array
        )

        # For number/select/switch there should be command subscriptions
        assert any(s.endswith("/config/setpoint/set") for s in mqtt.subscribed)
        assert any(s.endswith("/config/mode/set") for s in mqtt.subscribed)
        assert any(s.endswith("/config/out/set") for s in mqtt.subscribed)

        # Unregister should clear retained discovery entries
        hass.send_unregister_info()
        assert any(payload == "" for _, payload, _ in mqtt.published)
