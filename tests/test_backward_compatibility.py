"""
Tests to ensure MQTT backward compatibility during async migration.

These tests verify that MQTT message formats, topics, and Home Assistant
discovery messages remain identical after async refactoring. This is CRITICAL
to prevent breaking existing Home Assistant integrations.

Reference: BASELINE.md Section 3 - MQTT Message Format Contract
"""

from __future__ import annotations

import json

from aiomtec2mqtt import hass_int
from aiomtec2mqtt.const import Register


class TestMqttTopicFormat:
    """Verify MQTT topic naming remains unchanged."""

    def test_topic_prefix_is_mtec(self) -> None:
        """Topic prefix MUST be 'MTEC' not 'AIOMTEC'."""
        from aiomtec2mqtt.const import MTEC_TOPIC_ROOT

        assert MTEC_TOPIC_ROOT == "MTEC", (
            "Topic root MUST be 'MTEC' for backward compatibility. "
            "Home Assistant integrations depend on this exact prefix."
        )


class TestHomeAssistantDiscovery:
    """Verify Home Assistant discovery messages remain unchanged."""

    def test_device_info_structure(self) -> None:
        """Device info in discovery must include required fields."""
        reg_map = {
            "11000": {
                Register.NAME: "Grid Export",
                Register.GROUP: "now-base",
                Register.MQTT: "grid_export",
            }
        }

        hass = hass_int.HassIntegration(
            hass_base_topic="homeassistant",
            mqtt_topic="MTEC",
            register_map=reg_map,  # type: ignore[arg-type]
        )

        class DummyMqtt:
            def __init__(self) -> None:
                self.published: list[tuple[str, str, bool]] = []

            def publish(self, *, topic: str, payload: str, retain: bool = False) -> None:
                self.published.append((topic, payload, retain))

            def subscribe_to_topic(self, *, topic: str) -> None:
                pass

        mqtt = DummyMqtt()
        hass.initialize(
            mqtt=mqtt,  # type: ignore[arg-type]
            serial_no="A1B2C3D4",
            firmware_version="V27.52.4.0",
            equipment_info="GEN3 8kW",
        )

        # Parse discovery payloads
        for topic, payload, _ in mqtt.published:
            if "/config" in topic and payload:
                parsed = json.loads(payload)

                if "device" in parsed:
                    device = parsed["device"]

                    # Required device fields
                    assert "identifiers" in device, "Device must include 'identifiers'"
                    assert "name" in device, "Device must include 'name'"
                    assert "manufacturer" in device, "Device must include 'manufacturer'"

                    # Verify identifiers format
                    assert isinstance(device["identifiers"], list)
                    assert len(device["identifiers"]) > 0

    def test_discovery_payload_structure(self) -> None:
        """Discovery payloads must contain required HA fields."""
        reg_map = {
            "11000": {
                Register.NAME: "Grid Export",
                Register.GROUP: "now-base",
                Register.MQTT: "grid_export",
                Register.DEVICE_CLASS: "power",
                Register.STATE_CLASS: "measurement",
                Register.UNIT: "W",
            }
        }

        hass = hass_int.HassIntegration(
            hass_base_topic="homeassistant",
            mqtt_topic="MTEC",
            register_map=reg_map,  # type: ignore[arg-type]
        )

        class DummyMqtt:
            def __init__(self) -> None:
                self.published: list[tuple[str, str, bool]] = []

            def publish(self, *, topic: str, payload: str, retain: bool = False) -> None:
                self.published.append((topic, payload, retain))

            def subscribe_to_topic(self, *, topic: str) -> None:
                pass

        mqtt = DummyMqtt()
        hass.initialize(
            mqtt=mqtt, serial_no="A1B2C3D4", firmware_version="V27", equipment_info="EQ"
        )  # type: ignore[arg-type]

        # Parse discovery payloads
        for topic, payload, _ in mqtt.published:
            if "/config" in topic and payload:  # Skip unregister (empty payload)
                parsed = json.loads(payload)

                # Required fields
                assert "name" in parsed, "Discovery must include 'name'"
                assert "unique_id" in parsed, "Discovery must include 'unique_id'"
                assert "state_topic" in parsed, "Discovery must include 'state_topic'"

                # Verify state_topic uses MTEC prefix
                if "state_topic" in parsed:
                    assert parsed["state_topic"].startswith("MTEC/"), (
                        f"state_topic must start with 'MTEC/', got: {parsed['state_topic']}"
                    )

                # Verify unique_id uses MTEC prefix
                if "unique_id" in parsed:
                    assert parsed["unique_id"].startswith("MTEC_"), (
                        f"unique_id must start with 'MTEC_', got: {parsed['unique_id']}"
                    )

    def test_discovery_retain_flag(self) -> None:
        """Discovery messages must be published with retain=True."""
        reg_map = {
            "11000": {
                Register.NAME: "Grid Export",
                Register.GROUP: "now-base",
                Register.MQTT: "grid_export",
            }
        }

        hass = hass_int.HassIntegration(
            hass_base_topic="homeassistant",
            mqtt_topic="MTEC",
            register_map=reg_map,  # type: ignore[arg-type]
        )

        class DummyMqtt:
            def __init__(self) -> None:
                self.published: list[tuple[str, str, bool]] = []

            def publish(self, *, topic: str, payload: str, retain: bool = False) -> None:
                self.published.append((topic, payload, retain))

            def subscribe_to_topic(self, *, topic: str) -> None:
                pass

        mqtt = DummyMqtt()
        hass.initialize(
            mqtt=mqtt, serial_no="A1B2C3D4", firmware_version="V27", equipment_info="EQ"
        )  # type: ignore[arg-type]

        # Verify all discovery messages have retain=True
        for topic, _payload, retain in mqtt.published:
            if "/config" in topic:
                assert retain is True, f"Discovery message to {topic} must have retain=True"

    def test_discovery_topic_pattern(self) -> None:
        """Discovery topics must follow 'homeassistant/<platform>/<device_id>_<entity_id>/config'."""
        reg_map = {
            "11000": {
                Register.NAME: "Grid Export",
                Register.GROUP: "now-base",
                Register.MQTT: "grid_export",
                Register.DEVICE_CLASS: "power",
                Register.STATE_CLASS: "measurement",
                Register.UNIT: "W",
            }
        }

        hass = hass_int.HassIntegration(
            hass_base_topic="homeassistant",
            mqtt_topic="MTEC",
            register_map=reg_map,  # type: ignore[arg-type]
        )

        class DummyMqtt:
            def __init__(self) -> None:
                self.published: list[tuple[str, str, bool]] = []

            def publish(self, *, topic: str, payload: str, retain: bool = False) -> None:
                self.published.append((topic, payload, retain))

            def subscribe_to_topic(self, *, topic: str) -> None:
                pass

        mqtt = DummyMqtt()
        hass.initialize(
            mqtt=mqtt, serial_no="A1B2C3D4", firmware_version="V27", equipment_info="EQ"
        )  # type: ignore[arg-type]

        # Verify discovery topic format
        discovery_topics = [topic for topic, _, _ in mqtt.published]

        # Must match pattern: homeassistant/sensor/MTEC_grid_export/config
        assert any(
            topic.startswith("homeassistant/sensor/MTEC_") and topic.endswith("/config")
            for topic in discovery_topics
        ), (
            f"Discovery topics must match 'homeassistant/<platform>/MTEC_<entity>/config'. Got: {discovery_topics}"
        )

        # Must use MTEC prefix, not AIOMTEC
        for topic in discovery_topics:
            assert "AIOMTEC" not in topic, f"Discovery topic {topic} must not use AIOMTEC prefix"


class TestBackwardCompatibilitySummary:
    """Summary test to verify all critical invariants."""

    def test_critical_invariants_checklist(self) -> None:
        """Verify all critical backward compatibility requirements."""
        from aiomtec2mqtt.const import MTEC_PREFIX, MTEC_TOPIC_ROOT

        # ✅ Topic prefix is MTEC
        assert MTEC_TOPIC_ROOT == "MTEC", "Topic root must be MTEC"

        # ✅ Entity prefix is MTEC_
        assert MTEC_PREFIX == "MTEC_", "Entity prefix must be MTEC_"

        # Note: Additional runtime checks are performed in other test methods
        # This test serves as a quick smoke test for the most critical constants
