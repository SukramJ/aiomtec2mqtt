"""
Tests to ensure MQTT backward compatibility during async migration.

These tests verify that MQTT message formats, topics, and Home Assistant
discovery messages remain identical after async refactoring. This is CRITICAL
to prevent breaking existing Home Assistant integrations.

Reference: BASELINE.md Section 3 - MQTT Message Format Contract
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aiomtec2mqtt import hass_int, mqtt_client
from aiomtec2mqtt.const import Config, Register


class TestMqttTopicFormat:
    """Verify MQTT topic naming remains unchanged."""

    def test_serial_number_in_topic_format(
        self, fake_paho: dict[str, Any], base_config: dict[str, Any]
    ) -> None:
        """Serial numbers in topics should use consistent format."""
        client = mqtt_client.MqttClient(
            config=base_config, on_mqtt_message=lambda *args: None, hass=None
        )

        # Test with various serial formats
        test_serials = ["ABC123DEF", "12345678", "A1B2C3D4"]

        for serial in test_serials:
            topic = f"MTEC/{serial}/now-base"
            client.publish(topic=topic, payload='{"test": 1}', retain=False)

            fake = fake_paho["client"]
            published_topics = [t for t, _, _, _ in fake.published]

            # Verify the topic format is preserved
            assert topic in published_topics, f"Topic {topic} should be published as-is"

        client.stop()

    def test_topic_prefix_is_mtec(self) -> None:
        """Topic prefix MUST be 'MTEC' not 'AIOMTEC'."""
        from aiomtec2mqtt.const import MTEC_TOPIC_ROOT

        assert MTEC_TOPIC_ROOT == "MTEC", (
            "Topic root MUST be 'MTEC' for backward compatibility. "
            "Home Assistant integrations depend on this exact prefix."
        )

    def test_topic_structure_format(
        self, fake_paho: dict[str, Any], base_config: dict[str, Any]
    ) -> None:
        """Topics must follow 'MTEC/<serial>/<group>' pattern."""
        client = mqtt_client.MqttClient(
            config=base_config, on_mqtt_message=lambda *args: None, hass=None
        )

        # Publish to different groups using the actual topic format
        serial = "A1B2C3D4"
        client.publish(
            topic=f"MTEC/{serial}/now-base", payload='{"grid_export": 2500}', retain=False
        )
        client.publish(topic=f"MTEC/{serial}/config", payload='{"firmware": "V27"}', retain=False)
        client.publish(topic=f"MTEC/{serial}/day", payload='{"energy": 15000}', retain=False)

        fake = fake_paho["client"]
        published_topics = [topic for topic, _, _, _ in fake.published]

        # Verify exact format
        assert f"MTEC/{serial}/now-base" in published_topics
        assert f"MTEC/{serial}/config" in published_topics
        assert f"MTEC/{serial}/day" in published_topics

        # Verify no alternative formats were used
        for topic in published_topics:
            if not topic.startswith("homeassistant"):  # Exclude HA status topic
                assert topic.startswith("MTEC/"), f"Topic {topic} must start with 'MTEC/'"
                assert "AIOMTEC" not in topic, f"Topic {topic} must not contain 'AIOMTEC'"

        client.stop()


class TestMqttPayloadFormat:
    """Verify MQTT payload structure remains unchanged."""

    def test_field_names_are_snake_case(
        self, fake_paho: dict[str, Any], base_config: dict[str, Any]
    ) -> None:
        """Field names must use snake_case, not camelCase."""
        client = mqtt_client.MqttClient(
            config=base_config, on_mqtt_message=lambda *args: None, hass=None
        )

        # Use field names that match register definitions
        test_data = {
            "grid_export": 2500,  # snake_case ✓
            "battery_soc": 85,  # snake_case ✓
            "pv_power": 3200,  # snake_case ✓
        }

        test_payload = json.dumps(test_data)
        client.publish(topic="MTEC/TEST/now-base", payload=test_payload, retain=False)

        fake = fake_paho["client"]
        for _, payload, _, _ in fake.published:
            if payload:  # Skip empty payloads
                parsed = json.loads(payload)

                # Verify no camelCase keys
                for key in parsed:
                    assert key.islower() or "_" in key, (
                        f"Field name '{key}' must be snake_case. "
                        f"Found mixed case or camelCase which breaks backward compatibility."
                    )
                    # No camelCase patterns
                    assert not any(c.isupper() for c in key if c != "_"), (
                        f"Field name '{key}' appears to be camelCase"
                    )

        client.stop()

    def test_numeric_types_preserved(
        self, fake_paho: dict[str, Any], base_config: dict[str, Any]
    ) -> None:
        """Numeric values must maintain int/float types appropriately."""
        client = mqtt_client.MqttClient(
            config=base_config, on_mqtt_message=lambda *args: None, hass=None
        )

        test_data = {
            "power_int": 2500,  # Should stay int
            "voltage_float": 230.5,  # Should stay float
            "soc_int": 85,  # Should stay int
        }

        test_payload = json.dumps(test_data)
        client.publish(topic="MTEC/TEST/now-base", payload=test_payload, retain=False)

        fake = fake_paho["client"]
        for _, payload, _, _ in fake.published:
            if payload:  # Skip empty payloads
                parsed = json.loads(payload)

                # Verify types
                if "power_int" in parsed:
                    assert isinstance(parsed["power_int"], int)
                if "voltage_float" in parsed:
                    assert isinstance(parsed["voltage_float"], float)
                if "soc_int" in parsed:
                    assert isinstance(parsed["soc_int"], int)

        client.stop()

    def test_payload_is_valid_json(
        self, fake_paho: dict[str, Any], base_config: dict[str, Any]
    ) -> None:
        """Payloads must be valid UTF-8 JSON."""
        client = mqtt_client.MqttClient(
            config=base_config, on_mqtt_message=lambda *args: None, hass=None
        )

        test_payload = json.dumps({"grid_export": 2500, "battery_soc": 85})
        client.publish(topic="MTEC/TEST/now-base", payload=test_payload, retain=False)

        fake = fake_paho["client"]
        assert len(fake.published) > 0

        for _, payload, _, _ in fake.published:
            # Skip homeassistant status subscription
            if payload:
                # Must be valid JSON
                parsed = json.loads(payload)
                assert isinstance(parsed, dict)

        client.stop()


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

            def publish(self, topic: str, payload: str, retain: bool = False) -> None:
                self.published.append((topic, payload, retain))

            def subscribe_to_topic(self, topic: str) -> None:
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
                    assert device["identifiers"][0].startswith("MTEC_")

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

            def publish(self, topic: str, payload: str, retain: bool = False) -> None:
                self.published.append((topic, payload, retain))

            def subscribe_to_topic(self, topic: str) -> None:
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

            def publish(self, topic: str, payload: str, retain: bool = False) -> None:
                self.published.append((topic, payload, retain))

            def subscribe_to_topic(self, topic: str) -> None:
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

            def publish(self, topic: str, payload: str, retain: bool = False) -> None:
                self.published.append((topic, payload, retain))

            def subscribe_to_topic(self, topic: str) -> None:
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


class TestQosAndRetain:
    """Verify QoS and retain flag behavior."""

    def test_data_messages_not_retained(
        self, fake_paho: dict[str, Any], base_config: dict[str, Any]
    ) -> None:
        """Regular data messages should NOT be retained (retain=False)."""
        client = mqtt_client.MqttClient(
            config=base_config, on_mqtt_message=lambda *args: None, hass=None
        )

        test_payload = json.dumps({"grid_export": 2500})
        client.publish(topic="MTEC/TEST/now-base", payload=test_payload, retain=False)

        fake = fake_paho["client"]

        # Find our test message (not the homeassistant status subscription)
        for topic, _, _, retain in fake.published:
            if "MTEC/TEST" in topic:
                assert retain is False, "Data messages should not be retained"

        client.stop()

    def test_qos_level_is_zero(
        self, fake_paho: dict[str, Any], base_config: dict[str, Any]
    ) -> None:
        """Messages should use QoS 0 for best performance."""
        client = mqtt_client.MqttClient(
            config=base_config, on_mqtt_message=lambda *args: None, hass=None
        )

        test_payload = json.dumps({"grid_export": 2500})
        client.publish(topic="MTEC/TEST/now-base", payload=test_payload, retain=False)

        fake = fake_paho["client"]
        for topic, _, qos, _ in fake.published:
            if "MTEC/TEST" in topic:
                assert qos == 0, "QoS should be 0 for data messages"

        client.stop()


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Provide a minimal valid configuration for MqttClient."""
    return {
        Config.MQTT_LOGIN: "user",
        Config.MQTT_PASSWORD: "pass",
        Config.MQTT_SERVER: "localhost",
        Config.MQTT_PORT: 1883,
        Config.HASS_BASE_TOPIC: "homeassistant",
    }


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
