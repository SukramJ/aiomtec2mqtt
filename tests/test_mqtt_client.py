"""Tests for the MQTT client wrapper."""

from __future__ import annotations

from typing import Any

import pytest

from aiomtec2mqtt import mqtt_client as mqtt_mod
from aiomtec2mqtt.const import Config


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


def dummy_on_message(client: Any, userdata: Any, message: Any) -> None:
    """No-op message handler used by tests."""


class TestMqttClient:
    """Tests for MQTT client initialization, connection, and messaging."""

    def test_hass_subscription_on_connect(
        self, fake_paho: dict[str, Any], base_config: dict[str, Any]
    ) -> None:
        """When hass integration is provided, the client subscribes to its status topic on connect."""

        # Build a tiny hass object with required attribute only for the constructor to consider it truthy
        class DummyHass:
            pass

        hass = DummyHass()
        client = mqtt_mod.MqttClient(
            config=base_config, on_mqtt_message=dummy_on_message, hass=hass
        )  # type: ignore[arg-type]
        fake = fake_paho["client"]
        # After loop_start, on_connect subscribes to hass status topic
        hass_status = f"{base_config[Config.HASS_BASE_TOPIC]}/status"
        assert hass_status in fake.subscribed
        client.stop()

    def test_initialize_and_connects(
        self, fake_paho: dict[str, Any], base_config: dict[str, Any]
    ) -> None:
        """Client should create underlying paho client, set callbacks, and start loop."""
        client = mqtt_mod.MqttClient(
            config=base_config, on_mqtt_message=dummy_on_message, hass=None
        )
        # Our fake connects on loop_start automatically
        fake = fake_paho["client"]
        assert fake._connected is True
        # Verify that a last will was set and reconnect delay configured
        assert fake._will is not None
        assert fake._reconnect_delay == (1, 120)
        client.stop()  # Should cleanly stop without raising

    def test_publish_and_subscriptions(
        self, fake_paho: dict[str, Any], base_config: dict[str, Any]
    ) -> None:
        """Publish should record messages and subscribe/unsubscribe should maintain bookkeeping."""
        client = mqtt_mod.MqttClient(
            config=base_config, on_mqtt_message=dummy_on_message, hass=None
        )

        # subscribe while connected -> paho subscribe is called
        client.subscribe_to_topic(topic="homeassistant/status")
        client.subscribe_to_topic(topic="some/topic")
        client.subscribe_to_topic(topic="some/topic")  # second time is a no-op

        fake = fake_paho["client"]
        assert sorted(fake.subscribed) == ["homeassistant/status", "some/topic"]

        client.publish(topic="a/b", payload="hello", retain=False)
        client.publish(topic="a/b", payload="world", retain=True)
        assert fake.published[-1] == ("a/b", "world", 0, True)

        client.unsubscribe_from_topic(topic="some/topic")
        assert "some/topic" in fake.unsubscribed

        client.stop()
