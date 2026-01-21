"""Pytest fixtures and fakes for aiomtec2mqtt tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakePahoMessage:
    """A minimal fake of paho.mqtt.client.MQTTMessage for callbacks."""

    topic: str
    payload: bytes


# Note: fake_paho fixture removed - sync mqtt_client.py no longer exists
# The async MQTT client uses aiomqtt which is tested in test_async_mqtt_client.py
