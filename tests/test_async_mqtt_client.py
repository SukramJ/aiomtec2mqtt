"""Tests for async MQTT client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiomqtt
import pytest

from aiomtec2mqtt.async_mqtt_client import AsyncMqttClient
from aiomtec2mqtt.const import Config
from aiomtec2mqtt.exceptions import MqttConnectionError, MqttPublishError, MqttSubscribeError
from aiomtec2mqtt.health import HealthCheck
from aiomtec2mqtt.resilience import ConnectionState


@pytest.fixture
def config():
    """Provide test configuration."""
    return {
        Config.MQTT_SERVER: "mqtt.example.com",
        Config.MQTT_PORT: 1883,
        Config.MQTT_LOGIN: "testuser",
        Config.MQTT_PASSWORD: "testpass",
        Config.HASS_BASE_TOPIC: "homeassistant",
    }


@pytest.fixture
def health_check():
    """Provide health check manager."""
    return HealthCheck()


@pytest.fixture
def async_mqtt_client(config, health_check):
    """Provide AsyncMqttClient instance."""
    return AsyncMqttClient(
        config=config,
        health_check=health_check,
    )


class TestAsyncMqttClient:
    """Test AsyncMqttClient class."""

    @pytest.mark.asyncio
    async def test_connect_failure(self, async_mqtt_client):
        """Test connection failure."""
        with patch("aiomtec2mqtt.async_mqtt_client.aiomqtt.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client_class.return_value = mock_client

            with pytest.raises(MqttConnectionError):
                await async_mqtt_client.connect()

            assert async_mqtt_client.state == ConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_connect_success(self, async_mqtt_client):
        """Test successful connection."""
        with patch("aiomtec2mqtt.async_mqtt_client.aiomqtt.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            await async_mqtt_client.connect()

            assert async_mqtt_client.is_connected
            assert async_mqtt_client.state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_context_manager(self, async_mqtt_client):
        """Test async context manager."""
        with patch("aiomqtt.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            async with async_mqtt_client.connection():
                assert async_mqtt_client.is_connected

            mock_client.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, async_mqtt_client):
        """Test disconnection."""
        # Setup connected client
        mock_client = AsyncMock()
        mock_client.__aexit__ = AsyncMock()
        async_mqtt_client._client = mock_client
        async_mqtt_client._connected = True
        async_mqtt_client._state_machine.transition_to(new_state=ConnectionState.CONNECTED)

        await async_mqtt_client.disconnect()

        assert not async_mqtt_client.is_connected
        assert async_mqtt_client.state == ConnectionState.DISCONNECTED
        mock_client.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_integration(self, async_mqtt_client, health_check):
        """Test health check integration."""
        with patch("aiomqtt.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.publish = AsyncMock()
            mock_client_class.return_value = mock_client

            await async_mqtt_client.connect()
            async_mqtt_client._connected = True

            # Successful publish should record success
            await async_mqtt_client.publish(topic="test/topic", payload="payload")

            component_health = health_check.get_component_health(name="async_mqtt")
            assert component_health is not None
            assert component_health.last_success is not None

    @pytest.mark.asyncio
    async def test_initialization(self, async_mqtt_client):
        """Test client initialization."""
        assert async_mqtt_client._hostname == "mqtt.example.com"
        assert async_mqtt_client._port == 1883
        assert async_mqtt_client._username == "testuser"
        assert async_mqtt_client._password == "testpass"
        assert async_mqtt_client.state == ConnectionState.DISCONNECTED
        assert not async_mqtt_client.is_connected

    @pytest.mark.asyncio
    async def test_message_callback(self, config, health_check):
        """Test message listener with callback."""
        messages_received = []

        def on_message(message: aiomqtt.Message):
            messages_received.append(message)

        client = AsyncMqttClient(
            config=config,
            on_message=on_message,
            health_check=health_check,
        )

        with patch("aiomqtt.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()

            # Create async generator for messages
            async def message_generator():
                yield MagicMock(spec=aiomqtt.Message, topic="test/topic", payload=b"test")

            mock_client.messages = message_generator()
            mock_client_class.return_value = mock_client

            await client.connect()

            # Give message listener time to process
            import asyncio

            await asyncio.sleep(0.1)

            await client.disconnect()

    @pytest.mark.asyncio
    async def test_publish_not_connected(self, async_mqtt_client):
        """Test publish when not connected."""
        with pytest.raises(MqttPublishError) as exc_info:
            await async_mqtt_client.publish(topic="test/topic", payload="payload")

        assert "Not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_publish_success(self, async_mqtt_client):
        """Test successful publish."""
        # Setup connected client
        mock_client = AsyncMock()
        mock_client.publish = AsyncMock()
        async_mqtt_client._client = mock_client
        async_mqtt_client._connected = True
        async_mqtt_client._state_machine.transition_to(new_state=ConnectionState.CONNECTED)

        await async_mqtt_client.publish(topic="test/topic", payload="test payload")

        mock_client.publish.assert_called_once()
        call_args = mock_client.publish.call_args
        assert call_args.kwargs["topic"] == "test/topic"
        assert call_args.kwargs["payload"] == "test payload"

    @pytest.mark.asyncio
    async def test_publish_with_retain_and_qos(self, async_mqtt_client):
        """Test publish with retain and QoS."""
        mock_client = AsyncMock()
        mock_client.publish = AsyncMock()
        async_mqtt_client._client = mock_client
        async_mqtt_client._connected = True
        async_mqtt_client._state_machine.transition_to(new_state=ConnectionState.CONNECTED)

        await async_mqtt_client.publish(topic="test/topic", payload="payload", retain=True, qos=1)

        call_args = mock_client.publish.call_args
        assert call_args.kwargs["retain"] is True
        assert call_args.kwargs["qos"] == 1

    @pytest.mark.asyncio
    async def test_state_transitions(self, async_mqtt_client):
        """Test connection state transitions."""
        assert async_mqtt_client.state == ConnectionState.DISCONNECTED

        with patch("aiomqtt.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            # Connect
            await async_mqtt_client.connect()
            assert async_mqtt_client.state == ConnectionState.CONNECTED

            # Disconnect
            await async_mqtt_client.disconnect()
            assert async_mqtt_client.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_subscribe_not_connected(self, async_mqtt_client):
        """Test subscribe when not connected."""
        with pytest.raises(MqttSubscribeError) as exc_info:
            await async_mqtt_client.subscribe(topic="test/topic")

        assert "Not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_subscribe_success(self, async_mqtt_client):
        """Test successful subscription."""
        mock_client = AsyncMock()
        mock_client.subscribe = AsyncMock()
        async_mqtt_client._client = mock_client
        async_mqtt_client._connected = True
        async_mqtt_client._state_machine.transition_to(new_state=ConnectionState.CONNECTED)

        await async_mqtt_client.subscribe(topic="test/topic")

        mock_client.subscribe.assert_called_once_with("test/topic")
        assert "test/topic" in async_mqtt_client.subscribed_topics

    @pytest.mark.asyncio
    async def test_unsubscribe(self, async_mqtt_client):
        """Test unsubscription."""
        mock_client = AsyncMock()
        mock_client.unsubscribe = AsyncMock()
        async_mqtt_client._client = mock_client
        async_mqtt_client._connected = True
        async_mqtt_client._subscribed_topics.add("test/topic")

        await async_mqtt_client.unsubscribe(topic="test/topic")

        mock_client.unsubscribe.assert_called_once_with("test/topic")
        assert "test/topic" not in async_mqtt_client.subscribed_topics
