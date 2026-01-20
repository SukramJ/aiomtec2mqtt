"""Integration tests for async components."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiomtec2mqtt.async_coordinator import AsyncMtecCoordinator
from aiomtec2mqtt.async_modbus_client import AsyncModbusClient
from aiomtec2mqtt.async_mqtt_client import AsyncMqttClient
from aiomtec2mqtt.const import Config, Register, RegisterGroup


@pytest.fixture
def mock_config():
    """Provide mock configuration."""
    return {
        Config.MODBUS_IP: "192.168.1.100",
        Config.MODBUS_PORT: 502,
        Config.MODBUS_SLAVE: 1,
        Config.MODBUS_TIMEOUT: 5,
        Config.MQTT_SERVER: "mqtt.example.com",
        Config.MQTT_PORT: 1883,
        Config.MQTT_LOGIN: "testuser",
        Config.MQTT_PASSWORD: "testpass",
        Config.MQTT_TOPIC: "MTEC/12345",
        Config.MQTT_FLOAT_FORMAT: ".2f",
        Config.HASS_BASE_TOPIC: "homeassistant",
        Config.HASS_ENABLE: False,
        Config.HASS_BIRTH_GRACETIME: 0,
        Config.DEBUG: False,
        Config.REFRESH_NOW: 1,
        Config.REFRESH_DAY: 10,
        Config.REFRESH_STATIC: 3600,
        Config.REFRESH_CONFIG: 3600,
        Config.REFRESH_TOTAL: 3600,
    }


@pytest.fixture
def mock_register_map():
    """Provide mock register map."""
    return {
        "10100": {
            Register.NAME: "battery_soc",
            Register.UNIT: "%",
            Register.GROUP: RegisterGroup.BASE,
            Register.SCALE: 1,
        },
        "10101": {
            Register.NAME: "battery_voltage",
            Register.UNIT: "V",
            Register.GROUP: RegisterGroup.BASE,
            Register.SCALE: 10,
        },
        "31000": {
            Register.NAME: "daily_energy",
            Register.UNIT: "kWh",
            Register.GROUP: RegisterGroup.DAY,
            Register.SCALE: 10,
        },
    }


class TestAsyncIntegration:
    """Integration tests for async components."""

    @pytest.mark.asyncio
    async def test_concurrent_register_reads(self, mock_config, mock_register_map):
        """Test concurrent reading of multiple register groups."""
        modbus_client = AsyncModbusClient(
            config=mock_config,
            register_map=mock_register_map,
            register_groups=[RegisterGroup.BASE, RegisterGroup.DAY],
        )

        with patch.object(modbus_client, "_client") as mock_modbus:
            mock_response = MagicMock()
            mock_response.isError.return_value = False
            mock_response.registers = [50]
            mock_modbus.read_holding_registers = AsyncMock(return_value=mock_response)
            modbus_client._state_machine.transition_to(
                new_state=modbus_client._state_machine._state.CONNECTED
            )

            # Read multiple groups concurrently
            results = await asyncio.gather(
                modbus_client.read_register_group(group_name=RegisterGroup.BASE),
                modbus_client.read_register_group(group_name=RegisterGroup.DAY),
                return_exceptions=True,
            )

            # Verify both succeeded
            assert len(results) == 2
            assert isinstance(results[0], dict)
            assert isinstance(results[1], dict)

    @pytest.mark.asyncio
    async def test_coordinator_initialization(self):
        """Test coordinator initialization."""
        with (
            patch("aiomtec2mqtt.async_coordinator.init_config") as mock_init_config,
            patch("aiomtec2mqtt.async_coordinator.init_register_map") as mock_init_register_map,
        ):
            # Mock configuration
            mock_init_config.return_value = {
                Config.MODBUS_IP: "192.168.1.100",
                Config.MODBUS_PORT: 502,
                Config.MODBUS_SLAVE: 1,
                Config.MODBUS_TIMEOUT: 5,
                Config.MQTT_SERVER: "mqtt.example.com",
                Config.MQTT_PORT: 1883,
                Config.MQTT_LOGIN: "testuser",
                Config.MQTT_PASSWORD: "testpass",
                Config.MQTT_TOPIC: "MTEC/12345",
                Config.MQTT_FLOAT_FORMAT: ".2f",
                Config.HASS_BASE_TOPIC: "homeassistant",
                Config.HASS_ENABLE: False,
                Config.HASS_BIRTH_GRACETIME: 0,
                Config.DEBUG: False,
                Config.REFRESH_NOW: 1,
                Config.REFRESH_DAY: 10,
            }

            mock_init_register_map.return_value = ({}, [RegisterGroup.BASE])

            # Create coordinator
            coordinator = AsyncMtecCoordinator()

            # Verify initialization
            assert coordinator._modbus_client is not None
            assert coordinator._mqtt_client is not None
            assert coordinator._health_check is not None

    @pytest.mark.asyncio
    async def test_coordinator_shutdown(self):
        """Test coordinator graceful shutdown."""
        with (
            patch("aiomtec2mqtt.async_coordinator.init_config") as mock_init_config,
            patch("aiomtec2mqtt.async_coordinator.init_register_map") as mock_init_register_map,
        ):
            mock_init_config.return_value = {
                Config.MODBUS_IP: "192.168.1.100",
                Config.MODBUS_PORT: 502,
                Config.MODBUS_SLAVE: 1,
                Config.MODBUS_TIMEOUT: 5,
                Config.MQTT_SERVER: "mqtt.example.com",
                Config.MQTT_PORT: 1883,
                Config.MQTT_LOGIN: "testuser",
                Config.MQTT_PASSWORD: "testpass",
                Config.MQTT_TOPIC: "MTEC/12345",
                Config.MQTT_FLOAT_FORMAT: ".2f",
                Config.HASS_BASE_TOPIC: "homeassistant",
                Config.HASS_ENABLE: False,
                Config.HASS_BIRTH_GRACETIME: 0,
                Config.DEBUG: False,
                Config.REFRESH_NOW: 1,
                Config.REFRESH_DAY: 10,
            }

            mock_init_register_map.return_value = ({}, [RegisterGroup.BASE])

            coordinator = AsyncMtecCoordinator()

            # Request shutdown
            coordinator.shutdown()

            # Verify shutdown event is set
            assert coordinator._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_error_resilience(self, mock_config, mock_register_map):
        """Test error handling and resilience."""
        modbus_client = AsyncModbusClient(
            config=mock_config,
            register_map=mock_register_map,
            register_groups=[RegisterGroup.BASE],
        )

        # Simulate connection failure
        with patch.object(modbus_client, "_client") as mock_modbus:
            mock_modbus.read_holding_registers = AsyncMock(side_effect=TimeoutError())
            modbus_client._state_machine.transition_to(
                new_state=modbus_client._state_machine._state.CONNECTED
            )

            # Attempt read
            from aiomtec2mqtt.exceptions import ModbusTimeoutError

            with pytest.raises(ModbusTimeoutError):
                await modbus_client.read_holding_registers(address=10100, count=1)

            # Verify error was recorded
            assert modbus_client.error_count > 0

    @pytest.mark.asyncio
    async def test_health_check_monitoring(self, mock_config, mock_register_map):
        """Test health check monitoring during operations."""
        from aiomtec2mqtt.health import HealthCheck

        health_check = HealthCheck()

        _modbus_client = AsyncModbusClient(
            config=mock_config,
            register_map=mock_register_map,
            register_groups=[RegisterGroup.BASE],
            health_check=health_check,
        )

        _mqtt_client = AsyncMqttClient(config=mock_config, health_check=health_check)

        # Simulate successful operations
        health_check.record_success(name="async_modbus")
        health_check.record_success(name="async_mqtt")

        # Check system health
        system_health = health_check.check_health()

        assert system_health.is_healthy()
        assert len(system_health.components) == 2

    @pytest.mark.asyncio
    async def test_modbus_mqtt_integration(self, mock_config, mock_register_map):
        """Test integration between Modbus and MQTT clients."""
        # Create clients
        modbus_client = AsyncModbusClient(
            config=mock_config,
            register_map=mock_register_map,
            register_groups=[RegisterGroup.BASE],
        )

        mqtt_client = AsyncMqttClient(config=mock_config)

        # Mock Modbus responses
        with patch.object(modbus_client, "_client") as mock_modbus:
            mock_response = MagicMock()
            mock_response.isError.return_value = False
            mock_response.registers = [50, 245]  # battery_soc=50, battery_voltage=24.5V
            mock_modbus.read_holding_registers = AsyncMock(return_value=mock_response)
            modbus_client._state_machine.transition_to(
                new_state=modbus_client._state_machine._state.CONNECTED
            )

            # Mock MQTT client
            with patch.object(mqtt_client, "_client") as mock_mqtt:
                mock_mqtt.publish = AsyncMock()
                mqtt_client._connected = True

                # Read from Modbus
                data = await modbus_client.read_register_group(group_name=RegisterGroup.BASE)

                # Publish to MQTT
                import json

                payload = json.dumps(data)
                await mqtt_client.publish(topic="test/topic", payload=payload)

                # Verify
                assert "battery_soc" in data
                assert data["battery_soc"] == 50
                assert "battery_voltage" in data
                assert abs(data["battery_voltage"] - 24.5) < 0.01

                mock_mqtt.publish.assert_called_once()


class TestPerformance:
    """Performance tests for async implementation."""

    @pytest.mark.asyncio
    async def test_concurrent_read_performance(self, mock_config, mock_register_map):
        """Test performance of concurrent register reads."""
        import time

        modbus_client = AsyncModbusClient(
            config=mock_config,
            register_map=mock_register_map,
            register_groups=[RegisterGroup.BASE],
        )

        with patch.object(modbus_client, "_client") as mock_modbus:
            mock_response = MagicMock()
            mock_response.isError.return_value = False
            mock_response.registers = [50]

            # Simulate 10ms read time
            async def slow_read(*args, **kwargs):
                await asyncio.sleep(0.01)
                return mock_response

            mock_modbus.read_holding_registers = slow_read
            modbus_client._state_machine.transition_to(
                new_state=modbus_client._state_machine._state.CONNECTED
            )

            # Measure sequential reads
            start_sequential = time.time()
            for _ in range(10):
                await modbus_client.read_holding_registers(address=10100, count=1)
            sequential_time = time.time() - start_sequential

            # Measure concurrent reads
            start_concurrent = time.time()
            await asyncio.gather(
                *[modbus_client.read_holding_registers(address=10100, count=1) for _ in range(10)]
            )
            concurrent_time = time.time() - start_concurrent

            # Concurrent should be significantly faster
            assert concurrent_time < sequential_time * 0.5
