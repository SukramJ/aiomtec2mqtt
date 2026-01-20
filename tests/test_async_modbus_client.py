"""Tests for async Modbus client."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiomtec2mqtt.async_modbus_client import AsyncModbusClient
from aiomtec2mqtt.const import Config, Register
from aiomtec2mqtt.exceptions import ModbusConnectionError, ModbusReadError, ModbusTimeoutError
from aiomtec2mqtt.health import HealthCheck
from aiomtec2mqtt.resilience import ConnectionState


@pytest.fixture
def config():
    """Provide test configuration."""
    return {
        Config.MODBUS_IP: "192.168.1.100",
        Config.MODBUS_PORT: 502,
        Config.MODBUS_SLAVE: 1,
        Config.MODBUS_TIMEOUT: 5,
    }


@pytest.fixture
def register_map():
    """Provide test register map."""
    return {
        "10100": {
            Register.NAME: "battery_soc",
            Register.UNIT: "%",
            Register.GROUP: "BASE",
            Register.SCALE: 1,
        },
        "10101": {
            Register.NAME: "battery_voltage",
            Register.UNIT: "V",
            Register.GROUP: "BASE",
            Register.SCALE: 10,
        },
        "10102": {
            Register.NAME: "battery_current",
            Register.UNIT: "A",
            Register.GROUP: "BASE",
            Register.SCALE: 10,
        },
        "10120": {
            Register.NAME: "grid_power",
            Register.UNIT: "W",
            Register.GROUP: "BASE",
            Register.SCALE: 1,
        },
        "10121": {
            Register.NAME: "grid_frequency",
            Register.UNIT: "Hz",
            Register.GROUP: "BASE",
            Register.SCALE: 100,
        },
    }


@pytest.fixture
def register_groups():
    """Provide test register groups."""
    return ["BASE", "EXTENDED", "STATS"]


@pytest.fixture
def health_check():
    """Provide health check manager."""
    return HealthCheck()


@pytest.fixture
def async_modbus_client(config, register_map, register_groups, health_check):
    """Provide AsyncModbusClient instance."""
    return AsyncModbusClient(
        config=config,
        register_map=register_map,
        register_groups=register_groups,
        health_check=health_check,
    )


class TestAsyncModbusClient:
    """Test AsyncModbusClient class."""

    @pytest.mark.asyncio
    async def test_concurrent_group_reads(self, async_modbus_client):
        """Test concurrent reading of multiple register groups."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.isError.return_value = False
        mock_response.registers = [50, 245, 100]
        mock_client.read_holding_registers = AsyncMock(return_value=mock_response)

        async_modbus_client._client = mock_client
        async_modbus_client._state_machine.transition_to(new_state=ConnectionState.CONNECTED)

        # Read multiple groups concurrently
        results = await asyncio.gather(
            async_modbus_client.read_register_group(group_name="BASE"),
            async_modbus_client.read_register_group(group_name="BASE"),
            async_modbus_client.read_register_group(group_name="BASE"),
        )

        assert len(results) == 3
        assert all("battery_soc" in result for result in results)

    @pytest.mark.asyncio
    async def test_connect_failure(self, async_modbus_client):
        """Test connection failure."""
        with patch("aiomtec2mqtt.async_modbus_client.AsyncModbusTcpClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await async_modbus_client.connect()

            assert result is False
            assert async_modbus_client.state == ConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_connect_success(self, async_modbus_client):
        """Test successful connection."""
        with patch("aiomtec2mqtt.async_modbus_client.AsyncModbusTcpClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client_class.return_value = mock_client

            result = await async_modbus_client.connect()

            assert result is True
            assert async_modbus_client.state == ConnectionState.CONNECTED
            assert async_modbus_client._client is not None

    @pytest.mark.asyncio
    async def test_connect_timeout(self, async_modbus_client):
        """Test connection timeout."""
        with patch("aiomtec2mqtt.async_modbus_client.AsyncModbusTcpClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(side_effect=TimeoutError())
            mock_client_class.return_value = mock_client

            with pytest.raises(ModbusConnectionError):
                await async_modbus_client.connect()

            assert async_modbus_client.state == ConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_connection_context_manager(self, async_modbus_client):
        """Test async context manager."""
        with patch("aiomtec2mqtt.async_modbus_client.AsyncModbusTcpClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client.close = MagicMock()
            mock_client_class.return_value = mock_client

            async with async_modbus_client.connection():
                assert async_modbus_client.is_connected

            mock_client.close.assert_called_once()

    def test_decode_register_no_scale(self, async_modbus_client):
        """Test register decoding without scale."""
        mock_response = MagicMock()
        mock_response.registers = [100]

        reg_info = {
            Register.NAME: "battery_soc",
            Register.SCALE: 1,
        }

        result = async_modbus_client._decode_register(
            response=mock_response, offset=0, reg_info=reg_info
        )

        assert result == 100

    def test_decode_register_with_scale(self, async_modbus_client):
        """Test register decoding with scale."""
        mock_response = MagicMock()
        mock_response.registers = [245]  # Raw value 245

        reg_info = {
            Register.NAME: "battery_voltage",
            Register.SCALE: 10,
        }

        result = async_modbus_client._decode_register(
            response=mock_response, offset=0, reg_info=reg_info
        )

        assert result == 24.5  # 245 / 10

    @pytest.mark.asyncio
    async def test_disconnect(self, async_modbus_client):
        """Test disconnection."""
        # Setup connected client
        mock_client = MagicMock()
        async_modbus_client._client = mock_client
        async_modbus_client._state_machine.transition_to(new_state=ConnectionState.CONNECTED)

        await async_modbus_client.disconnect()

        mock_client.close.assert_called_once()
        assert async_modbus_client.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_health_check_integration(self, async_modbus_client, health_check):
        """Test health check integration."""
        with patch("aiomtec2mqtt.async_modbus_client.AsyncModbusTcpClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client_class.return_value = mock_client

            await async_modbus_client.connect()

            component_health = health_check.get_component_health(name="async_modbus")
            assert component_health is not None
            assert component_health.last_success is not None

    @pytest.mark.asyncio
    async def test_initialization(self, async_modbus_client):
        """Test client initialization."""
        assert async_modbus_client._modbus_host == "192.168.1.100"
        assert async_modbus_client._modbus_port == 502
        assert async_modbus_client._modbus_slave == 1
        assert async_modbus_client._modbus_timeout == 5
        assert async_modbus_client.state == ConnectionState.DISCONNECTED
        assert async_modbus_client.error_count == 0

    @pytest.mark.asyncio
    async def test_read_holding_registers_not_connected(self, async_modbus_client):
        """Test read when not connected."""
        with pytest.raises(ModbusReadError) as exc_info:
            await async_modbus_client.read_holding_registers(address=10100, count=1)

        assert "Not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_read_holding_registers_success(self, async_modbus_client):
        """Test successful register read."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.isError.return_value = False
        mock_response.registers = [50, 245, 100]  # Example values
        mock_client.read_holding_registers = AsyncMock(return_value=mock_response)

        async_modbus_client._client = mock_client
        async_modbus_client._state_machine.transition_to(new_state=ConnectionState.CONNECTED)

        result = await async_modbus_client.read_holding_registers(address=10100, count=3)

        assert result == mock_response
        assert not result.isError()

    @pytest.mark.asyncio
    async def test_read_holding_registers_timeout(self, async_modbus_client):
        """Test read timeout."""
        mock_client = AsyncMock()
        mock_client.read_holding_registers = AsyncMock(side_effect=TimeoutError())

        async_modbus_client._client = mock_client
        async_modbus_client._state_machine.transition_to(new_state=ConnectionState.CONNECTED)

        with pytest.raises(ModbusTimeoutError):
            await async_modbus_client.read_holding_registers(address=10100, count=1)

        assert async_modbus_client.error_count > 0

    @pytest.mark.asyncio
    async def test_read_register_group(self, async_modbus_client, register_map):
        """Test reading a register group."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.isError.return_value = False
        mock_response.registers = [
            50,
            245,
            100,
        ]  # battery_soc=50, battery_voltage=24.5V, current=10A
        mock_client.read_holding_registers = AsyncMock(return_value=mock_response)

        async_modbus_client._client = mock_client
        async_modbus_client._state_machine.transition_to(new_state=ConnectionState.CONNECTED)

        result = await async_modbus_client.read_register_group(group_name="BASE")

        assert "battery_soc" in result
        assert "battery_voltage" in result
        assert "battery_current" in result

    @pytest.mark.asyncio
    async def test_register_clustering(self, async_modbus_client):
        """Test register clustering optimization."""
        registers = ["10100", "10101", "10102", "10120", "10121"]

        clusters = async_modbus_client._get_register_clusters(registers=registers)

        # Should create 2 clusters (gap between 10102 and 10120 is > 10)
        assert len(clusters) == 2
        assert clusters[0]["start"] == 10100
        assert clusters[0]["count"] == 3
        assert clusters[1]["start"] == 10120
        assert clusters[1]["count"] == 2
