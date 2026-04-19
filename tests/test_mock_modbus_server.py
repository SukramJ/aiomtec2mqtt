"""Tests for :mod:`aiomtec2mqtt.mock_modbus_server`."""

from __future__ import annotations

import pytest

from aiomtec2mqtt.mock_modbus_server import MockModbusResponse, MockModbusTransport


class TestMockModbusResponse:
    def test_error_response(self) -> None:
        response = MockModbusResponse(registers=[], _is_error=True)
        assert response.isError() is True

    def test_success_response(self) -> None:
        response = MockModbusResponse(registers=[1, 2, 3])
        assert response.registers == [1, 2, 3]
        assert response.isError() is False


class TestMockModbusTransport:
    @pytest.mark.asyncio
    async def test_call_history(self) -> None:
        t = MockModbusTransport()
        await t.connect()
        await t.read_holding_registers(address=100, count=1, slave=1)
        await t.write_registers(address=200, values=[1], slave=1)
        t.close()

        assert len(t.read_calls()) == 1
        assert t.read_calls()[0].address == 100
        assert len(t.write_calls()) == 1
        assert t.write_calls()[0].values == [1]
        kinds = [c.kind for c in t.calls]
        assert kinds == ["connect", "read", "write", "close"]

    @pytest.mark.asyncio
    async def test_close_resets_connection(self) -> None:
        t = MockModbusTransport()
        await t.connect()
        assert t.connected is True
        t.close()
        assert t.connected is False

    @pytest.mark.asyncio
    async def test_connect_failure(self) -> None:
        t = MockModbusTransport(fail_connect=True)
        assert await t.connect() is False
        assert t.connected is False

    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        t = MockModbusTransport()
        assert await t.connect() is True
        assert t.connected is True
        assert t.calls[-1].kind == "connect"

    @pytest.mark.asyncio
    async def test_read_fails_n_times(self) -> None:
        t = MockModbusTransport(
            register_values={10100: [42]},
            fail_read_n_times=2,
        )
        first = await t.read_holding_registers(address=10100, count=1, slave=1)
        second = await t.read_holding_registers(address=10100, count=1, slave=1)
        third = await t.read_holding_registers(address=10100, count=1, slave=1)
        assert first.isError() is True
        assert second.isError() is True
        assert third.isError() is False
        assert third.registers == [42]

    @pytest.mark.asyncio
    async def test_read_pads_short_values(self) -> None:
        t = MockModbusTransport(register_values={10100: [42]})
        response = await t.read_holding_registers(address=10100, count=4, slave=1)
        assert response.registers == [42, 0, 0, 0]

    @pytest.mark.asyncio
    async def test_read_returns_scripted_values(self) -> None:
        t = MockModbusTransport(register_values={10100: [0x1234, 0x5678]})
        await t.connect()
        response = await t.read_holding_registers(address=10100, count=2, slave=1)
        assert response.isError() is False
        assert response.registers == [0x1234, 0x5678]

    @pytest.mark.asyncio
    async def test_read_truncates_long_values(self) -> None:
        t = MockModbusTransport(register_values={10100: [1, 2, 3, 4, 5]})
        response = await t.read_holding_registers(address=10100, count=2, slave=1)
        assert response.registers == [1, 2]

    @pytest.mark.asyncio
    async def test_read_unknown_address_returns_zeroes(self) -> None:
        t = MockModbusTransport()
        await t.connect()
        response = await t.read_holding_registers(address=99999, count=3, slave=1)
        assert response.registers == [0, 0, 0]

    @pytest.mark.asyncio
    async def test_reset_clears_state(self) -> None:
        t = MockModbusTransport()
        await t.connect()
        await t.read_holding_registers(address=100, count=1, slave=1)
        t.reset()
        assert t.calls == []
        assert t.connected is False

    @pytest.mark.asyncio
    async def test_write_failure(self) -> None:
        t = MockModbusTransport(fail_writes=True)
        result = await t.write_registers(address=20000, values=[100], slave=1)
        assert result.isError() is True
        assert 20000 not in t.register_values

    @pytest.mark.asyncio
    async def test_write_persists_values(self) -> None:
        t = MockModbusTransport()
        await t.connect()
        result = await t.write_registers(address=20000, values=[100], slave=1)
        assert result.isError() is False
        assert t.register_values[20000] == [100]
        # Subsequent read sees the written value
        read = await t.read_holding_registers(address=20000, count=1, slave=1)
        assert read.registers == [100]
