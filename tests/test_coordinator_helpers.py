"""Tests for small helper routines and MQTT payload formatting in coordinator."""

from __future__ import annotations

import types
from types import SimpleNamespace
from typing import Any

import pytest

from aiomtec2mqtt import modbus_client as modbus_mod
from aiomtec2mqtt.const import Config, Register, RegisterGroup
from aiomtec2mqtt.mtec_coordinator import (
    MtecCoordinator,
    _convert_code,
    _get_equipment_info,
    _has_bit,
)


class TestHelperFunctions:
    """Tests for coordinator helper functions."""

    def test_get_equipment_info_known_and_unknown(self) -> None:
        """_get_equipment_info should map numeric code pairs to model names or 'unknown'."""
        assert _get_equipment_info(value="31 4") == "5.0K-30A-1P"
        assert _get_equipment_info(value="99 99").lower() == "unknown"

    def test_has_bit_and_convert_code_int_and_bits(self) -> None:
        """_has_bit and _convert_code should work for both int codes and bitstrings."""
        # Simple bits check
        assert _has_bit(val=0b1010, idx=1) is True
        assert _has_bit(val=0b1010, idx=0) is False

        # For int value, direct mapping lookup
        items = {0: "Zero", 1: "One", 2: "Two"}
        assert _convert_code(value=2, value_items=items) == "Two"
        # For bitstring, collect mapped bits; also return OK when none
        assert _convert_code(value="0000 0010", value_items=items) == "One"
        assert _convert_code(value="0000 0000", value_items=items) == "OK"


class TestMqttPayloadFormatting:
    """Tests for MQTT payload formatting."""

    def test_write_to_mqtt_payload_formatting(self) -> None:
        """write_to_mqtt should format float, bool and strings according to rules."""
        published: list[tuple[str, str]] = []

        class DummyMQTT:
            def publish(self, topic: str, payload: str) -> None:  # noqa: D401 - simple recorder
                published.append((topic, payload))

        # Build a lightweight object with attributes consumed by the method
        # Note: _mqtt_float_format should be without braces (e.g., ".2f" not "{:.2f}")
        self_like = SimpleNamespace(_mqtt_float_format=".2f", _mqtt_client=DummyMQTT())

        pvdata = {
            "p1": {Register.VALUE: 3.14159},
            "p2": {Register.VALUE: True},
            "p3": {Register.VALUE: "text"},
        }
        # Call unbound function with our fake self
        MtecCoordinator.write_to_mqtt(
            self_like, pvdata=pvdata, topic_base="base", group=RegisterGroup.PV
        )

        topics = dict(published)
        assert topics["base/now-pv/p1/state"] == "3.14"
        assert topics["base/now-pv/p2/state"] == "1"
        assert topics["base/now-pv/p3/state"] == "text"


class TestModbusClient:
    """Tests for Modbus client operations."""

    def test_modbus_client_read_decode_and_write(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MTECModbusClient should connect, read clusters, decode types, and write registers."""
        # Fake FramerType and ReadHoldingRegistersResponse
        monkeypatch.setattr(modbus_mod, "FramerType", lambda x: x)

        class Resp:
            def __init__(self, regs: list[int], error: bool = False) -> None:
                self.registers = regs
                self._error = error

            def isError(self) -> bool:  # noqa: N802 - API compat
                return self._error

        # Build a fake ModbusTcpClient implementation
        class FakeModbus:
            class DT:
                UINT16 = object()
                INT16 = object()
                UINT32 = object()
                INT32 = object()
                STRING = object()

            DATATYPE = DT()

            def __init__(
                self, host: str, port: int, framer: str, timeout: int, retries: int
            ) -> None:  # noqa: D401 - signature compat
                self._open = False
                self._writes: list[tuple[int, int, int]] = []

            def close(self) -> None:
                self._open = False

            def connect(self) -> bool:
                self._open = True
                return True

            def convert_from_registers(self, registers: list[int], data_type: object) -> int | str:
                if data_type is self.DATATYPE.UINT16:
                    return int(registers[0])
                if data_type is self.DATATYPE.INT16:
                    v = int(registers[0])
                    return v - 0x10000 if v & 0x8000 else v
                if data_type is self.DATATYPE.UINT32:
                    return (int(registers[0]) << 16) + int(registers[1])
                if data_type is self.DATATYPE.INT32:
                    v = (int(registers[0]) << 16) + int(registers[1])
                    return v - 0x1_0000_0000 if v & 0x8000_0000 else v
                if data_type is self.DATATYPE.STRING:
                    # Produce a string with spaces and nulls to exercise rstrip path
                    return "ABC \x00  "
                raise AssertionError("unexpected type")

            def is_socket_open(self) -> bool:
                return self._open

            def read_holding_registers(self, address: int, count: int, device_id: int) -> Resp:
                # Return a "ramp" of count numbers starting at address for predictable decoding
                regs = list(range(address, address + count))
                return Resp(regs)

            def write_register(
                self, address: int, value: int, device_id: int
            ) -> types.SimpleNamespace:
                self._writes.append((address, value, device_id))
                return types.SimpleNamespace(isError=lambda: False)

        monkeypatch.setattr(modbus_mod, "ModbusTcpClient", FakeModbus)

        # Minimal register map covering many decode branches and features
        reg_map = {
            "10000": {
                Register.NAME: "u16",
                Register.LENGTH: 1,
                Register.TYPE: "U16",
                Register.UNIT: "x",
            },
            "10001": {Register.NAME: "i16", Register.LENGTH: 1, Register.TYPE: "I16"},
            "10002": {
                Register.NAME: "u32",
                Register.LENGTH: 2,
                Register.TYPE: "U32",
                Register.SCALE: 10,
            },
            "10004": {Register.NAME: "i32", Register.LENGTH: 2, Register.TYPE: "I32"},
            "10006": {Register.NAME: "byte1", Register.LENGTH: 1, Register.TYPE: "BYTE"},
            "10007": {Register.NAME: "byte2", Register.LENGTH: 2, Register.TYPE: "BYTE"},
            "10009": {Register.NAME: "bit2", Register.LENGTH: 2, Register.TYPE: "BIT"},
            "10011": {Register.NAME: "dat", Register.LENGTH: 3, Register.TYPE: "DAT"},
            "10014": {Register.NAME: "str", Register.LENGTH: 2, Register.TYPE: "STR"},
            # writable with scale and mqtt name + value_items mapping
            "20000": {
                Register.NAME: "set",
                Register.LENGTH: 1,
                Register.TYPE: "U16",
                Register.SCALE: 2,
                Register.WRITABLE: True,
                Register.MQTT: "set",
                Register.VALUE_ITEMS: {0: "Off", 1: "On"},
            },
        }

        groups = ["now-base"]
        cfg = {
            Config.MODBUS_FRAMER: "socket",
            Config.MODBUS_IP: "host",
            Config.MODBUS_PORT: 502,
            Config.MODBUS_RETRIES: 1,
            Config.MODBUS_SLAVE: 1,
            Config.MODBUS_TIMEOUT: 5,
        }

        api = modbus_mod.MTECModbusClient(config=cfg, register_map=reg_map, register_groups=groups)
        assert api.connect() is True

        # Read all registers (None) triggers cluster generation and decoding
        data = api.read_modbus_data()
        # sanity checks for a few decodes and scaling
        assert data["10000"][Register.VALUE] == 10000
        # For our fake data, INT16 decoding of 10001 stays positive because the sign bit isn't set
        assert data["10001"][Register.VALUE] == 10001
        # u32 scaled by 10: raw is (10002<<16)+10003 => 655501075, scaled => 65550107.5
        assert abs(data["10002"][Register.VALUE] - 65550107.5) < 1e-6
        assert " " in data["10006"][Register.VALUE]  # BYTE string
        assert data["10014"][Register.VALUE] == "ABC"  # stripped

        # Test write by name with display value mapping and scale handling
        assert api.write_register_by_name(name="set", value="On") is True

        # Direct write with invalid register
        assert api.write_register(register="99999", value=1) is False

        # Read specific registers triggers caching key path
        cluster = api._get_register_clusters(registers=["10000", "10001", "XYZ"])  # noqa: SLF001
        assert isinstance(cluster, list)

        api.disconnect()

    def test_modbus_client_write_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """write_register should validate writable flag and numeric value conversion/errors."""
        monkeypatch.setattr(modbus_mod, "FramerType", lambda x: x)

        class FakeModbusErr:
            DATATYPE = types.SimpleNamespace()

            def __init__(self, *a: Any, **k: Any) -> None:
                pass

            def close(self) -> None:  # noqa: D401 - no-op
                pass

            def connect(self) -> bool:
                return True

            def is_socket_open(self) -> bool:
                return True

            def write_register(
                self, address: int, value: int, device_id: int
            ) -> types.SimpleNamespace:
                # Simulate an error response
                return types.SimpleNamespace(isError=lambda: True)

        monkeypatch.setattr(modbus_mod, "ModbusTcpClient", FakeModbusErr)

        reg_map = {
            "1": {
                Register.NAME: "ro",
                Register.LENGTH: 1,
                Register.TYPE: "U16",
                Register.WRITABLE: False,
            },
            "2": {
                Register.NAME: "rw",
                Register.LENGTH: 1,
                Register.TYPE: "U16",
                Register.WRITABLE: True,
            },
        }
        cfg = {
            Config.MODBUS_FRAMER: "socket",
            Config.MODBUS_IP: "host",
            Config.MODBUS_PORT: 502,
            Config.MODBUS_RETRIES: 1,
            Config.MODBUS_SLAVE: 1,
            Config.MODBUS_TIMEOUT: 5,
        }
        api = modbus_mod.MTECModbusClient(config=cfg, register_map=reg_map, register_groups=["g"])  # type: ignore[list-item]

        # RO register
        assert api.write_register(register="1", value=1) is False
        # Invalid numeric string
        assert api.write_register(register="2", value="not-a-number") is False
        # Error response from client
        assert api.write_register(register="2", value=1) is False


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_util_list_functions_exercise_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """The utility list functions should iterate and log entries using a fake API object."""
        import logging

        caplog.set_level(logging.INFO, logger="aiomtec2mqtt.util.mtec_util")

        class FakeApi:
            register_groups = [RegisterGroup.PV, RegisterGroup.GRID]
            register_map = {
                "10": {
                    Register.NAME: "A",
                    Register.MQTT: "mqtt_a",
                    Register.UNIT: "W",
                    Register.GROUP: RegisterGroup.PV,
                    Register.WRITABLE: False,
                },
                "11": {
                    Register.NAME: "B",
                    Register.MQTT: None,
                    Register.UNIT: "",
                    Register.GROUP: RegisterGroup.GRID,
                    Register.WRITABLE: True,
                },
                "x": {
                    Register.NAME: "Calc",
                    Register.MQTT: "calc",
                    Register.UNIT: "V",
                    Register.GROUP: None,
                    Register.WRITABLE: False,
                },
            }

            def get_register_list(self, group: RegisterGroup) -> list[str]:  # noqa: D401 - fake API
                return [k for k, v in self.register_map.items() if v[Register.GROUP] == group]

            def read_modbus_data(
                self, registers: list[str] | None = None
            ) -> dict[str, dict[str, Any]]:  # noqa: D401 - fake API
                regs = registers or list(self.register_map)
                return {
                    r: {
                        Register.NAME: self.register_map[r][Register.NAME],
                        Register.VALUE: 1,
                        Register.UNIT: self.register_map[r][Register.UNIT],
                    }
                    for r in regs
                    if r in self.register_map
                }

        from aiomtec2mqtt.util import mtec_util as util

        api = FakeApi()
        util.list_register_config(api=api)  # should log lines
        util.list_register_config_by_groups(api=api)  # should log per-group listings
        # smoke test: no exceptions and at least one log record produced
        assert caplog.records != []
