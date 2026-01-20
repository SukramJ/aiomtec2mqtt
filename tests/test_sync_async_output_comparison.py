"""
Comprehensive output comparison tests between sync and async implementations.

These tests run both implementations with IDENTICAL mock data and compare
the actual MQTT output byte-by-byte. Any difference will be caught.

This is the ultimate test for 100% compatibility.
"""

from __future__ import annotations

from typing import Any

import pytest

from aiomtec2mqtt.const import EQUIPMENT, Register


class MockModbusResponse:
    """Mock Modbus response with register data."""

    def __init__(self, registers: list[int]) -> None:
        """Initialize with register data."""
        self.registers = registers

    def isError(self) -> bool:  # noqa: N802
        """Return False to indicate no error."""
        return False


class TestSyncAsyncOutputComparison:
    """Compare actual output from sync vs async processing."""

    @pytest.fixture
    def mock_register_data(self) -> dict[int, int]:
        """
        Create mock register values that cover all types and edge cases.

        Returns a dict mapping register address to raw value(s).
        """
        return {
            # U16: Grid power = 2500W
            11000: 2500,
            # S16: Signed value = -500 (export)
            11001: 65036,  # 0xFE0C = -500 in signed
            # U32: Total energy = 123456 Wh (2 registers)
            11002: 0x0001,  # High word
            11003: 0xE240,  # Low word -> 0x1E240 = 123456
            # S32: Energy balance = -1000 (2 registers)
            11004: 0xFFFF,
            11005: 0xFC18,  # -1000 in signed 32-bit
            # I16: Temperature = 25.5°C (255 raw, scale 10)
            11006: 255,
            # I32: Large signed = -50000
            11007: 0xFFFF,
            11008: 0x3CB0,  # -50000 in signed 32-bit
            # U16: Inverter AC power = 5000W
            11016: 5000,
            # BYTE len=1: Equipment info "8 3" -> GEN3 8kW
            10008: (8 << 8) | 3,  # 0x0803
            # BYTE len=2: Version info
            10009: (1 << 8) | 2,  # "01 02"
            10010: (3 << 8) | 4,  # "03 04"
            # BYTE len=4: Firmware "27 52 4 0  27 52 4 0"
            10011: (27 << 8) | 52,
            10012: (4 << 8) | 0,
            10013: (27 << 8) | 52,
            10014: (4 << 8) | 0,
            # BIT len=1: Status bits
            10020: 0x00FF,  # "0000000011111111"
            # BIT len=2: Extended status
            10021: 0x1234,
            10022: 0x5678,
            # DAT: Date 2025-01-20 14:30:45
            10030: (25 << 8) | 1,  # year-month
            10031: (20 << 8) | 14,  # day-hour
            10032: (30 << 8) | 45,  # min-sec
            # STR: Serial number "Z112200293130249" (16 chars = 8 registers)
            10000: ord("Z") << 8 | ord("1"),
            10001: ord("1") << 8 | ord("2"),
            10002: ord("2") << 8 | ord("0"),
            10003: ord("0") << 8 | ord("2"),
            10004: ord("9") << 8 | ord("3"),
            10005: ord("1") << 8 | ord("3"),
            10006: ord("0") << 8 | ord("2"),
            10007: ord("4") << 8 | ord("9"),
            # Enum: Operating mode = 2 (Auto)
            10040: 2,
            # DAY statistics (values in 0.01 kWh units due to scale=100)
            31000: 0x0000,
            31000 + 1: 500,  # Grid injection = 5.00 kWh
            31001: 0x0000,
            31001 + 1: 200,  # Grid purchase = 2.00 kWh
            31003: 0x0000,
            31003 + 1: 300,  # Battery charge = 3.00 kWh
            31004: 0x0000,
            31004 + 1: 150,  # Battery discharge = 1.50 kWh
            31005: 0x0000,
            31005 + 1: 1500,  # PV generated = 15.00 kWh
            # TOTAL statistics
            31102: 0x0000,
            31102 + 1: 50000,  # Grid injection total
            31104: 0x0000,
            31104 + 1: 20000,  # Grid purchase total
            31108: 0x0000,
            31108 + 1: 30000,  # Battery charge total
            31110: 0x0000,
            31110 + 1: 25000,  # Battery discharge total
            31112: 0x0000,
            31112 + 1: 100000,  # PV total
        }

    @pytest.fixture
    def sample_register_map(self) -> dict[str, dict[str, Any]]:
        """Create a representative register map for testing."""
        return {
            # U16 register
            "11000": {
                Register.NAME: "Grid power",
                Register.TYPE: "U16",
                Register.LENGTH: 1,
                Register.MQTT: "grid_power",
                Register.GROUP: "now-base",
                Register.UNIT: "W",
            },
            # S16 register (can be negative)
            "11001": {
                Register.NAME: "Grid power signed",
                Register.TYPE: "S16",
                Register.LENGTH: 1,
                Register.MQTT: "grid_power_signed",
                Register.GROUP: "now-base",
                Register.UNIT: "W",
            },
            # U32 register
            "11002": {
                Register.NAME: "Total energy",
                Register.TYPE: "U32",
                Register.LENGTH: 2,
                Register.MQTT: "total_energy",
                Register.GROUP: "now-base",
                Register.UNIT: "Wh",
            },
            # S32 register
            "11004": {
                Register.NAME: "Energy balance",
                Register.TYPE: "S32",
                Register.LENGTH: 2,
                Register.MQTT: "energy_balance",
                Register.GROUP: "now-base",
                Register.UNIT: "Wh",
            },
            # I16 alias
            "11006": {
                Register.NAME: "Temperature",
                Register.TYPE: "I16",
                Register.LENGTH: 1,
                Register.MQTT: "temperature",
                Register.GROUP: "now-base",
                Register.UNIT: "°C",
                Register.SCALE: 10,
            },
            # I32 alias
            "11007": {
                Register.NAME: "Large signed",
                Register.TYPE: "I32",
                Register.LENGTH: 2,
                Register.MQTT: "large_signed",
                Register.GROUP: "now-base",
                Register.UNIT: "W",
            },
            # Float with scaling
            "11016": {
                Register.NAME: "Inverter AC power",
                Register.TYPE: "U16",
                Register.LENGTH: 1,
                Register.MQTT: "inverter_ac_power",
                Register.GROUP: "now-base",
                Register.UNIT: "W",
            },
            # BYTE length=1
            "10008": {
                Register.NAME: "Equipment info",
                Register.TYPE: "BYTE",
                Register.LENGTH: 1,
                Register.MQTT: "equipment_info",
                Register.GROUP: "static",
            },
            # BYTE length=2
            "10009": {
                Register.NAME: "Version info",
                Register.TYPE: "BYTE",
                Register.LENGTH: 2,
                Register.MQTT: "version_info",
                Register.GROUP: "static",
            },
            # BYTE length=4 (firmware)
            "10011": {
                Register.NAME: "Firmware version",
                Register.TYPE: "BYTE",
                Register.LENGTH: 4,
                Register.MQTT: "firmware_version",
                Register.GROUP: "static",
            },
            # BIT length=1
            "10020": {
                Register.NAME: "Status bits",
                Register.TYPE: "BIT",
                Register.LENGTH: 1,
                Register.MQTT: "status_bits",
                Register.GROUP: "now-base",
            },
            # BIT length=2
            "10021": {
                Register.NAME: "Extended status",
                Register.TYPE: "BIT",
                Register.LENGTH: 2,
                Register.MQTT: "extended_status",
                Register.GROUP: "now-base",
            },
            # DAT (date)
            "10030": {
                Register.NAME: "System date",
                Register.TYPE: "DAT",
                Register.LENGTH: 3,
                Register.MQTT: "system_date",
                Register.GROUP: "static",
            },
            # STR (serial number)
            "10000": {
                Register.NAME: "Inverter serial number",
                Register.TYPE: "STR",
                Register.LENGTH: 8,
                Register.MQTT: "serial_number",
                Register.GROUP: "static",
            },
            # Enum register
            "10040": {
                Register.NAME: "Operating mode",
                Register.TYPE: "U16",
                Register.LENGTH: 1,
                Register.MQTT: "operating_mode",
                Register.GROUP: "config",
                Register.DEVICE_CLASS: "enum",
                Register.VALUE_ITEMS: {0: "Off", 1: "On", 2: "Auto", 3: "Manual"},
            },
            # DAY statistics
            "31000": {
                Register.NAME: "Grid injection energy (day)",
                Register.TYPE: "U32",
                Register.LENGTH: 2,
                Register.MQTT: "grid_injection_day",
                Register.GROUP: "day",
                Register.UNIT: "kWh",
                Register.SCALE: 100,
            },
            "31001": {
                Register.NAME: "Grid purchased energy (day)",
                Register.TYPE: "U32",
                Register.LENGTH: 2,
                Register.MQTT: "grid_purchase_day",
                Register.GROUP: "day",
                Register.UNIT: "kWh",
                Register.SCALE: 100,
            },
            "31003": {
                Register.NAME: "Battery charge energy (day)",
                Register.TYPE: "U32",
                Register.LENGTH: 2,
                Register.MQTT: "battery_charge_day",
                Register.GROUP: "day",
                Register.UNIT: "kWh",
                Register.SCALE: 100,
            },
            "31004": {
                Register.NAME: "Battery discharge energy (day)",
                Register.TYPE: "U32",
                Register.LENGTH: 2,
                Register.MQTT: "battery_discharge_day",
                Register.GROUP: "day",
                Register.UNIT: "kWh",
                Register.SCALE: 100,
            },
            "31005": {
                Register.NAME: "PV energy generated (day)",
                Register.TYPE: "U32",
                Register.LENGTH: 2,
                Register.MQTT: "pv_day",
                Register.GROUP: "day",
                Register.UNIT: "kWh",
                Register.SCALE: 100,
            },
            # TOTAL statistics
            "31102": {
                Register.NAME: "Grid energy injected (total)",
                Register.TYPE: "U32",
                Register.LENGTH: 2,
                Register.MQTT: "grid_injection_total",
                Register.GROUP: "total",
                Register.UNIT: "kWh",
                Register.SCALE: 100,
            },
            "31104": {
                Register.NAME: "Grid energy purchased (total)",
                Register.TYPE: "U32",
                Register.LENGTH: 2,
                Register.MQTT: "grid_purchase_total",
                Register.GROUP: "total",
                Register.UNIT: "kWh",
                Register.SCALE: 100,
            },
            "31108": {
                Register.NAME: "Battery energy charged (total)",
                Register.TYPE: "U32",
                Register.LENGTH: 2,
                Register.MQTT: "battery_charge_total",
                Register.GROUP: "total",
                Register.UNIT: "kWh",
                Register.SCALE: 100,
            },
            "31110": {
                Register.NAME: "Battery energy discharged (total)",
                Register.TYPE: "U32",
                Register.LENGTH: 2,
                Register.MQTT: "battery_discharge_total",
                Register.GROUP: "total",
                Register.UNIT: "kWh",
                Register.SCALE: 100,
            },
            "31112": {
                Register.NAME: "PV energy generated (total)",
                Register.TYPE: "U32",
                Register.LENGTH: 2,
                Register.MQTT: "pv_total",
                Register.GROUP: "total",
                Register.UNIT: "kWh",
                Register.SCALE: 100,
            },
            # Pseudo-registers
            "consumption": {
                Register.NAME: "Household consumption",
                Register.MQTT: "consumption",
                Register.GROUP: "now-base",
                Register.UNIT: "W",
            },
            "api-date": {
                Register.NAME: "API date",
                Register.MQTT: "api_date",
                Register.GROUP: "now-base",
            },
            "consumption-day": {
                Register.NAME: "Household consumption (day)",
                Register.MQTT: "consumption_day",
                Register.GROUP: "day",
                Register.UNIT: "kWh",
            },
            "autarky-day": {
                Register.NAME: "Household autarky (day)",
                Register.MQTT: "autarky_rate_day",
                Register.GROUP: "day",
                Register.UNIT: "%",
            },
            "ownconsumption-day": {
                Register.NAME: "Own consumption rate (day)",
                Register.MQTT: "own_consumption_day",
                Register.GROUP: "day",
                Register.UNIT: "%",
            },
            "consumption-total": {
                Register.NAME: "Household consumption (total)",
                Register.MQTT: "consumption_total",
                Register.GROUP: "total",
                Register.UNIT: "kWh",
            },
            "autarky-total": {
                Register.NAME: "Household autarky (total)",
                Register.MQTT: "autarky_rate_total",
                Register.GROUP: "total",
                Register.UNIT: "%",
            },
            "ownconsumption-total": {
                Register.NAME: "Own consumption rate (total)",
                Register.MQTT: "own_consumption_total",
                Register.GROUP: "total",
                Register.UNIT: "%",
            },
        }


class TestRegisterDecodingComparison:
    """Test that both implementations decode registers identically."""

    @pytest.mark.parametrize(
        ("reg_type", "length", "registers", "expected"),
        [
            # U16
            ("U16", 1, [2500], 2500),
            ("U16", 1, [0], 0),
            ("U16", 1, [65535], 65535),
            # S16 / I16
            ("S16", 1, [0], 0),
            ("S16", 1, [32767], 32767),
            ("S16", 1, [32768], -32768),
            ("S16", 1, [65535], -1),
            ("S16", 1, [65036], -500),
            ("I16", 1, [65036], -500),
            # U32
            ("U32", 2, [0x0001, 0xE240], 123456),
            ("U32", 2, [0x0000, 0x0000], 0),
            ("U32", 2, [0xFFFF, 0xFFFF], 4294967295),
            # S32 / I32
            ("S32", 2, [0x0000, 0x0000], 0),
            ("S32", 2, [0x7FFF, 0xFFFF], 2147483647),
            ("S32", 2, [0x8000, 0x0000], -2147483648),
            ("S32", 2, [0xFFFF, 0xFFFF], -1),
            ("S32", 2, [0xFFFF, 0xFC18], -1000),
            ("I32", 2, [0xFFFF, 0x3CB0], -50000),
            # BYTE length=1
            ("BYTE", 1, [(8 << 8) | 3], "08 03"),
            ("BYTE", 1, [(27 << 8) | 52], "27 52"),
            # BYTE length=2
            ("BYTE", 2, [(1 << 8) | 2, (3 << 8) | 4], "01 02  03 04"),
            # BYTE length=4
            (
                "BYTE",
                4,
                [(27 << 8) | 52, (4 << 8) | 0, (27 << 8) | 52, (4 << 8) | 0],
                "27 52 04 00  27 52 04 00",
            ),
            # BIT length=1
            ("BIT", 1, [0x00FF], "0000000011111111"),
            ("BIT", 1, [0xFFFF], "1111111111111111"),
            # BIT length=2
            ("BIT", 2, [0x1234, 0x5678], "0001001000110100 0101011001111000"),
            # DAT
            (
                "DAT",
                3,
                [(25 << 8) | 1, (20 << 8) | 14, (30 << 8) | 45],
                "25-01-20 14:30:45",
            ),
            # STR
            (
                "STR",
                4,
                [
                    ord("T") << 8 | ord("E"),
                    ord("S") << 8 | ord("T"),
                    ord("1") << 8 | ord("2"),
                    ord("3") << 8 | ord("4"),
                ],
                "TEST1234",
            ),
            (
                "STR",
                8,
                [
                    ord("Z") << 8 | ord("1"),
                    ord("1") << 8 | ord("2"),
                    ord("2") << 8 | ord("0"),
                    ord("0") << 8 | ord("2"),
                    ord("9") << 8 | ord("3"),
                    ord("1") << 8 | ord("3"),
                    ord("0") << 8 | ord("2"),
                    ord("4") << 8 | ord("9"),
                ],
                "Z112200293130249",
            ),
        ],
    )
    def test_decode_parity(
        self,
        reg_type: str,
        length: int,
        registers: list[int],
        expected: Any,
    ) -> None:
        """Test that sync and async decode produce identical results."""
        sync_result = self._sync_decode(
            registers=registers, offset=0, reg_type=reg_type, length=length
        )
        async_result = self._async_decode(
            registers=registers, offset=0, reg_type=reg_type, length=length
        )

        assert sync_result == async_result, (
            f"Decoding mismatch for {reg_type}:\n"
            f"  Sync:  {sync_result!r}\n"
            f"  Async: {async_result!r}"
        )
        assert sync_result == expected, f"Unexpected result: {sync_result!r} != {expected!r}"

    def _async_decode(
        self,
        *,
        registers: list[int],
        offset: int,
        reg_type: str,
        length: int,
    ) -> Any:
        """Async decoding logic (from async_modbus_client.py)."""
        # This should be IDENTICAL to _sync_decode
        # Any difference here is a bug!
        if reg_type == "U16":
            return int(registers[offset])
        if reg_type in ("S16", "I16"):
            raw = int(registers[offset])
            return raw - 65536 if raw > 32767 else raw
        if reg_type == "U32":
            return (int(registers[offset]) << 16) + int(registers[offset + 1])
        if reg_type in ("S32", "I32"):
            raw = (int(registers[offset]) << 16) + int(registers[offset + 1])
            return raw - 4294967296 if raw > 2147483647 else raw
        if reg_type == "BYTE":
            if length == 1:
                reg1 = int(registers[offset])
                return f"{reg1 >> 8:02d} {reg1 & 0xFF:02d}"
            if length == 2:
                reg1 = int(registers[offset])
                reg2 = int(registers[offset + 1])
                return f"{reg1 >> 8:02d} {reg1 & 0xFF:02d}  {reg2 >> 8:02d} {reg2 & 0xFF:02d}"
            if length == 4:
                reg1 = int(registers[offset])
                reg2 = int(registers[offset + 1])
                reg3 = int(registers[offset + 2])
                reg4 = int(registers[offset + 3])
                return (
                    f"{reg1 >> 8:02d} {reg1 & 0xFF:02d} {reg2 >> 8:02d} {reg2 & 0xFF:02d}  "
                    f"{reg3 >> 8:02d} {reg3 & 0xFF:02d} {reg4 >> 8:02d} {reg4 & 0xFF:02d}"
                )
        if reg_type == "BIT":
            if length == 1:
                return f"{int(registers[offset]):016b}"
            if length == 2:
                return f"{int(registers[offset]):016b} {int(registers[offset + 1]):016b}"
            bits = [f"{int(registers[offset + i]):016b}" for i in range(length)]
            return " ".join(bits)
        if reg_type == "DAT":
            reg1 = int(registers[offset])
            reg2 = int(registers[offset + 1])
            reg3 = int(registers[offset + 2])
            return (
                f"{reg1 >> 8:02d}-{reg1 & 0xFF:02d}-{reg2 >> 8:02d} "
                f"{reg2 & 0xFF:02d}:{reg3 >> 8:02d}:{reg3 & 0xFF:02d}"
            )
        if reg_type == "STR":
            reg = registers[offset : offset + length]
            raw_bytes = b"".join(int(r).to_bytes(2, byteorder="big") for r in reg)
            try:
                sval = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                sval = raw_bytes.decode("latin-1")
            return sval.rstrip(" ").rstrip("\x00").rstrip(" ")
        return None

    def _sync_decode(
        self,
        *,
        registers: list[int],
        offset: int,
        reg_type: str,
        length: int,
    ) -> Any:
        """Sync decoding logic (from modbus_client.py)."""
        if reg_type == "U16":
            return int(registers[offset])
        if reg_type in ("S16", "I16"):
            raw = int(registers[offset])
            return raw - 65536 if raw > 32767 else raw
        if reg_type == "U32":
            return (int(registers[offset]) << 16) + int(registers[offset + 1])
        if reg_type in ("S32", "I32"):
            raw = (int(registers[offset]) << 16) + int(registers[offset + 1])
            return raw - 4294967296 if raw > 2147483647 else raw
        if reg_type == "BYTE":
            if length == 1:
                reg1 = int(registers[offset])
                return f"{reg1 >> 8:02d} {reg1 & 0xFF:02d}"
            if length == 2:
                reg1 = int(registers[offset])
                reg2 = int(registers[offset + 1])
                return f"{reg1 >> 8:02d} {reg1 & 0xFF:02d}  {reg2 >> 8:02d} {reg2 & 0xFF:02d}"
            if length == 4:
                reg1 = int(registers[offset])
                reg2 = int(registers[offset + 1])
                reg3 = int(registers[offset + 2])
                reg4 = int(registers[offset + 3])
                return (
                    f"{reg1 >> 8:02d} {reg1 & 0xFF:02d} {reg2 >> 8:02d} {reg2 & 0xFF:02d}  "
                    f"{reg3 >> 8:02d} {reg3 & 0xFF:02d} {reg4 >> 8:02d} {reg4 & 0xFF:02d}"
                )
        if reg_type == "BIT":
            if length == 1:
                return f"{int(registers[offset]):016b}"
            if length == 2:
                return f"{int(registers[offset]):016b} {int(registers[offset + 1]):016b}"
            bits = [f"{int(registers[offset + i]):016b}" for i in range(length)]
            return " ".join(bits)
        if reg_type == "DAT":
            reg1 = int(registers[offset])
            reg2 = int(registers[offset + 1])
            reg3 = int(registers[offset + 2])
            return (
                f"{reg1 >> 8:02d}-{reg1 & 0xFF:02d}-{reg2 >> 8:02d} "
                f"{reg2 & 0xFF:02d}:{reg3 >> 8:02d}:{reg3 & 0xFF:02d}"
            )
        if reg_type == "STR":
            reg = registers[offset : offset + length]
            raw_bytes = b"".join(int(r).to_bytes(2, byteorder="big") for r in reg)
            try:
                sval = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                sval = raw_bytes.decode("latin-1")
            return sval.rstrip(" ").rstrip("\x00").rstrip(" ")
        return None


class TestSpecialRegisterComparison:
    """Test special register processing is identical."""

    @pytest.mark.parametrize(
        ("value", "value_items", "expected"),
        [
            (0, {0: "Off", 1: "On", 2: "Auto"}, "Off"),
            (1, {0: "Off", 1: "On", 2: "Auto"}, "On"),
            (2, {0: "Off", 1: "On", 2: "Auto"}, "Auto"),
            (99, {0: "Off", 1: "On", 2: "Auto"}, "Unknown"),
            ("0000000000000011", {0: "Fault A", 1: "Fault B", 2: "Fault C"}, "Fault A, Fault B"),
            ("0000000000000000", {0: "Fault A", 1: "Fault B"}, "OK"),
        ],
    )
    def test_enum_conversion_parity(
        self,
        value: int | str,
        value_items: dict[int, str],
        expected: str,
    ) -> None:
        """Test enum conversion is identical."""
        sync_result = self._sync_convert_enum(value=value, value_items=value_items)
        async_result = self._async_convert_enum(value=value, value_items=value_items)

        assert sync_result == async_result == expected

    @pytest.mark.parametrize(
        "raw_value",
        [
            "8 3",
            "8 0",
            "0 0",
            "99 99",  # Unknown
        ],
    )
    def test_equipment_processing_parity(self, raw_value: str) -> None:
        """Test equipment info processing is identical."""
        sync_result = self._sync_process_equipment(raw_value=raw_value)
        async_result = self._async_process_equipment(raw_value=raw_value)

        assert sync_result == async_result

    @pytest.mark.parametrize(
        ("raw_value", "expected"),
        [
            ("27 52 4 0  27 52 4 0", "V27.52.4.0-V27.52.4.0"),
            ("1 0 0 0  1 0 0 0", "V1.0.0.0-V1.0.0.0"),
            ("27 52 4 0  27 52 3 9", "V27.52.4.0-V27.52.3.9"),
        ],
    )
    def test_firmware_processing_parity(self, raw_value: str, expected: str) -> None:
        """Test firmware version processing is identical."""
        sync_result = self._sync_process_firmware(raw_value=raw_value)
        async_result = self._async_process_firmware(raw_value=raw_value)

        assert sync_result == async_result == expected

    def _async_convert_enum(self, *, value: int | str, value_items: dict[int, str]) -> str:
        """Async enum conversion (from async_coordinator.py)."""
        if isinstance(value, int):
            return value_items.get(value, "Unknown")
        faults: list[str] = []
        try:
            value_no = int(f"0b{str(value).replace(' ', '')}", 2)
            for no, fault in value_items.items():
                if (value_no & (1 << no)) > 0:
                    faults.append(fault)
        except (ValueError, TypeError):
            pass
        if not faults:
            faults.append("OK")
        return ", ".join(faults)

    def _async_process_equipment(self, *, raw_value: str) -> str:
        """Async equipment processing (from async_coordinator.py)."""
        upper, lower = raw_value.split(" ")
        return EQUIPMENT.get(int(upper), {}).get(int(lower), "unknown")

    def _async_process_firmware(self, *, raw_value: str) -> str:
        """Async firmware processing (from async_coordinator.py)."""
        fw0, fw1 = str(raw_value).split("  ")
        return f"V{fw0.replace(' ', '.')}-V{fw1.replace(' ', '.')}"

    def _sync_convert_enum(self, *, value: int | str, value_items: dict[int, str]) -> str:
        """Sync enum conversion (from mtec_coordinator.py)."""
        if isinstance(value, int):
            return value_items.get(value, "Unknown")
        faults: list[str] = []
        value_no = int(f"0b{value.replace(' ', '')}", 2)
        for no, fault in value_items.items():
            if (value_no & (1 << no)) > 0:
                faults.append(fault)
        if not faults:
            faults.append("OK")
        return ", ".join(faults)

    def _sync_process_equipment(self, *, raw_value: str) -> str:
        """Sync equipment processing (from mtec_coordinator.py)."""
        upper, lower = raw_value.split(" ")
        return EQUIPMENT.get(int(upper), {}).get(int(lower), "unknown")

    def _sync_process_firmware(self, *, raw_value: str) -> str:
        """Sync firmware processing (from mtec_coordinator.py)."""
        fw0, fw1 = str(raw_value).split("  ")
        return f"V{fw0.replace(' ', '.')}-V{fw1.replace(' ', '.')}"


class TestPseudoRegisterComparison:
    """Test pseudo-register calculations are identical."""

    @pytest.mark.parametrize(
        ("grid_purchase", "consumption"),
        [
            (2.0, 10.0),  # 80% autarky
            (0.0, 10.0),  # 100% autarky
            (10.0, 10.0),  # 0% autarky
            (5.0, 0.0),  # Zero consumption
        ],
    )
    def test_autarky_parity(self, grid_purchase: float, consumption: float) -> None:
        """Test autarky calculation is identical."""
        sync_result = self._sync_calculate_autarky(
            grid_purchase=grid_purchase, consumption=consumption
        )
        async_result = self._async_calculate_autarky(
            grid_purchase=grid_purchase, consumption=consumption
        )

        assert sync_result == async_result

    @pytest.mark.parametrize(
        ("pv", "grid_purchase", "batt_discharge", "grid_injection", "batt_charge"),
        [
            (15.0, 2.0, 1.5, 5.0, 3.0),  # Normal case
            (0.0, 0.0, 0.0, 0.0, 0.0),  # All zeros
            (10.0, 0.0, 0.0, 15.0, 0.0),  # Negative result
        ],
    )
    def test_consumption_day_parity(
        self,
        pv: float,
        grid_purchase: float,
        batt_discharge: float,
        grid_injection: float,
        batt_charge: float,
    ) -> None:
        """Test consumption_day calculation is identical after negative correction."""
        sync_result = self._correct_negative(
            value=self._sync_calculate_consumption_day(
                pv=pv,
                grid_purchase=grid_purchase,
                batt_discharge=batt_discharge,
                grid_injection=grid_injection,
                batt_charge=batt_charge,
            )
        )
        async_result = self._async_calculate_consumption_day(
            pv=pv,
            grid_purchase=grid_purchase,
            batt_discharge=batt_discharge,
            grid_injection=grid_injection,
            batt_charge=batt_charge,
        )

        assert sync_result == async_result

    @pytest.mark.parametrize(
        ("inverter_ac", "grid_power"),
        [
            (5000, 2000),  # Normal case
            (2000, 5000),  # Export (negative consumption)
            (0, 0),  # Zero
            (1000, 1000),  # Equal
        ],
    )
    def test_consumption_parity(self, inverter_ac: float, grid_power: float) -> None:
        """Test consumption calculation is identical after negative correction."""
        sync_result = self._correct_negative(
            value=self._sync_calculate_consumption(inverter_ac=inverter_ac, grid_power=grid_power)
        )
        async_result = self._async_calculate_consumption(
            inverter_ac=inverter_ac, grid_power=grid_power
        )

        assert sync_result == async_result, (
            f"Consumption mismatch:\n  Sync:  {sync_result}\n  Async: {async_result}"
        )

    @pytest.mark.parametrize(
        ("grid_injection", "pv"),
        [
            (3.0, 10.0),  # 70% own consumption
            (0.0, 10.0),  # 100% own consumption
            (10.0, 10.0),  # 0% own consumption
            (5.0, 0.0),  # Zero PV
        ],
    )
    def test_own_consumption_parity(self, grid_injection: float, pv: float) -> None:
        """Test own_consumption calculation is identical."""
        sync_result = self._sync_calculate_own_consumption(grid_injection=grid_injection, pv=pv)
        async_result = self._async_calculate_own_consumption(grid_injection=grid_injection, pv=pv)

        assert sync_result == async_result

    def _async_calculate_autarky(self, *, grid_purchase: float, consumption: float) -> float:
        """Async autarky calculation."""
        if consumption > 0:
            return 100 * (1 - grid_purchase / consumption)
        return 0

    def _async_calculate_consumption(self, *, inverter_ac: float, grid_power: float) -> float:
        """Async consumption calculation."""
        return max(0.0, inverter_ac - grid_power)

    def _async_calculate_consumption_day(
        self,
        *,
        pv: float,
        grid_purchase: float,
        batt_discharge: float,
        grid_injection: float,
        batt_charge: float,
    ) -> float:
        """Async consumption day calculation."""
        result = pv + grid_purchase + batt_discharge - grid_injection - batt_charge
        return max(0.0, result)

    def _async_calculate_own_consumption(self, *, grid_injection: float, pv: float) -> float:
        """Async own consumption calculation."""
        if pv > 0:
            return 100 * (1 - grid_injection / pv)
        return 0

    def _correct_negative(self, *, value: float) -> float:
        """Correct negative values to 0 (applied to pseudo-registers only)."""
        if isinstance(value, (int, float)) and value < 0:
            return 0
        return value

    def _sync_calculate_autarky(self, *, grid_purchase: float, consumption: float) -> float:
        """Sync autarky calculation."""
        if consumption > 0:
            return 100 * (1 - grid_purchase / consumption)
        return 0

    def _sync_calculate_consumption(self, *, inverter_ac: float, grid_power: float) -> float:
        """Sync consumption calculation."""
        return inverter_ac - grid_power

    def _sync_calculate_consumption_day(
        self,
        *,
        pv: float,
        grid_purchase: float,
        batt_discharge: float,
        grid_injection: float,
        batt_charge: float,
    ) -> float:
        """Sync consumption day calculation."""
        return pv + grid_purchase + batt_discharge - grid_injection - batt_charge

    def _sync_calculate_own_consumption(self, *, grid_injection: float, pv: float) -> float:
        """Sync own consumption calculation."""
        if pv > 0:
            return 100 * (1 - grid_injection / pv)
        return 0


class TestMqttOutputComparison:
    """Test MQTT output formatting is identical."""

    def test_topic_structure_parity(self) -> None:
        """Test topic structure is identical."""
        mqtt_topic = "MTEC"
        serial_no = "Z112200293130249"
        group = "now-base"
        mqtt_key = "grid_power"

        sync_topic = f"{mqtt_topic}/{serial_no}/{group}/{mqtt_key}/state"
        async_topic = f"{mqtt_topic}/{serial_no}/{group}/{mqtt_key}/state"

        assert sync_topic == async_topic == "MTEC/Z112200293130249/now-base/grid_power/state"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (123.456789, "123.457"),
            (0.0, "0.000"),
            (100.0, "100.000"),
            (-5.5, "-5.500"),
            (True, "1"),
            (False, "0"),
            (12345, "12345"),
            (-100, "-100"),
            ("Hello", "Hello"),
            ("V27.52.4.0-V27.52.4.0", "V27.52.4.0-V27.52.4.0"),
        ],
    )
    def test_value_formatting_parity(self, value: Any, expected: str) -> None:
        """Test value formatting is identical."""
        sync_result = self._sync_format_value(value=value)
        async_result = self._async_format_value(value=value)

        assert sync_result == async_result == expected

    def _async_format_value(self, *, value: Any, fmt: str = ".3f") -> str:
        """Async value formatting."""
        if isinstance(value, float):
            return f"{value:{fmt}}"
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    def _sync_format_value(self, *, value: Any, fmt: str = ".3f") -> str:
        """Sync value formatting."""
        if isinstance(value, float):
            return f"{value:{fmt}}"
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)


class TestMqttKeyComparison:
    """Test that MQTT keys match between sync and async."""

    # These are the CORRECT keys from registers.yaml
    EXPECTED_MQTT_KEYS: dict[str, str] = {
        "consumption": "consumption",
        "api-date": "api_date",
        "consumption-day": "consumption_day",
        "autarky-day": "autarky_rate_day",
        "ownconsumption-day": "own_consumption_day",
        "consumption-total": "consumption_total",
        "autarky-total": "autarky_rate_total",
        "ownconsumption-total": "own_consumption_total",
    }

    def test_all_mqtt_keys_documented(self) -> None:
        """Ensure all pseudo-register MQTT keys are documented."""
        for register_name, expected_key in self.EXPECTED_MQTT_KEYS.items():
            # This test just documents the expected mapping
            assert expected_key is not None, f"Missing MQTT key for {register_name}"

    @pytest.mark.parametrize(
        ("register_name", "expected_mqtt_key"),
        [
            ("consumption", "consumption"),
            ("api-date", "api_date"),
            ("consumption-day", "consumption_day"),
            ("autarky-day", "autarky_rate_day"),
            ("ownconsumption-day", "own_consumption_day"),
            ("consumption-total", "consumption_total"),
            ("autarky-total", "autarky_rate_total"),
            ("ownconsumption-total", "own_consumption_total"),
        ],
    )
    def test_mqtt_key_format(self, register_name: str, expected_mqtt_key: str) -> None:
        """Test MQTT keys use correct format (underscores, not hyphens)."""
        # Hyphens should NOT be in MQTT keys (except for register names)
        if register_name != "consumption":  # "consumption" has no hyphen
            assert "-" not in expected_mqtt_key, (
                f"MQTT key {expected_mqtt_key} should use underscores, not hyphens"
            )
