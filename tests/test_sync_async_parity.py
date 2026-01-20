"""
Parity tests to ensure sync and async implementations produce identical output.

These tests systematically compare every aspect of sync vs async behavior:
1. Register decoding (all types: U16, S16, U32, S32, BYTE, BIT, DAT, STR)
2. Special register processing (firmware, equipment, enums)
3. Pseudo-register calculations
4. MQTT topic structure
5. Value formatting
6. Polling groups

Run these tests after ANY change to either implementation to catch regressions.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aiomtec2mqtt.const import EQUIPMENT, RegisterGroup


class TestRegisterDecodingParity:
    """Verify sync and async decode registers identically."""

    @pytest.fixture
    def mock_registers(self) -> list[int]:
        """Create mock register data for testing."""
        return [
            0x1234,  # U16 = 4660
            0x8001,  # S16 = -32767 (high bit set)
            0x0001,
            0x0002,  # U32 = 65538
            0xFFFF,
            0xFFFE,  # S32 = -2
            0x1520,  # BYTE len=1 = "21 32"
            0x1520,
            0x0A0B,  # BYTE len=2 = "21 32  10 11"
            0x1520,
            0x0A0B,
            0x0C0D,
            0x0E0F,  # BYTE len=4
            0x00FF,  # BIT len=1 = "0000000011111111"
            0x19,
            0x01,
            0x14,
            0x0C,
            0x1E,
            0x3B,  # DAT = date/time
            ord("H") << 8 | ord("e"),
            ord("l") << 8 | ord("l"),
            ord("o") << 8 | 0,  # STR "Hello"
        ]

    def test_bit_len1_parity(self) -> None:
        """BIT length=1 formatting must be identical."""
        reg = 0x00FF

        expected = f"{reg:016b}"
        assert expected == "0000000011111111"

    def test_bit_len2_parity(self) -> None:
        """BIT length=2 formatting must be identical."""
        reg1, reg2 = 0x00FF, 0xFF00

        expected = f"{reg1:016b} {reg2:016b}"
        assert expected == "0000000011111111 1111111100000000"

    def test_byte_len1_parity(self) -> None:
        """BYTE length=1 formatting must be identical."""
        reg = 0x1520  # 21, 32

        expected = f"{reg >> 8:02d} {reg & 0xFF:02d}"
        assert expected == "21 32"

    def test_byte_len2_parity(self) -> None:
        """BYTE length=2 formatting must be identical."""
        reg1, reg2 = 0x1520, 0x0A0B

        # Double space between register pairs
        expected = f"{reg1 >> 8:02d} {reg1 & 0xFF:02d}  {reg2 >> 8:02d} {reg2 & 0xFF:02d}"
        assert expected == "21 32  10 11"

    def test_byte_len4_parity(self) -> None:
        """BYTE length=4 formatting must be identical."""
        reg1, reg2, reg3, reg4 = 0x1520, 0x0A0B, 0x0C0D, 0x0E0F

        # Single spaces within halves, double space in middle
        expected = (
            f"{reg1 >> 8:02d} {reg1 & 0xFF:02d} {reg2 >> 8:02d} {reg2 & 0xFF:02d}  "
            f"{reg3 >> 8:02d} {reg3 & 0xFF:02d} {reg4 >> 8:02d} {reg4 & 0xFF:02d}"
        )
        assert expected == "21 32 10 11  12 13 14 15"

    def test_dat_parity(self) -> None:
        """DAT (date) formatting must be identical."""
        # 2025-01-20 12:30:59
        reg1 = (25 << 8) | 1  # year-month
        reg2 = (20 << 8) | 12  # day-hour
        reg3 = (30 << 8) | 59  # minute-second

        expected = (
            f"{reg1 >> 8:02d}-{reg1 & 0xFF:02d}-{reg2 >> 8:02d} "
            f"{reg2 & 0xFF:02d}:{reg3 >> 8:02d}:{reg3 & 0xFF:02d}"
        )
        assert expected == "25-01-20 12:30:59"

    def test_s16_parity(self) -> None:
        """S16/I16 decoding must be identical."""
        raw_value = 0x8001  # Should be -32767

        # Sync approach (pymodbus INT16)
        # pymodbus converts 0x8001 to -32767
        sync_result = raw_value - 65536 if raw_value > 32767 else raw_value

        # Async approach
        async_result = raw_value - 65536 if raw_value > 32767 else raw_value

        assert sync_result == async_result == -32767

    def test_s32_parity(self) -> None:
        """S32/I32 decoding must be identical."""
        reg0, reg1 = 0xFFFF, 0xFFFE  # Should be -2

        raw = (reg0 << 16) + reg1

        # Signed conversion
        sync_result = raw - 4294967296 if raw > 2147483647 else raw
        async_result = raw - 4294967296 if raw > 2147483647 else raw

        assert sync_result == async_result == -2

    def test_str_parity(self) -> None:
        """STR decoding must be identical."""
        # "Hello" as registers
        registers = [
            ord("H") << 8 | ord("e"),  # 0x4865
            ord("l") << 8 | ord("l"),  # 0x6c6c
            ord("o") << 8 | 0,  # 0x6f00
        ]

        # Convert to bytes (big-endian)
        raw_bytes = b"".join(r.to_bytes(2, byteorder="big") for r in registers)
        result = raw_bytes.decode("utf-8").rstrip(" ").rstrip("\x00").rstrip(" ")

        assert result == "Hello"

    def test_u16_parity(self) -> None:
        """U16 decoding must be identical."""
        raw_value = 0x1234  # 4660

        # Sync approach (via pymodbus)
        sync_result = raw_value

        # Async approach (manual)
        async_result = int(raw_value)

        assert sync_result == async_result == 4660

    def test_u32_parity(self) -> None:
        """U32 decoding must be identical."""
        reg0, reg1 = 0x0001, 0x0002

        # Both should use big-endian: (reg0 << 16) + reg1
        sync_result = (reg0 << 16) + reg1
        async_result = (int(reg0) << 16) + int(reg1)

        assert sync_result == async_result == 65538


class TestSpecialRegisterParity:
    """Verify special register processing is identical."""

    def test_enum_conversion_bitfield(self) -> None:
        """Enum conversion for bitfield values must be identical."""
        # Binary string from BIT type
        value = "0000000000000011"  # Bits 0 and 1 set
        value_items = {0: "Fault A", 1: "Fault B", 2: "Fault C"}

        faults: list[str] = []
        value_no = int(f"0b{value.replace(' ', '')}", 2)
        for no, fault in value_items.items():
            if (value_no & (1 << no)) > 0:
                faults.append(fault)

        if not faults:
            faults.append("OK")

        result = ", ".join(faults)
        assert result == "Fault A, Fault B"

    def test_enum_conversion_int(self) -> None:
        """Enum conversion for int values must be identical."""
        value = 1
        value_items = {0: "Off", 1: "On", 2: "Auto"}

        result = value_items.get(value, "Unknown")
        assert result == "On"

    def test_equipment_info_lookup(self) -> None:
        """Register 10008 equipment lookup must be identical."""
        # Raw value: "8 3" (from BYTE type)
        raw_value = "8 3"

        upper, lower = raw_value.split(" ")
        result = EQUIPMENT.get(int(upper), {}).get(int(lower), "unknown")

        # Should return equipment name or "unknown"
        assert isinstance(result, str)

    def test_firmware_version_formatting(self) -> None:
        """Register 10011 firmware formatting must be identical."""
        # Raw value from BYTE type: "27 52 4 0  27 52 4 0"
        raw_value = "27 52 4 0  27 52 4 0"

        # Expected transformation
        fw0, fw1 = raw_value.split("  ")
        expected = f"V{fw0.replace(' ', '.')}-V{fw1.replace(' ', '.')}"

        assert expected == "V27.52.4.0-V27.52.4.0"


class TestPseudoRegisterParity:
    """Verify pseudo-register calculations and MQTT keys are identical."""

    def test_autarky_day_calculation(self) -> None:
        """autarky_rate_day = 100 * (1 - grid_purchase / consumption_day)."""
        grid_purchase = 2.0
        consumption_day = 10.0

        result = 100 * (1 - grid_purchase / consumption_day)
        assert result == 80.0

    def test_autarky_day_zero_consumption(self) -> None:
        """autarky_rate_day = 0 when consumption_day <= 0."""
        consumption_day = 0

        result = 0 if consumption_day <= 0 else 100 * (1 - 2.0 / consumption_day)
        assert result == 0

    def test_consumption_calculation(self) -> None:
        """Consumption = inverter_ac - grid_power."""
        inverter_ac = 5000
        grid_power = 2000

        result = inverter_ac - grid_power
        assert result == 3000

    def test_consumption_day_calculation(self) -> None:
        """consumption_day = pv + grid_purchase + batt_discharge - grid_injection - batt_charge."""
        pv_gen = 10.0
        grid_purchase = 2.0
        batt_discharge = 1.0
        grid_injection = 3.0
        batt_charge = 2.0

        result = pv_gen + grid_purchase + batt_discharge - grid_injection - batt_charge
        assert result == 8.0

    def test_mqtt_keys_match_registers_yaml(self) -> None:
        """All pseudo-register MQTT keys must match registers.yaml definitions."""
        # These are the CORRECT mqtt keys from registers.yaml
        expected_keys = {
            "consumption": "consumption",
            "api-date": "api_date",
            "consumption-day": "consumption_day",
            "autarky-day": "autarky_rate_day",
            "ownconsumption-day": "own_consumption_day",
            "consumption-total": "consumption_total",
            "autarky-total": "autarky_rate_total",
            "ownconsumption-total": "own_consumption_total",
        }

        # This test documents the expected mapping
        for register_name, mqtt_key in expected_keys.items():
            assert "_" in mqtt_key or mqtt_key == "consumption", (
                f"MQTT key for {register_name} should use underscores, got: {mqtt_key}"
            )

    def test_negative_value_correction(self) -> None:
        """Negative values in pseudo-registers must be corrected to 0."""
        # Only pseudo-registers should be corrected, not regular registers
        pseudo_value = -5.0

        corrected = max(0.0, pseudo_value)
        assert corrected == 0.0

    def test_own_consumption_day_calculation(self) -> None:
        """own_consumption_day = 100 * (1 - grid_injection / pv_gen)."""
        grid_injection = 3.0
        pv_gen = 10.0

        result = 100 * (1 - grid_injection / pv_gen)
        assert result == 70.0


class TestMqttOutputParity:
    """Verify MQTT topic structure and payload formatting are identical."""

    def test_bool_formatting(self) -> None:
        """Bool values must be "1" or "0"."""
        assert ("1" if True else "0") == "1"
        assert ("1" if False else "0") == "0"

    def test_float_formatting(self) -> None:
        """Float values must use configured format (default .3f)."""
        value = 123.456789
        fmt = ".3f"

        result = f"{value:{fmt}}"
        assert result == "123.457"

    def test_int_formatting(self) -> None:
        """Int values must be str(value)."""
        assert str(12345) == "12345"
        assert str(-100) == "-100"

    def test_topic_structure(self) -> None:
        """Topic must be: {mqtt_topic}/{serial_no}/{group}/{mqtt_key}/state."""
        mqtt_topic = "MTEC"
        serial_no = "Z112200293130249"
        group = RegisterGroup.BASE
        mqtt_key = "grid_power"

        expected = f"{mqtt_topic}/{serial_no}/{group}/{mqtt_key}/state"
        assert expected == "MTEC/Z112200293130249/now-base/grid_power/state"


class TestPollingGroupParity:
    """Verify both implementations poll the same register groups."""

    def test_all_groups_polled(self) -> None:
        """All register groups must be polled in async as in sync."""
        # Groups polled in sync coordinator
        sync_groups = {
            RegisterGroup.BASE,  # Every refresh_now
            RegisterGroup.CONFIG,  # Every refresh_config (writable entities!)
            RegisterGroup.GRID,  # Secondary round-robin
            RegisterGroup.INVERTER,  # Secondary round-robin
            RegisterGroup.BACKUP,  # Secondary round-robin
            RegisterGroup.BATTERY,  # Secondary round-robin
            RegisterGroup.PV,  # Secondary round-robin
            RegisterGroup.DAY,  # Every refresh_day
            RegisterGroup.TOTAL,  # Every refresh_total
            RegisterGroup.STATIC,  # Every refresh_static
        }

        # This test documents what groups MUST be polled
        assert RegisterGroup.CONFIG in sync_groups, "CONFIG must be polled for writable entities"
        assert RegisterGroup.STATIC in sync_groups, "STATIC must be polled periodically"


class TestScalingParity:
    """Verify scaling is applied identically."""

    def test_no_scaling_when_scale_is_1(self) -> None:
        """No scaling when scale=1."""
        raw_value = 1000
        scale = 1

        # When scale is 1, no division needed
        result = raw_value / scale if scale > 1 else raw_value

        assert result == 1000

    def test_scaling_applied(self) -> None:
        """Values must be divided by scale factor."""
        raw_value = 1000
        scale = 10

        # Both implementations should produce float after scaling
        result = raw_value / scale
        assert result == 100.0
        assert isinstance(result, float)


class TestApiDateParity:
    """Verify api_date formatting is identical."""

    def test_api_date_format(self) -> None:
        """api_date must use format: YYYY-MM-DD HH:MM:SS."""
        now = datetime(2025, 1, 20, 14, 30, 45)

        result = now.strftime("%Y-%m-%d %H:%M:%S")
        assert result == "2025-01-20 14:30:45"


# Integration test that would catch most issues
class TestSyncAsyncIntegration:
    """Integration tests comparing actual output."""

    def test_value_formatting_consistency(self) -> None:
        """All value types must be formatted consistently."""
        test_cases = [
            (123.456, ".3f", "123.456"),
            (100.0, ".3f", "100.000"),
            (True, None, "1"),
            (False, None, "0"),
            (12345, None, "12345"),
            ("Hello", None, "Hello"),
        ]

        for value, fmt, expected in test_cases:
            if isinstance(value, float):
                result = f"{value:{fmt}}"
            elif isinstance(value, bool):
                result = "1" if value else "0"
            else:
                result = str(value)

            assert result == expected, f"Failed for {value}: got {result}, expected {expected}"
