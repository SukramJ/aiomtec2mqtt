"""Unit tests for :class:`AsyncMtecCoordinator` helper methods.

These tests exercise pure-ish coordinator logic (value conversion, formatting,
register-value processing, pseudo-register computation) without spinning up
Modbus or MQTT connections. Dependencies are patched at import time.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiomtec2mqtt.const import Config, Register, RegisterGroup


def _base_config() -> dict[Config, Any]:
    return {
        Config.MODBUS_IP: "192.168.1.100",
        Config.MODBUS_PORT: 502,
        Config.MODBUS_SLAVE: 1,
        Config.MODBUS_TIMEOUT: 5,
        Config.MQTT_SERVER: "mqtt.example.com",
        Config.MQTT_PORT: 1883,
        Config.MQTT_LOGIN: "u",
        Config.MQTT_PASSWORD: "p",
        Config.MQTT_TOPIC: "MTEC",
        Config.MQTT_FLOAT_FORMAT: ".3f",
        Config.HASS_BASE_TOPIC: "homeassistant",
        Config.HASS_ENABLE: False,
        Config.HASS_BIRTH_GRACETIME: 0,
        Config.DEBUG: False,
        Config.REFRESH_NOW: 1,
        Config.REFRESH_DAY: 10,
        Config.REFRESH_STATIC: 3600,
        Config.REFRESH_CONFIG: 60,
        Config.REFRESH_TOTAL: 300,
    }


def _register_map() -> dict[str, dict[str, Any]]:
    return {
        "10008": {
            Register.NAME: "Equipment info",
            Register.MQTT: "equipment_info",
            Register.GROUP: RegisterGroup.STATIC,
        },
        "10011": {
            Register.NAME: "Firmware version",
            Register.MQTT: "firmware_version",
            Register.GROUP: RegisterGroup.STATIC,
        },
        "10100": {
            Register.NAME: "Inverter serial number",
            Register.MQTT: "serial_no",
            Register.GROUP: RegisterGroup.STATIC,
        },
        "11000": {
            Register.NAME: "Grid power",
            Register.MQTT: "grid_power",
            Register.GROUP: RegisterGroup.BASE,
        },
        "11016": {
            Register.NAME: "Inverter AC power",
            Register.MQTT: "inverter_ac_power",
            Register.GROUP: RegisterGroup.BASE,
        },
        "12000": {
            Register.NAME: "Inverter status",
            Register.MQTT: "status",
            Register.GROUP: RegisterGroup.BASE,
            Register.DEVICE_CLASS: "enum",
            Register.VALUE_ITEMS: {0: "Idle", 1: "Running", 2: "Fault"},
        },
    }


@pytest.fixture
def coordinator():
    """Provide a coordinator with mocked modbus + mqtt clients."""
    from aiomtec2mqtt.async_coordinator import AsyncMtecCoordinator

    with (
        patch("aiomtec2mqtt.async_coordinator.init_config", return_value=_base_config()),
        patch(
            "aiomtec2mqtt.async_coordinator.init_register_map",
            return_value=(_register_map(), [RegisterGroup.BASE, RegisterGroup.STATIC]),
        ),
    ):
        coord = AsyncMtecCoordinator()
        coord._topic_base = "MTEC/SN123"
        coord._serial_no = "SN123"
        coord._mqtt_client = MagicMock()
        coord._mqtt_client.publish = AsyncMock()
        coord._mqtt_client.subscribe = AsyncMock()
        coord._mqtt_client.is_connected = True
        coord._mqtt_client.reconnect = AsyncMock(return_value=True)
        coord._modbus_client = MagicMock()
        coord._modbus_client.read_register_group = AsyncMock(return_value={})
        coord._modbus_client.error_count = 0
        return coord


class TestConvertCode:
    def test_binary_string_multiple_faults(self, coordinator) -> None:
        # "101" → bits 0 and 2 set → faults at positions 0 and 2
        result = coordinator._convert_code(value="101", value_items={0: "A", 1: "B", 2: "C"})
        assert "A" in result and "C" in result and "B" not in result

    def test_binary_string_no_faults_returns_ok(self, coordinator) -> None:
        result = coordinator._convert_code(value="000", value_items={0: "A", 1: "B"})
        assert result == "OK"

    def test_binary_string_with_spaces(self, coordinator) -> None:
        result = coordinator._convert_code(value="1 0 1", value_items={0: "A", 2: "C"})
        assert "A" in result and "C" in result

    def test_integer_known(self, coordinator) -> None:
        result = coordinator._convert_code(value=1, value_items={0: "Off", 1: "On"})
        assert result == "On"

    def test_integer_unknown_returns_unknown(self, coordinator) -> None:
        result = coordinator._convert_code(value=99, value_items={0: "Off"})
        assert result == "Unknown"

    def test_invalid_string_returns_ok(self, coordinator) -> None:
        result = coordinator._convert_code(value="not-binary", value_items={0: "A"})
        assert result == "OK"


class TestFormatValue:
    def test_bool_false_maps_to_zero(self, coordinator) -> None:
        assert coordinator._format_value(value=False) == "0"

    def test_bool_true_maps_to_one(self, coordinator) -> None:
        assert coordinator._format_value(value=True) == "1"

    def test_float_applies_configured_format(self, coordinator) -> None:
        assert coordinator._format_value(value=3.14159) == "3.142"

    def test_int_passes_through_as_str(self, coordinator) -> None:
        assert coordinator._format_value(value=42) == "42"

    def test_string_passes_through(self, coordinator) -> None:
        assert coordinator._format_value(value="hello") == "hello"


class TestProcessRegisterValue:
    def test_enum_conversion_invoked(self, coordinator) -> None:
        reg_info = {
            Register.DEVICE_CLASS: "enum",
            Register.VALUE_ITEMS: {0: "Idle", 1: "Running"},
        }
        result = coordinator._process_register_value(
            register_addr="12000", value=1, reg_info=reg_info
        )
        assert result == "Running"

    def test_equipment_lookup(self, coordinator) -> None:
        result = coordinator._process_register_value(
            register_addr="10008", value="30 3", reg_info={}
        )
        assert result == "8.0K-25A-3P"

    def test_equipment_non_numeric(self, coordinator) -> None:
        result = coordinator._process_register_value(
            register_addr="10008", value="abc def", reg_info={}
        )
        assert result == "abc def"

    def test_equipment_unknown_code(self, coordinator) -> None:
        result = coordinator._process_register_value(
            register_addr="10008", value="99 99", reg_info={}
        )
        assert result == "unknown"

    def test_firmware_formatting(self, coordinator) -> None:
        result = coordinator._process_register_value(
            register_addr="10011",
            value="1 2 3  4 5 6",
            reg_info={},
        )
        assert result == "V1.2.3-V4.5.6"

    def test_firmware_malformed_returns_raw(self, coordinator) -> None:
        # no double-space → no split → original returned
        result = coordinator._process_register_value(
            register_addr="10011", value="nodouble", reg_info={}
        )
        assert result == "nodouble"

    def test_passthrough_when_no_special_case(self, coordinator) -> None:
        result = coordinator._process_register_value(register_addr="99999", value=42, reg_info={})
        assert result == 42


class TestShutdown:
    def test_shutdown_sets_event(self, coordinator) -> None:
        assert not coordinator._shutdown_event.is_set()
        coordinator.shutdown()
        assert coordinator._shutdown_event.is_set()


class TestPublishRegisterData:
    @pytest.mark.asyncio
    async def test_publish_register_unknown_name_skipped(self, coordinator) -> None:
        await coordinator._publish_register_data(
            data={"Unregistered": 1}, group=RegisterGroup.BASE
        )
        # Only pseudo-registers get published (consumption + api_date)
        topics = [c.kwargs["topic"] for c in coordinator._mqtt_client.publish.call_args_list]
        for t in topics:
            assert "Unregistered" not in t

    @pytest.mark.asyncio
    async def test_publish_registers_emits_state_topics(self, coordinator) -> None:
        data = {"Grid power": 150, "Inverter AC power": 400}
        await coordinator._publish_register_data(data=data, group=RegisterGroup.BASE)
        published = {
            call.kwargs["topic"] for call in coordinator._mqtt_client.publish.call_args_list
        }
        assert "MTEC/SN123/now-base/grid_power/state" in published
        assert "MTEC/SN123/now-base/inverter_ac_power/state" in published

    @pytest.mark.asyncio
    async def test_publish_skips_without_topic_base(self, coordinator) -> None:
        coordinator._topic_base = None
        await coordinator._publish_register_data(data={"Grid power": 1}, group=RegisterGroup.BASE)
        coordinator._mqtt_client.publish.assert_not_called()


class TestPublishPseudoRegisters:
    @pytest.mark.asyncio
    async def test_base_consumption_and_date(self, coordinator) -> None:
        raw = {"Inverter AC power": 500, "Grid power": 200}
        await coordinator._publish_pseudo_registers(
            processed_data={}, raw_data=raw, group=RegisterGroup.BASE
        )
        calls = coordinator._mqtt_client.publish.call_args_list
        topics = {c.kwargs["topic"]: c.kwargs["payload"] for c in calls}
        assert topics["MTEC/SN123/now-base/consumption/state"] == "300.000"
        assert "MTEC/SN123/now-base/api_date/state" in topics

    @pytest.mark.asyncio
    async def test_base_consumption_clamped_to_zero(self, coordinator) -> None:
        raw = {"Inverter AC power": 100, "Grid power": 500}
        await coordinator._publish_pseudo_registers(
            processed_data={}, raw_data=raw, group=RegisterGroup.BASE
        )
        calls = coordinator._mqtt_client.publish.call_args_list
        topics = {c.kwargs["topic"]: c.kwargs["payload"] for c in calls}
        assert topics["MTEC/SN123/now-base/consumption/state"] == "0.000"

    @pytest.mark.asyncio
    async def test_day_autarky_calculation(self, coordinator) -> None:
        raw = {
            "PV energy generated (day)": 10.0,
            "Grid purchased energy (day)": 2.0,
            "Battery discharge energy (day)": 1.0,
            "Grid injection energy (day)": 3.0,
            "Battery charge energy (day)": 0.0,
        }
        await coordinator._publish_pseudo_registers(
            processed_data={}, raw_data=raw, group=RegisterGroup.DAY
        )
        calls = coordinator._mqtt_client.publish.call_args_list
        topics = {c.kwargs["topic"]: c.kwargs["payload"] for c in calls}
        # consumption_day = 10 + 2 + 1 - 3 - 0 = 10
        assert topics["MTEC/SN123/day/consumption_day/state"] == "10.000"
        # autarky = 100 * (1 - 2/10) = 80
        assert topics["MTEC/SN123/day/autarky_rate_day/state"] == "80.000"
        # own_consumption = 100 * (1 - 3/10) = 70
        assert topics["MTEC/SN123/day/own_consumption_day/state"] == "70.000"

    @pytest.mark.asyncio
    async def test_day_zero_consumption_yields_zero_autarky(self, coordinator) -> None:
        raw = {
            "PV energy generated (day)": 0.0,
            "Grid purchased energy (day)": 0.0,
            "Battery discharge energy (day)": 0.0,
            "Grid injection energy (day)": 0.0,
            "Battery charge energy (day)": 0.0,
        }
        await coordinator._publish_pseudo_registers(
            processed_data={}, raw_data=raw, group=RegisterGroup.DAY
        )
        calls = coordinator._mqtt_client.publish.call_args_list
        topics = {c.kwargs["topic"]: c.kwargs["payload"] for c in calls}
        assert topics["MTEC/SN123/day/autarky_rate_day/state"] == "0"
        assert topics["MTEC/SN123/day/own_consumption_day/state"] == "0"

    @pytest.mark.asyncio
    async def test_skips_without_topic_base(self, coordinator) -> None:
        coordinator._topic_base = None
        await coordinator._publish_pseudo_registers(
            processed_data={}, raw_data={}, group=RegisterGroup.BASE
        )
        coordinator._mqtt_client.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_total_pseudo_registers(self, coordinator) -> None:
        raw = {
            "PV energy generated (total)": 1000.0,
            "Grid energy purchased (total)": 200.0,
            "Battery energy discharged (total)": 100.0,
            "Grid energy injected (total)": 300.0,
            "Battery energy charged (total)": 50.0,
        }
        await coordinator._publish_pseudo_registers(
            processed_data={}, raw_data=raw, group=RegisterGroup.TOTAL
        )
        calls = coordinator._mqtt_client.publish.call_args_list
        topics = {c.kwargs["topic"]: c.kwargs["payload"] for c in calls}
        # consumption_total = 1000 + 200 + 100 - 300 - 50 = 950
        assert topics["MTEC/SN123/total/consumption_total/state"] == "950.000"


class TestOnMqttMessage:
    def test_hass_birth_triggers_rediscovery(self, coordinator) -> None:
        msg = MagicMock()
        msg.topic = coordinator._hass_status_topic
        msg.payload = b"online"
        coordinator._hass_discovery_sent = True
        coordinator._on_mqtt_message(msg)
        assert coordinator._hass_discovery_sent is False

    def test_unrelated_topic_ignored(self, coordinator) -> None:
        msg = MagicMock()
        msg.topic = "other/unrelated/topic"
        msg.payload = b"x"
        coordinator._on_mqtt_message(msg)
        assert coordinator._pending_write_queue.empty()

    def test_write_command_enqueued(self, coordinator) -> None:
        msg = MagicMock()
        msg.topic = f"{coordinator._topic_base}/config/setting/set"
        msg.payload = b"42"
        coordinator._on_mqtt_message(msg)
        key, val = coordinator._pending_write_queue.get_nowait()
        assert key == "setting"
        assert val == "42"


class TestInitializeFromStaticData:
    @pytest.mark.asyncio
    async def test_already_initialized_returns_true(self, coordinator) -> None:
        coordinator._serial_no = "ALREADY"
        assert await coordinator._initialize_from_static_data() is True

    @pytest.mark.asyncio
    async def test_empty_data_returns_false(self, coordinator) -> None:
        coordinator._serial_no = None
        coordinator._modbus_client.read_register_group = AsyncMock(return_value={})
        assert await coordinator._initialize_from_static_data() is False

    @pytest.mark.asyncio
    async def test_exception_returns_false(self, coordinator) -> None:
        coordinator._serial_no = None
        coordinator._modbus_client.read_register_group = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        assert await coordinator._initialize_from_static_data() is False

    @pytest.mark.asyncio
    async def test_happy_path_sets_state(self, coordinator) -> None:
        coordinator._serial_no = None
        coordinator._topic_base = None
        coordinator._modbus_client.read_register_group = AsyncMock(
            return_value={
                "Inverter serial number": "SN999",
                "Firmware version": "V1.2",
                "Equipment info": "8.0K-25A-3P",
            }
        )
        assert await coordinator._initialize_from_static_data() is True
        assert coordinator._serial_no == "SN999"
        assert coordinator._topic_base == "MTEC/SN999"

    @pytest.mark.asyncio
    async def test_missing_serial_returns_false(self, coordinator) -> None:
        coordinator._serial_no = None
        coordinator._modbus_client.read_register_group = AsyncMock(
            return_value={"Firmware version": "v1"}
        )
        assert await coordinator._initialize_from_static_data() is False


class TestMqttWatchdog:
    @pytest.mark.asyncio
    async def test_cancels_cleanly(self, coordinator) -> None:
        task = asyncio.create_task(coordinator._mqtt_watchdog())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_reconnects_when_disconnected(self, coordinator) -> None:
        coordinator._mqtt_client.is_connected = False
        coordinator._mqtt_client.reconnect = AsyncMock(return_value=True)

        async def _signal_shutdown() -> None:
            await asyncio.sleep(0.05)
            coordinator._shutdown_event.set()

        await asyncio.gather(coordinator._mqtt_watchdog(), _signal_shutdown())
        coordinator._mqtt_client.reconnect.assert_awaited()


class TestModbusWatchdog:
    @pytest.mark.asyncio
    async def test_reconnects_when_errors_exceed_threshold(self, coordinator) -> None:
        coordinator._modbus_client.error_count = 50
        coordinator._modbus_client.disconnect = AsyncMock()
        coordinator._modbus_client.connect = AsyncMock(return_value=True)

        async def fake_sleep(delay):  # noqa: ARG001
            coordinator._shutdown_event.set()

        with patch("aiomtec2mqtt.async_coordinator.asyncio.sleep", new=fake_sleep):
            await coordinator._modbus_watchdog()
        coordinator._modbus_client.disconnect.assert_awaited()
