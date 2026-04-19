"""Extended tests for :class:`AsyncMtecCoordinator`.

Complements :mod:`tests.test_async_coordinator_unit` by exercising the
async loops (health check, watchdogs, write queue, polling), HASS
discovery / birth flows, reconnection, and construction-time wiring.

All external dependencies are stubbed; ``asyncio.sleep`` is patched out
so the loops complete in microseconds.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiomtec2mqtt.const import Config, Register, RegisterGroup
from aiomtec2mqtt.health import ComponentHealth, HealthStatus, SystemHealth


def _base_config(*, hass: bool = False) -> dict[Config, Any]:
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
        Config.MQTT_FLOAT_FORMAT: "{:.3f}",  # braces deliberately included
        Config.HASS_BASE_TOPIC: "homeassistant",
        Config.HASS_ENABLE: hass,
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
    }


def _stop_after_n_sleeps(coordinator, n: int = 1):
    """Build a fake ``asyncio.sleep`` that flips shutdown after N calls.

    Avoids needing ``asyncio.gather`` with a watchdog task — keeping the
    loop body single-threaded prevents accidental infinite loops when the
    real ``asyncio.sleep`` is patched out and never yields.
    """
    counter = {"n": 0}

    async def fake_sleep(delay):  # noqa: ARG001
        counter["n"] += 1
        if counter["n"] >= n:
            coordinator._shutdown_event.set()

    return fake_sleep


def _build_coordinator(*, hass: bool = False):
    """Return a coordinator with all I/O components mocked."""
    from aiomtec2mqtt.async_coordinator import AsyncMtecCoordinator

    with (
        patch("aiomtec2mqtt.async_coordinator.init_config", return_value=_base_config(hass=hass)),
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
    coord._modbus_client.write_register_by_name = AsyncMock(return_value=True)
    coord._modbus_client.connect = AsyncMock(return_value=True)
    coord._modbus_client.disconnect = AsyncMock()
    coord._modbus_client.error_count = 0
    return coord


@pytest.fixture
def coordinator():
    return _build_coordinator()


@pytest.fixture
def coordinator_with_hass():
    coord = _build_coordinator(hass=True)
    coord._hass = MagicMock()
    coord._hass.is_initialized = False
    coord._hass.initialize = MagicMock()
    coord._hass._devices_array = [
        ("homeassistant/sensor/x/config", '{"a":1}', None),
        ("homeassistant/number/y/config", '{"b":2}', "MTEC/SN123/y/set"),
    ]
    return coord


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_float_format_braces_stripped(self, coordinator) -> None:
        # "{:.3f}" should normalize to ".3f"
        assert coordinator._mqtt_float_format == ".3f"

    def test_hass_disabled_means_hass_attr_none(self, coordinator) -> None:
        assert coordinator._hass is None

    def test_hass_enabled_creates_integration(self) -> None:
        from aiomtec2mqtt.async_coordinator import AsyncMtecCoordinator

        with (
            patch(
                "aiomtec2mqtt.async_coordinator.init_config", return_value=_base_config(hass=True)
            ),
            patch(
                "aiomtec2mqtt.async_coordinator.init_register_map",
                return_value=(_register_map(), [RegisterGroup.BASE]),
            ),
        ):
            coord = AsyncMtecCoordinator()
        assert coord._hass is not None

    def test_registers_by_group_index_built(self, coordinator) -> None:
        # Both groups configured in the test fixture.
        assert RegisterGroup.BASE in coordinator._registers_by_group
        assert RegisterGroup.STATIC in coordinator._registers_by_group
        assert "11000" in coordinator._registers_by_group[RegisterGroup.BASE]

    def test_topic_base_starts_unset(self) -> None:
        # Rebuild raw and check the *raw* default (no fixture setup).
        from aiomtec2mqtt.async_coordinator import AsyncMtecCoordinator

        with (
            patch("aiomtec2mqtt.async_coordinator.init_config", return_value=_base_config()),
            patch(
                "aiomtec2mqtt.async_coordinator.init_register_map",
                return_value=(_register_map(), [RegisterGroup.BASE]),
            ),
        ):
            raw = AsyncMtecCoordinator()
        assert raw._topic_base is None
        assert raw._serial_no is None


# ---------------------------------------------------------------------------
# Health check loop
# ---------------------------------------------------------------------------


class TestHealthCheckLoop:
    @pytest.mark.asyncio
    async def test_cancels_cleanly(self, coordinator) -> None:
        task = asyncio.create_task(coordinator._health_check_loop())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_logs_healthy_on_healthy_system(self, coordinator) -> None:
        healthy = SystemHealth(
            status=HealthStatus.HEALTHY,
            components={},
            check_time=datetime.now(UTC),
            message="ok",
        )
        coordinator._health_check.check_health = MagicMock(return_value=healthy)

        with patch(
            "aiomtec2mqtt.async_coordinator.asyncio.sleep",
            new=_stop_after_n_sleeps(coordinator),
        ):
            await coordinator._health_check_loop()
        coordinator._health_check.check_health.assert_called()

    @pytest.mark.asyncio
    async def test_logs_unhealthy_components(self, coordinator) -> None:
        bad = ComponentHealth(name="modbus", status=HealthStatus.UNHEALTHY, error_count=7)
        unhealthy = SystemHealth(
            status=HealthStatus.UNHEALTHY,
            components={"modbus": bad},
            check_time=datetime.now(UTC),
            message="modbus down",
        )
        coordinator._health_check.check_health = MagicMock(return_value=unhealthy)

        with patch(
            "aiomtec2mqtt.async_coordinator.asyncio.sleep",
            new=_stop_after_n_sleeps(coordinator),
        ):
            await coordinator._health_check_loop()
        coordinator._health_check.check_health.assert_called()

    @pytest.mark.asyncio
    async def test_swallows_exception_and_continues(self, coordinator) -> None:
        coordinator._health_check.check_health = MagicMock(side_effect=RuntimeError("boom"))

        with patch(
            "aiomtec2mqtt.async_coordinator.asyncio.sleep",
            new=_stop_after_n_sleeps(coordinator),
        ):
            await coordinator._health_check_loop()  # Must not raise


# ---------------------------------------------------------------------------
# Reconnect modbus
# ---------------------------------------------------------------------------


class TestReconnectModbus:
    @pytest.mark.asyncio
    async def test_reconnect_failure_logged(self, coordinator) -> None:
        coordinator._modbus_client.connect = AsyncMock(return_value=False)
        with patch("aiomtec2mqtt.async_coordinator.asyncio.sleep", new=AsyncMock()):
            await coordinator._reconnect_modbus()  # No exception
        coordinator._modbus_client.connect.assert_awaited()

    @pytest.mark.asyncio
    async def test_reconnect_success(self, coordinator) -> None:
        coordinator._modbus_client.connect = AsyncMock(return_value=True)
        with patch("aiomtec2mqtt.async_coordinator.asyncio.sleep", new=AsyncMock()):
            await coordinator._reconnect_modbus()
        coordinator._modbus_client.disconnect.assert_awaited()
        coordinator._modbus_client.connect.assert_awaited()

    @pytest.mark.asyncio
    async def test_reconnect_swallows_exception(self, coordinator) -> None:
        coordinator._modbus_client.disconnect = AsyncMock(side_effect=RuntimeError("nope"))
        with patch("aiomtec2mqtt.async_coordinator.asyncio.sleep", new=AsyncMock()):
            await coordinator._reconnect_modbus()  # No exception


# ---------------------------------------------------------------------------
# HASS discovery + birth
# ---------------------------------------------------------------------------


class TestSendHassDiscovery:
    @pytest.mark.asyncio
    async def test_init_failure_aborts(self, coordinator_with_hass) -> None:
        coordinator_with_hass._serial_no = None
        coordinator_with_hass._modbus_client.read_register_group = AsyncMock(return_value={})
        await coordinator_with_hass._send_hass_discovery()
        coordinator_with_hass._mqtt_client.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_hass_is_noop(self, coordinator) -> None:
        coordinator._hass = None
        await coordinator._send_hass_discovery()
        coordinator._mqtt_client.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_exception_logged_no_raise(self, coordinator_with_hass) -> None:
        coordinator_with_hass._mqtt_client.publish = AsyncMock(side_effect=RuntimeError("net"))
        await coordinator_with_hass._send_hass_discovery()
        # Did not raise; flag remains False
        assert coordinator_with_hass._hass_discovery_sent is False

    @pytest.mark.asyncio
    async def test_publishes_devices_and_subscribes_command_topics(
        self, coordinator_with_hass
    ) -> None:
        await coordinator_with_hass._send_hass_discovery()
        # Two device entries → two publish calls
        assert coordinator_with_hass._mqtt_client.publish.await_count == 2
        # One had a command topic → one subscribe
        coordinator_with_hass._mqtt_client.subscribe.assert_awaited_once()
        assert coordinator_with_hass._hass_discovery_sent is True

    @pytest.mark.asyncio
    async def test_subscribe_failure_does_not_abort(self, coordinator_with_hass) -> None:
        coordinator_with_hass._mqtt_client.subscribe = AsyncMock(side_effect=RuntimeError("nope"))
        await coordinator_with_hass._send_hass_discovery()
        # All publishes still happened, flag still set
        assert coordinator_with_hass._mqtt_client.publish.await_count == 2
        assert coordinator_with_hass._hass_discovery_sent is True


class TestWaitForHassBirth:
    @pytest.mark.asyncio
    async def test_no_hass_is_noop(self, coordinator) -> None:
        coordinator._hass = None
        await coordinator._wait_for_hass_birth()
        coordinator._mqtt_client.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_discovery_after_gracetime(self, coordinator_with_hass) -> None:
        with patch("aiomtec2mqtt.async_coordinator.asyncio.sleep", new=AsyncMock()):
            await coordinator_with_hass._wait_for_hass_birth()
        coordinator_with_hass._mqtt_client.subscribe.assert_any_await(
            topic=coordinator_with_hass._hass_status_topic
        )
        # Discovery publish triggered for both registered devices
        assert coordinator_with_hass._mqtt_client.publish.await_count >= 2

    @pytest.mark.asyncio
    async def test_skips_discovery_if_already_sent(self, coordinator_with_hass) -> None:
        coordinator_with_hass._hass_discovery_sent = True
        with patch("aiomtec2mqtt.async_coordinator.asyncio.sleep", new=AsyncMock()):
            await coordinator_with_hass._wait_for_hass_birth()
        # No discovery publishes
        coordinator_with_hass._mqtt_client.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_subscribe_failure_does_not_raise(self, coordinator_with_hass) -> None:
        coordinator_with_hass._mqtt_client.subscribe = AsyncMock(side_effect=RuntimeError("x"))
        with patch("aiomtec2mqtt.async_coordinator.asyncio.sleep", new=AsyncMock()):
            # Will still attempt discovery
            await coordinator_with_hass._wait_for_hass_birth()


# ---------------------------------------------------------------------------
# Process write queue
# ---------------------------------------------------------------------------


class TestProcessWriteQueue:
    @pytest.mark.asyncio
    async def test_cancels_cleanly(self, coordinator) -> None:
        task = asyncio.create_task(coordinator._process_write_queue())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_processes_queued_write_failure(self, coordinator) -> None:
        coordinator._modbus_client.write_register_by_name = AsyncMock(return_value=False)
        await coordinator._pending_write_queue.put(("setting", "0"))

        async def stop_after_processed() -> None:
            await coordinator._pending_write_queue.join()
            coordinator._shutdown_event.set()

        await asyncio.gather(coordinator._process_write_queue(), stop_after_processed())
        coordinator._modbus_client.write_register_by_name.assert_awaited()

    @pytest.mark.asyncio
    async def test_processes_queued_write_success(self, coordinator) -> None:
        coordinator._modbus_client.write_register_by_name = AsyncMock(return_value=True)
        await coordinator._pending_write_queue.put(("setting", "42"))

        async def stop_after_processed() -> None:
            await coordinator._pending_write_queue.join()
            coordinator._shutdown_event.set()

        await asyncio.gather(coordinator._process_write_queue(), stop_after_processed())
        coordinator._modbus_client.write_register_by_name.assert_awaited_with(
            name="setting", value="42"
        )

    @pytest.mark.asyncio
    async def test_swallows_write_exception(self, coordinator) -> None:
        coordinator._modbus_client.write_register_by_name = AsyncMock(
            side_effect=RuntimeError("x")
        )
        await coordinator._pending_write_queue.put(("k", "v"))

        async def stop_after_processed() -> None:
            await coordinator._pending_write_queue.join()
            coordinator._shutdown_event.set()

        await asyncio.gather(coordinator._process_write_queue(), stop_after_processed())
        coordinator._modbus_client.write_register_by_name.assert_awaited()


# ---------------------------------------------------------------------------
# on_mqtt_message edge cases
# ---------------------------------------------------------------------------


class TestOnMqttMessageEdges:
    def test_empty_payload_is_handled(self, coordinator) -> None:
        msg = MagicMock()
        msg.topic = "other/topic"
        msg.payload = None
        coordinator._on_mqtt_message(msg)  # No exception
        assert coordinator._pending_write_queue.empty()

    def test_queue_full_does_not_raise(self, coordinator) -> None:
        coordinator._pending_write_queue = asyncio.Queue(maxsize=1)
        coordinator._pending_write_queue.put_nowait(("first", "1"))
        msg = MagicMock()
        msg.topic = f"{coordinator._topic_base}/grp/second/set"
        msg.payload = b"2"
        # Should log but not raise.
        coordinator._on_mqtt_message(msg)

    def test_too_short_topic_ignored(self, coordinator) -> None:
        # Setup short base so a 3-segment path can still match prefix.
        coordinator._topic_base = "X"
        msg = MagicMock()
        msg.topic = "X/key/set"  # only 3 parts → < 4 → skipped
        msg.payload = b"v"
        coordinator._on_mqtt_message(msg)
        assert coordinator._pending_write_queue.empty()

    def test_topic_outside_topic_base_ignored(self, coordinator) -> None:
        msg = MagicMock()
        msg.topic = "different-prefix/x/set"
        msg.payload = b"1"
        coordinator._on_mqtt_message(msg)
        assert coordinator._pending_write_queue.empty()

    def test_topic_under_base_without_set_ignored(self, coordinator) -> None:
        msg = MagicMock()
        msg.topic = f"{coordinator._topic_base}/group/key/state"
        msg.payload = b"1"
        coordinator._on_mqtt_message(msg)
        assert coordinator._pending_write_queue.empty()


# ---------------------------------------------------------------------------
# Polling tasks
# ---------------------------------------------------------------------------


class TestPollingTasks:
    @pytest.mark.asyncio
    async def test_base_polling_cancels_cleanly(self, coordinator) -> None:
        # Pre-set serial so init loop returns immediately.
        coordinator._serial_no = "SN"
        task = asyncio.create_task(coordinator._poll_base_registers())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_base_polling_swallows_read_exception(self, coordinator) -> None:
        coordinator._serial_no = "SN"
        coordinator._modbus_client.read_register_group = AsyncMock(side_effect=RuntimeError("x"))
        with patch(
            "aiomtec2mqtt.async_coordinator.asyncio.sleep",
            new=_stop_after_n_sleeps(coordinator),
        ):
            await coordinator._poll_base_registers()  # Must not raise

    @pytest.mark.asyncio
    async def test_config_polling_cancels_cleanly(self, coordinator) -> None:
        task = asyncio.create_task(coordinator._poll_config_registers())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_day_polling_cancels_cleanly(self, coordinator) -> None:
        task = asyncio.create_task(coordinator._poll_day_statistics())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_secondary_polling_cancels_cleanly(self, coordinator) -> None:
        task = asyncio.create_task(coordinator._poll_secondary_registers())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_secondary_round_robins(self, coordinator) -> None:
        # Force enough cycles to wrap once.
        from aiomtec2mqtt.const import SECONDARY_REGISTER_GROUPS

        groups_seen: list[Any] = []
        target = len(SECONDARY_REGISTER_GROUPS) + 1

        async def fake_read(*, group_name) -> dict[str, Any]:
            groups_seen.append(group_name)
            if len(groups_seen) >= target:
                coordinator._shutdown_event.set()
            return {}

        coordinator._modbus_client.read_register_group = AsyncMock(side_effect=fake_read)

        # Pre-set topic_base so the secondary loop skips its init sleep.
        async def noop_sleep(delay):  # noqa: ARG001
            return None

        with patch("aiomtec2mqtt.async_coordinator.asyncio.sleep", new=noop_sleep):
            await coordinator._poll_secondary_registers()
        # First entry must be GRID (index 0); after wrap we land on GRID again.
        assert groups_seen[0] == SECONDARY_REGISTER_GROUPS[0]
        assert groups_seen[-1] == SECONDARY_REGISTER_GROUPS[0]

    @pytest.mark.asyncio
    async def test_static_polling_cancels_cleanly(self, coordinator) -> None:
        task = asyncio.create_task(coordinator._poll_static_registers())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_total_polling_cancels_cleanly(self, coordinator) -> None:
        task = asyncio.create_task(coordinator._poll_total_statistics())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# initialize_from_static_data — HASS branch
# ---------------------------------------------------------------------------


class TestInitializeWithHass:
    @pytest.mark.asyncio
    async def test_initializes_hass_when_present(self, coordinator_with_hass) -> None:
        coordinator_with_hass._serial_no = None
        coordinator_with_hass._topic_base = None
        coordinator_with_hass._modbus_client.read_register_group = AsyncMock(
            return_value={
                "Inverter serial number": "SN-X",
                "Firmware version": "v1",
                "Equipment info": "info",
            }
        )
        ok = await coordinator_with_hass._initialize_from_static_data()
        assert ok is True
        coordinator_with_hass._hass.initialize.assert_called_once()
        kwargs = coordinator_with_hass._hass.initialize.call_args.kwargs
        assert kwargs["serial_no"] == "SN-X"
        assert kwargs["firmware_version"] == "v1"
        assert kwargs["equipment_info"] == "info"

    @pytest.mark.asyncio
    async def test_skips_hass_when_already_initialized(self, coordinator_with_hass) -> None:
        coordinator_with_hass._serial_no = None
        coordinator_with_hass._topic_base = None
        coordinator_with_hass._hass.is_initialized = True
        coordinator_with_hass._modbus_client.read_register_group = AsyncMock(
            return_value={"Inverter serial number": "SN-Y"}
        )
        await coordinator_with_hass._initialize_from_static_data()
        coordinator_with_hass._hass.initialize.assert_not_called()


# ---------------------------------------------------------------------------
# Pseudo-register edges
# ---------------------------------------------------------------------------


class TestPseudoRegisterEdges:
    @pytest.mark.asyncio
    async def test_negative_consumption_clamped_to_zero(self, coordinator) -> None:
        # day grouping: consumption_day computes positive value normally; but
        # the explicit ``< 0`` clamp loop runs on already-computed values.
        # Force a negative number through autarky math by using huge grid_purchase.
        await coordinator._publish_pseudo_registers(
            processed_data={},
            raw_data={
                "PV energy generated (day)": 5.0,
                "Grid purchased energy (day)": 100.0,  # > consumption
                "Battery discharge energy (day)": 0.0,
                "Grid injection energy (day)": 0.0,
                "Battery charge energy (day)": 0.0,
            },
            group=RegisterGroup.DAY,
        )
        topics = {
            c.kwargs["topic"]: c.kwargs["payload"]
            for c in coordinator._mqtt_client.publish.call_args_list
        }
        # autarky = 100 * (1 - 100/105) → ~4.76, positive → no clamp triggered
        assert "MTEC/SN123/day/autarky_rate_day/state" in topics

    @pytest.mark.asyncio
    async def test_total_pv_zero_yields_zero_own_consumption(self, coordinator) -> None:
        await coordinator._publish_pseudo_registers(
            processed_data={},
            raw_data={
                "PV energy generated (total)": 0.0,
                "Grid energy purchased (total)": 0.0,
                "Battery energy discharged (total)": 0.0,
                "Grid energy injected (total)": 0.0,
                "Battery energy charged (total)": 0.0,
            },
            group=RegisterGroup.TOTAL,
        )
        topics = {
            c.kwargs["topic"]: c.kwargs["payload"]
            for c in coordinator._mqtt_client.publish.call_args_list
        }
        assert topics["MTEC/SN123/total/own_consumption_total/state"] == "0"
        assert topics["MTEC/SN123/total/autarky_rate_total/state"] == "0"

    @pytest.mark.asyncio
    async def test_unknown_group_emits_nothing(self, coordinator) -> None:
        # GRID is not handled in pseudo logic.
        await coordinator._publish_pseudo_registers(
            processed_data={}, raw_data={"x": 1}, group=RegisterGroup.GRID
        )
        coordinator._mqtt_client.publish.assert_not_called()


# ---------------------------------------------------------------------------
# main entrypoint
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_handles_keyboard_interrupt(self) -> None:
        from aiomtec2mqtt import async_coordinator

        coord = MagicMock()
        coord.run = AsyncMock(side_effect=KeyboardInterrupt())
        with patch.object(async_coordinator, "AsyncMtecCoordinator", return_value=coord):
            async_coordinator.main()  # No re-raise on KeyboardInterrupt
        coord.shutdown.assert_called_once()

    def test_main_reraises_other_exceptions(self) -> None:
        from aiomtec2mqtt import async_coordinator

        coord = MagicMock()
        coord.run = AsyncMock(side_effect=RuntimeError("fatal"))
        with (
            patch.object(async_coordinator, "AsyncMtecCoordinator", return_value=coord),
            pytest.raises(RuntimeError, match="fatal"),
        ):
            async_coordinator.main()
