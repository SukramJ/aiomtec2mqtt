"""Smoke tests verifying the public surface and the pytest plugin entry point."""

from __future__ import annotations

import pytest


def test_version_exposed() -> None:
    import aiomtec2mqtt_test_support

    assert isinstance(aiomtec2mqtt_test_support.__version__, str)


def test_transports_re_exports_resolve() -> None:
    from aiomtec2mqtt_test_support.transports import (
        Frame,
        MockModbusResponse,
        MockModbusTransport,
        SessionMetadata,
        SessionPlayer,
        SessionRecorder,
    )

    # Make sure each name actually points at the upstream class.
    from aiomtec2mqtt.mock_modbus_server import (
        MockModbusResponse as UpstreamResponse,
        MockModbusTransport as UpstreamTransport,
    )
    from aiomtec2mqtt.session_replay import (
        Frame as UpstreamFrame,
        SessionMetadata as UpstreamMeta,
        SessionPlayer as UpstreamPlayer,
        SessionRecorder as UpstreamRecorder,
    )

    assert MockModbusResponse is UpstreamResponse
    assert MockModbusTransport is UpstreamTransport
    assert Frame is UpstreamFrame
    assert SessionMetadata is UpstreamMeta
    assert SessionPlayer is UpstreamPlayer
    assert SessionRecorder is UpstreamRecorder


def test_fakes_re_exports_resolve() -> None:
    from aiomtec2mqtt_test_support.fakes import FakeModbusClient

    from aiomtec2mqtt.testing import FakeModbusClient as UpstreamFake

    assert FakeModbusClient is UpstreamFake


@pytest.mark.asyncio
async def test_mock_modbus_fixture(mock_modbus) -> None:
    """The plugin fixture is registered automatically (no import needed)."""
    mock_modbus.register_values[100] = [42]
    await mock_modbus.connect()
    response = await mock_modbus.read_holding_registers(100)
    assert response.registers == [42]


def test_fake_mqtt_client_fixture(fake_mqtt_client) -> None:
    assert fake_mqtt_client.published_messages == []


def test_replay_session_factory(replay_session) -> None:
    player = replay_session(frames=[{100: [1]}, {100: [2]}])
    assert player.frame_count == 2


@pytest.mark.asyncio
async def test_assert_register_published_helper(fake_mqtt_client) -> None:
    from aiomtec2mqtt_test_support.assertions import assert_register_published

    await fake_mqtt_client.connect()
    await fake_mqtt_client.publish(
        topic="MTEC/serial/now-base",
        payload='{"battery_soc": 75, "grid_power": 1500}',
    )
    assert_register_published(
        fake_mqtt_client,
        "battery_soc",
        75,
        topic_suffix="/now-base",
    )


def test_assert_mqtt_topic_seen_failure_message(fake_mqtt_client) -> None:
    from aiomtec2mqtt_test_support.assertions import assert_mqtt_topic_seen

    with pytest.raises(AssertionError, match="No publish to a topic"):
        assert_mqtt_topic_seen(fake_mqtt_client, "/now-base")


@pytest.mark.asyncio
async def test_assert_mqtt_topic_seen_count_mismatch(fake_mqtt_client) -> None:
    from aiomtec2mqtt_test_support.assertions import assert_mqtt_topic_seen

    await fake_mqtt_client.connect()
    await fake_mqtt_client.publish(topic="MTEC/x/now-base", payload="{}")
    with pytest.raises(AssertionError, match="Expected 2 publishes"):
        assert_mqtt_topic_seen(fake_mqtt_client, "/now-base", times=2)
