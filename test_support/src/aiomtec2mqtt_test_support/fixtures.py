"""
Pytest fixtures published as a plugin via the ``pytest11`` entry point.

Importing ``aiomtec2mqtt_test_support.fixtures`` is *not* required in user
code — ``pytest`` discovers and registers the fixtures automatically once the
package is installed. The fixtures are intentionally narrow: each one returns
a freshly initialised object so tests stay isolated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest

from aiomtec2mqtt.mock_modbus_server import MockModbusTransport
from aiomtec2mqtt.session_replay import Frame, SessionPlayer
from aiomtec2mqtt.testing import (
    FakeConfigProvider,
    FakeHealthMonitor,
    FakeModbusClient,
    FakeMqttClient,
)

__all__ = [
    "fake_config",
    "fake_health",
    "fake_modbus_client",
    "fake_mqtt_client",
    "mock_modbus",
    "replay_session",
]


@pytest.fixture
def mock_modbus() -> MockModbusTransport:
    """Fresh, empty :class:`MockModbusTransport` for one test."""
    return MockModbusTransport()


@pytest.fixture
def fake_modbus_client() -> FakeModbusClient:
    """Protocol-conformant Modbus fake with no preloaded register data."""
    return FakeModbusClient()


@pytest.fixture
def fake_mqtt_client() -> FakeMqttClient:
    """Protocol-conformant MQTT fake that records every publish."""
    return FakeMqttClient()


@pytest.fixture
def fake_config() -> FakeConfigProvider:
    """Empty config provider you can populate via ``set(key=..., value=...)``."""
    return FakeConfigProvider()


@pytest.fixture
def fake_health() -> FakeHealthMonitor:
    """Health monitor with no registered components."""
    return FakeHealthMonitor()


@pytest.fixture
def replay_session() -> Callable[..., SessionPlayer]:
    """Factory fixture: build a :class:`SessionPlayer` from frame definitions.

    Usage::

        def test_x(replay_session):
            player = replay_session(frames=[{100: [1, 2]}, {100: [3, 4]}])
            ...

    Each ``frames`` entry is the register dict for one frame; timestamps are
    auto-assigned in 1-second increments because most tests do not care about
    them.
    """

    def _factory(
        *,
        frames: Iterable[dict[int, list[int]]],
        loop: bool = True,
    ) -> SessionPlayer:
        frame_objs = [
            Frame(ts=float(idx), registers=dict(regs))
            for idx, regs in enumerate(frames)
        ]
        return SessionPlayer(frame_objs, loop=loop)

    return _factory
