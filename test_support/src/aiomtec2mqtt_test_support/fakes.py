"""
Re-exports for the protocol-conformant fake clients shipped by ``aiomtec2mqtt``.

These classes implement the `*Protocol` types from ``aiomtec2mqtt.protocols``
without touching real networks, sockets, or files. They are the right choice
for unit tests that need a working stand-in but do not care about wire-level
behaviour. For tests that drive the production ``AsyncModbusClient`` code
paths, use :class:`MockModbusTransport` from
:mod:`aiomtec2mqtt_test_support.transports` instead.
"""

from __future__ import annotations

from aiomtec2mqtt.testing import (
    FakeConfigProvider,
    FakeHealthMonitor,
    FakeModbusClient,
    FakeMqttClient,
    create_test_container,
)

__all__ = [
    "FakeConfigProvider",
    "FakeHealthMonitor",
    "FakeModbusClient",
    "FakeMqttClient",
    "create_test_container",
]
