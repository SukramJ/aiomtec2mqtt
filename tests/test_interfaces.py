"""Smoke tests for :mod:`aiomtec2mqtt.interfaces` re-export surface."""

from __future__ import annotations


def test_interfaces_reexport_matches_protocols() -> None:
    """Both modules must expose identical protocol objects (not just names)."""
    from aiomtec2mqtt import interfaces, protocols

    assert interfaces.ModbusClientProtocol is protocols.ModbusClientProtocol
    assert interfaces.MqttClientProtocol is protocols.MqttClientProtocol
    assert interfaces.CoordinatorProtocol is protocols.CoordinatorProtocol
    assert interfaces.ConfigProviderProtocol is protocols.ConfigProviderProtocol
    assert interfaces.HealthMonitorProtocol is protocols.HealthMonitorProtocol
    assert interfaces.RegisterProcessorProtocol is protocols.RegisterProcessorProtocol
    assert interfaces.FormulaEvaluatorProtocol is protocols.FormulaEvaluatorProtocol


def test_coordinator_protocol_is_runtime_checkable() -> None:
    from aiomtec2mqtt.interfaces import CoordinatorProtocol

    class _Fake:
        async def run(self) -> None:
            return None

        async def shutdown(self) -> None:
            return None

    assert isinstance(_Fake(), CoordinatorProtocol)


def test_coordinator_protocol_rejects_incomplete_impl() -> None:
    from aiomtec2mqtt.interfaces import CoordinatorProtocol

    class _Missing:
        async def run(self) -> None:
            return None

    assert not isinstance(_Missing(), CoordinatorProtocol)
