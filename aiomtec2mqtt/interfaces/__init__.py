"""
Public protocol interfaces for :mod:`aiomtec2mqtt`.

This namespace groups the ``Protocol`` classes that define the seams between
the coordinator and its dependencies. It re-exports the existing definitions
from :mod:`aiomtec2mqtt.protocols` so callers have a single, stable import
path:

.. code-block:: python

    from aiomtec2mqtt.interfaces import ModbusClientProtocol, MqttClientProtocol

The legacy :mod:`aiomtec2mqtt.protocols` module stays in place for backwards
compatibility; both paths resolve to the same objects.

(c) 2026 by SukramJ
"""

from __future__ import annotations

from aiomtec2mqtt.protocols import (
    ConfigProviderProtocol,
    CoordinatorProtocol,
    FormulaEvaluatorProtocol,
    HealthMonitorProtocol,
    ModbusClientProtocol,
    MqttClientProtocol,
    RegisterProcessorProtocol,
)

__all__ = [
    "ConfigProviderProtocol",
    "CoordinatorProtocol",
    "FormulaEvaluatorProtocol",
    "HealthMonitorProtocol",
    "ModbusClientProtocol",
    "MqttClientProtocol",
    "RegisterProcessorProtocol",
]
