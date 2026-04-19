"""
Client subpackage — Modbus + MQTT transport-level clients.

Re-exports the existing ``AsyncModbusClient`` and ``AsyncMqttClient`` under a
cohesive namespace so callers can write::

    from aiomtec2mqtt.client import AsyncModbusClient, AsyncMqttClient

The original flat modules (:mod:`aiomtec2mqtt.async_modbus_client`,
:mod:`aiomtec2mqtt.async_mqtt_client`) remain in place for backwards
compatibility. When the project completes its sub-package migration, the
implementations will move into this directory and the flat modules will
become shims.

(c) 2026 by SukramJ
"""

from __future__ import annotations

from aiomtec2mqtt.async_modbus_client import AsyncModbusClient
from aiomtec2mqtt.async_mqtt_client import AsyncMqttClient

__all__ = ["AsyncModbusClient", "AsyncMqttClient"]
