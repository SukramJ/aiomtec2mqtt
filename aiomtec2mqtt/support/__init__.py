"""
Support subpackage — cross-cutting infrastructure.

Bundles exceptions, resilience primitives, health tracking, structured logging,
shutdown handling, and the DI container under one namespace::

    from aiomtec2mqtt.support import CircuitBreaker, HealthCheck, ShutdownManager

(c) 2026 by SukramJ
"""

from __future__ import annotations

from aiomtec2mqtt.container import ServiceContainer, create_container
from aiomtec2mqtt.exceptions import (
    ConfigException,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
    ModbusConnectionError,
    ModbusDeviceError,
    ModbusException,
    ModbusReadError,
    ModbusTimeoutError,
    ModbusWriteError,
    MqttAuthenticationError,
    MqttConnectionError,
    MqttException,
    MqttPublishError,
    MqttSubscribeError,
    MtecException,
    RetryableException,
)
from aiomtec2mqtt.health import ComponentHealth, HealthCheck, HealthStatus, SystemHealth
from aiomtec2mqtt.resilience import (
    BackoffConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerStats,
    CircuitState,
    ConnectionState,
    ConnectionStateInfo,
    ConnectionStateMachine,
    ExponentialBackoff,
)
from aiomtec2mqtt.shutdown import ShutdownManager, get_shutdown_manager
from aiomtec2mqtt.structured_logging import (
    JSONFormatter,
    StructuredLogger,
    get_structured_logger,
    setup_structured_logging,
)

__all__ = [
    "BackoffConfig",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitBreakerStats",
    "CircuitState",
    "ComponentHealth",
    "ConfigException",
    "ConfigFileNotFoundError",
    "ConfigParseError",
    "ConfigValidationError",
    "ConnectionState",
    "ConnectionStateInfo",
    "ConnectionStateMachine",
    "ExponentialBackoff",
    "HealthCheck",
    "HealthStatus",
    "JSONFormatter",
    "ModbusConnectionError",
    "ModbusDeviceError",
    "ModbusException",
    "ModbusReadError",
    "ModbusTimeoutError",
    "ModbusWriteError",
    "MqttAuthenticationError",
    "MqttConnectionError",
    "MqttException",
    "MqttPublishError",
    "MqttSubscribeError",
    "MtecException",
    "RetryableException",
    "ServiceContainer",
    "ShutdownManager",
    "StructuredLogger",
    "SystemHealth",
    "create_container",
    "get_shutdown_manager",
    "get_structured_logger",
    "setup_structured_logging",
]
