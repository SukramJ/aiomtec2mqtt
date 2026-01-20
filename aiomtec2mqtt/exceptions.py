"""
Exceptions for aiomtec2mqtt.

This module provides a typed exception hierarchy that enables proper error
discrimination and handling throughout the application.

(c) 2026 by SukramJ

Exception Hierarchy:
    MtecException (base)
    ├── ConfigException
    │   ├── ConfigValidationError
    │   ├── ConfigFileNotFoundError
    │   └── ConfigParseError
    ├── ModbusException
    │   ├── ModbusConnectionError
    │   ├── ModbusTimeoutError
    │   ├── ModbusReadError
    │   ├── ModbusWriteError
    │   └── ModbusDeviceError
    ├── MqttException
    │   ├── MqttConnectionError
    │   ├── MqttPublishError
    │   ├── MqttSubscribeError
    │   └── MqttAuthenticationError
    └── RetryableException (mixin)
"""

from __future__ import annotations

from typing import Any


class MtecException(Exception):
    """
    Base exception for aiomtec2mqtt.

    All custom exceptions in this application should inherit from this base class.
    This allows catching all application-specific errors with a single except clause.
    """

    def __init__(self, *, message: str, details: dict[str, Any] | None = None) -> None:
        """
        Initialize exception with message and optional details.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context

        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class RetryableException(MtecException):
    """
    Mixin for exceptions that can be retried.

    Exceptions that inherit from this indicate transient failures that may
    succeed on retry (network issues, temporary device unavailability, etc.).
    """


# ============================================================================
# Configuration Exceptions
# ============================================================================


class ConfigException(MtecException):
    """Base exception for configuration-related errors."""


class ConfigValidationError(ConfigException):
    """
    Configuration validation failed.

    Raised when the configuration file contains invalid values or missing
    required fields.
    """


class ConfigFileNotFoundError(ConfigException):
    """
    Configuration file not found.

    Raised when the expected configuration file does not exist at the
    specified path.
    """


class ConfigParseError(ConfigException):
    """
    Configuration file parsing failed.

    Raised when the configuration file exists but cannot be parsed
    (e.g., invalid YAML syntax).
    """


# ============================================================================
# Modbus Exceptions
# ============================================================================


class ModbusException(MtecException):
    """Base exception for Modbus communication errors."""

    def __init__(
        self,
        *,
        message: str,
        address: int | None = None,
        slave_id: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize Modbus exception with optional address and slave ID.

        Args:
            message: Human-readable error message
            address: Modbus register address (if applicable)
            slave_id: Modbus slave ID (if applicable)
            details: Optional dictionary with additional error context

        """
        super().__init__(message=message, details=details)
        self.address = address
        self.slave_id = slave_id


class ModbusConnectionError(ModbusException, RetryableException):
    """
    Failed to establish Modbus connection.

    This is a retryable exception - connection failures are often transient
    and may succeed on retry.
    """


class ModbusTimeoutError(ModbusException, RetryableException):
    """
    Modbus operation timed out.

    This is a retryable exception - timeouts may be caused by temporary
    device load or network congestion.
    """


class ModbusReadError(ModbusException, RetryableException):
    """
    Failed to read from Modbus register.

    This is a retryable exception - read errors may be transient.
    """


class ModbusWriteError(ModbusException, RetryableException):
    """
    Failed to write to Modbus register.

    This is a retryable exception - write errors may be transient.
    """


class ModbusDeviceError(ModbusException):
    """
    Device returned an error response.

    This is NOT retryable - device errors indicate invalid register addresses,
    unsupported operations, or device-specific issues that won't resolve on retry.
    """


# ============================================================================
# MQTT Exceptions
# ============================================================================


class MqttException(MtecException):
    """Base exception for MQTT communication errors."""

    def __init__(
        self,
        *,
        message: str,
        topic: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize MQTT exception with optional topic.

        Args:
            message: Human-readable error message
            topic: MQTT topic (if applicable)
            details: Optional dictionary with additional error context

        """
        super().__init__(message=message, details=details)
        self.topic = topic


class MqttConnectionError(MqttException, RetryableException):
    """
    Failed to establish MQTT connection.

    This is a retryable exception - connection failures are often transient.
    """


class MqttPublishError(MqttException, RetryableException):
    """
    Failed to publish MQTT message.

    This is a retryable exception - publish failures may be transient.
    """


class MqttSubscribeError(MqttException, RetryableException):
    """
    Failed to subscribe to MQTT topic.

    This is a retryable exception - subscribe failures may be transient.
    """


class MqttAuthenticationError(MqttException):
    """
    MQTT authentication failed.

    This is NOT retryable - authentication failures indicate invalid credentials
    or authorization issues that won't resolve on retry.
    """
