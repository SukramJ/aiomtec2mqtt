"""Tests for structured logging."""

from __future__ import annotations

import json
import logging

import pytest

from aiomtec2mqtt.structured_logging import (
    JSONFormatter,
    StructuredLogger,
    get_structured_logger,
    setup_structured_logging,
)


class TestStructuredLogger:
    """Test StructuredLogger class."""

    def test_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test debug logging."""
        logger = StructuredLogger(name="test_logger", level=logging.DEBUG, enable_json=False)

        with caplog.at_level(logging.DEBUG):
            logger.debug(message="Debug message")

        assert "Debug message" in caplog.text

    def test_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test error logging."""
        logger = StructuredLogger(name="test_logger", enable_json=False)

        with caplog.at_level(logging.ERROR):
            logger.error(message="Error message")

        assert "Error message" in caplog.text

    def test_exception_with_exc(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test exception logging with exception object."""
        logger = StructuredLogger(name="test_logger", enable_json=False)

        exc = ValueError("Test error")

        with caplog.at_level(logging.ERROR):
            logger.exception(message="Exception occurred", exc=exc)

        assert "Exception occurred" in caplog.text

    def test_initialization(self) -> None:
        """Test logger initialization."""
        logger = StructuredLogger(name="test_logger", enable_json=False)

        assert logger.logger.name == "test_logger"
        assert logger.logger.level == logging.INFO
        assert logger.enable_json is False

    def test_initialization_custom_level(self) -> None:
        """Test initialization with custom log level."""
        logger = StructuredLogger(name="test_logger", level=logging.DEBUG, enable_json=False)

        assert logger.logger.level == logging.DEBUG

    def test_initialization_with_json(self) -> None:
        """Test logger initialization with JSON formatting."""
        logger = StructuredLogger(name="test_logger", enable_json=True)

        assert logger.enable_json is True
        assert len(logger.logger.handlers) == 1
        assert isinstance(logger.logger.handlers[0].formatter, JSONFormatter)

    def test_log_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test info logging."""
        logger = StructuredLogger(name="test_logger", enable_json=False)

        with caplog.at_level(logging.INFO):
            logger.info(message="Test message")

        assert "Test message" in caplog.text

    def test_log_with_context_non_json(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging with context in non-JSON mode."""
        logger = StructuredLogger(name="test_logger", enable_json=False)

        with caplog.at_level(logging.INFO):
            logger.info(message="Test message", component="modbus", operation="read")

        assert "Test message" in caplog.text
        assert "component=modbus" in caplog.text
        assert "operation=read" in caplog.text

    def test_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning logging."""
        logger = StructuredLogger(name="test_logger", enable_json=False)

        with caplog.at_level(logging.WARNING):
            logger.warning(message="Warning message")

        assert "Warning message" in caplog.text


class TestJSONFormatter:
    """Test JSONFormatter class."""

    def test_format_basic_record(self) -> None:
        """Test formatting basic log record."""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert data["line"] == 10
        assert "timestamp" in data

    def test_format_with_context(self) -> None:
        """Test formatting record with context."""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.context = {"component": "modbus", "operation": "read"}  # type: ignore[attr-defined]

        output = formatter.format(record)
        data = json.loads(output)

        assert data["component"] == "modbus"
        assert data["operation"] == "read"

    def test_format_with_exception(self) -> None:
        """Test formatting record with exception."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "Test error"
        assert "traceback" in data["exception"]


class TestSetupStructuredLogging:
    """Test setup_structured_logging function."""

    def test_setup_json_logging(self) -> None:
        """Test setting up JSON logging."""
        setup_structured_logging(enable_json=True, level=logging.INFO)

        root_logger = logging.getLogger()

        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) >= 1
        # Check if any handler has JSONFormatter
        has_json_formatter = any(
            isinstance(handler.formatter, JSONFormatter) for handler in root_logger.handlers
        )
        assert has_json_formatter

    def test_setup_standard_logging(self) -> None:
        """Test setting up standard logging."""
        setup_structured_logging(enable_json=False, level=logging.DEBUG)

        root_logger = logging.getLogger()

        assert root_logger.level == logging.DEBUG
        assert len(root_logger.handlers) >= 1


class TestGetStructuredLogger:
    """Test get_structured_logger function."""

    def test_get_logger(self) -> None:
        """Test getting structured logger."""
        logger = get_structured_logger(name="test_logger")

        assert isinstance(logger, StructuredLogger)
        assert logger.logger.name == "test_logger"

    def test_multiple_loggers(self) -> None:
        """Test getting multiple loggers."""
        logger1 = get_structured_logger(name="logger1")
        logger2 = get_structured_logger(name="logger2")

        assert logger1.logger.name == "logger1"
        assert logger2.logger.name == "logger2"
        assert logger1 is not logger2
