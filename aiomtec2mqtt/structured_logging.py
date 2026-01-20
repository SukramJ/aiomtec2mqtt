"""
Structured logging with JSON output.

This module provides structured logging capabilities with JSON formatting
for better log aggregation, parsing, and analysis in production environments.

(c) 2026 by SukramJ
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import sys
import traceback
from typing import Any, Final

_LOGGER: Final = logging.getLogger(__name__)


class StructuredLogger:
    """
    Structured logger with JSON output.

    Provides methods for logging with structured context fields,
    making logs easier to parse and analyze in log aggregation systems.
    """

    def __init__(
        self,
        *,
        name: str,
        level: int = logging.INFO,
        enable_json: bool = True,
    ) -> None:
        """
        Initialize structured logger.

        Args:
            name: Logger name
            level: Log level
            enable_json: Enable JSON formatting (if False, uses standard format)

        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.enable_json = enable_json

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        if enable_json:
            handler.setFormatter(JSONFormatter())
        else:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def debug(self, *, message: str, **context: Any) -> None:
        """
        Log debug message.

        Args:
            message: Log message
            **context: Additional context fields

        """
        self.log(level=logging.DEBUG, message=message, **context)

    def error(self, *, message: str, **context: Any) -> None:
        """
        Log error message.

        Args:
            message: Log message
            **context: Additional context fields

        """
        self.log(level=logging.ERROR, message=message, **context)

    def exception(self, *, message: str, exc: Exception | None = None, **context: Any) -> None:
        """
        Log exception with traceback.

        Args:
            message: Log message
            exc: Exception instance
            **context: Additional context fields

        """
        if exc:
            context["exception_type"] = type(exc).__name__
            context["exception_message"] = str(exc)
            context["traceback"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )

        self.log(level=logging.ERROR, message=message, **context)

    def info(self, *, message: str, **context: Any) -> None:
        """
        Log info message.

        Args:
            message: Log message
            **context: Additional context fields

        """
        self.log(level=logging.INFO, message=message, **context)

    def log(
        self,
        *,
        level: int,
        message: str,
        **context: Any,
    ) -> None:
        """
        Log message with context.

        Args:
            level: Log level
            message: Log message
            **context: Additional context fields

        """
        if self.enable_json:
            # Pass context as extra fields
            self.logger.log(level, message, extra={"context": context})
        else:
            # Format context as key=value pairs
            if context:
                context_str = " ".join(f"{k}={v}" for k, v in context.items())
                message = f"{message} | {context_str}"
            self.logger.log(level, message)

    def warning(self, *, message: str, **context: Any) -> None:
        """
        Log warning message.

        Args:
            message: Log message
            **context: Additional context fields

        """
        self.log(level=logging.WARNING, message=message, **context)


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter.

    Formats log records as JSON objects with structured fields
    for better parsing and analysis.
    """

    def format(self, record: logging.LogRecord) -> str:  # kwonly: disable
        """
        Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON formatted log string

        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context fields if present
        if hasattr(record, "context"):
            log_data.update(record.context)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_data)


def setup_structured_logging(
    *,
    level: int = logging.INFO,
    enable_json: bool = True,
) -> None:
    """
    Set up structured logging for the entire application.

    Args:
        level: Log level
        enable_json: Enable JSON formatting

    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if enable_json:
        handler.setFormatter(JSONFormatter())
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    _LOGGER.info(
        "Structured logging initialized", extra={"context": {"json_enabled": enable_json}}
    )


def get_structured_logger(*, name: str) -> StructuredLogger:
    """
    Get structured logger instance.

    Args:
        name: Logger name

    Returns:
        StructuredLogger instance

    """
    return StructuredLogger(name=name)
