"""
Shutdown management for aiomtec2mqtt.

This module provides thread-safe shutdown signaling to replace the global
run_status variable. It supports graceful shutdown on signals and can be
used in both synchronous and asynchronous contexts.

(c) 2026 by SukramJ
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import signal
import threading
from typing import Any, Final

_LOGGER: Final = logging.getLogger(__name__)


class ShutdownManager:
    """
    Thread-safe shutdown manager.

    This class manages application shutdown using threading.Event for thread-safe
    signaling. It supports signal handlers and shutdown callbacks.

    Example:
        shutdown_manager = ShutdownManager()

        # Register signal handlers
        shutdown_manager.register_signal_handlers()

        # Main loop
        while not shutdown_manager.should_shutdown():
            # ... work ...

        # Or wait for shutdown
        shutdown_manager.wait(timeout=5.0)

    """

    def __init__(self) -> None:
        """Initialize shutdown manager."""
        self._shutdown_event = threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        _LOGGER.debug("Shutdown manager initialized")

    def register_callback(self, *, callback: Callable[[], None]) -> None:
        """
        Register a callback to be called on shutdown.

        Callbacks are called in the order they were registered.

        Args:
            callback: Function to call on shutdown (no arguments)

        """
        with self._lock:
            self._callbacks.append(callback)
            _LOGGER.debug("Registered shutdown callback: %s", callback.__name__)

    def register_signal_handlers(self) -> None:
        """Register signal handlers for SIGTERM and SIGINT."""
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        _LOGGER.info("Signal handlers registered (SIGTERM, SIGINT)")

    def reset(self) -> None:
        """Reset shutdown state (for testing)."""
        self._shutdown_event.clear()
        with self._lock:
            self._callbacks.clear()
        _LOGGER.debug("Shutdown manager reset")

    def should_shutdown(self) -> bool:
        """
        Check if shutdown has been triggered.

        Returns:
            True if shutdown was triggered, False otherwise

        """
        return self._shutdown_event.is_set()

    def shutdown(self) -> None:
        """Trigger shutdown and execute callbacks."""
        if self._shutdown_event.is_set():
            _LOGGER.debug("Shutdown already triggered")
            return

        _LOGGER.info("Triggering shutdown")
        self._shutdown_event.set()

        # Execute callbacks
        with self._lock:
            for callback in self._callbacks:
                try:
                    _LOGGER.debug("Executing shutdown callback: %s", callback.__name__)
                    callback()
                except Exception:  # noqa: BLE001
                    _LOGGER.exception(
                        "Error in shutdown callback %s",
                        callback.__name__,
                    )

    def signal_handler(self, signal_number: int, _frame: Any) -> None:  # kwonly: disable
        """
        Handle shutdown signals.

        Args:
            signal_number: Signal number received
            _frame: Stack frame (unused)

        """
        signal_name = signal.Signals(signal_number).name
        _LOGGER.warning(
            "Received signal %s (%d). Graceful shutdown initiated.",
            signal_name,
            signal_number,
        )
        self.shutdown()

    def wait(self, *, timeout: float | None = None) -> bool:
        """
        Wait for shutdown signal.

        Args:
            timeout: Maximum time to wait in seconds (None = indefinite)

        Returns:
            True if shutdown was triggered, False if timeout occurred

        """
        return self._shutdown_event.wait(timeout=timeout)


# Global singleton instance for convenience
_shutdown_manager: ShutdownManager | None = None
_manager_lock = threading.Lock()


def get_shutdown_manager() -> ShutdownManager:
    """
    Get global shutdown manager instance (singleton).

    Returns:
        Global ShutdownManager instance

    """
    global _shutdown_manager  # noqa: PLW0603  # pylint: disable=global-statement

    if _shutdown_manager is None:
        with _manager_lock:
            if _shutdown_manager is None:
                _shutdown_manager = ShutdownManager()

    return _shutdown_manager
