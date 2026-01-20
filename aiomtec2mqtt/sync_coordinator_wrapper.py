"""
Synchronous wrapper for AsyncMtecCoordinator to maintain backward compatibility.

This module provides a synchronous facade that wraps the async coordinator,
allowing existing code to use the new async implementation without changes.

The wrapper uses asyncio.run() to bridge synchronous and asynchronous code,
ensuring full API compatibility with the original MtecCoordinator.

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

import asyncio
import logging
import signal
import threading
from typing import Any, Final

from aiomtec2mqtt.async_coordinator import AsyncMtecCoordinator

_LOGGER: Final = logging.getLogger(__name__)


class SyncMtecCoordinatorWrapper:
    """
    Synchronous wrapper for AsyncMtecCoordinator.

    This class maintains the same API as the original MtecCoordinator
    but uses the async implementation internally.
    """

    def __init__(self) -> None:
        """Initialize the sync wrapper."""
        self._coordinator = AsyncMtecCoordinator()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_requested: bool = False

        # Register signal handlers
        signal.signal(signalnum=signal.SIGTERM, handler=self._signal_handler)
        signal.signal(signalnum=signal.SIGINT, handler=self._signal_handler)

        _LOGGER.info("Sync coordinator wrapper initialized")

    @property
    def coordinator(self) -> AsyncMtecCoordinator:
        """Get the underlying async coordinator."""
        return self._coordinator

    def is_running(self) -> bool:
        """
        Check if coordinator is running.

        Returns:
            True if running

        """
        return self._thread is not None and self._thread.is_alive()

    def run(self) -> None:
        """
        Run the coordinator in blocking mode.

        This method blocks until shutdown is requested.
        """
        _LOGGER.info("Starting sync coordinator wrapper")

        try:
            # Run async coordinator in sync context
            asyncio.run(self._coordinator.run())
        except KeyboardInterrupt:
            _LOGGER.info("Received keyboard interrupt")
            self.shutdown()
        except Exception:
            _LOGGER.exception("Fatal error in coordinator")
            raise
        finally:
            _LOGGER.info("Sync coordinator wrapper stopped")

    def shutdown(self) -> None:
        """Request shutdown of the coordinator."""
        if self._shutdown_requested:
            return

        _LOGGER.info("Shutdown requested")
        self._shutdown_requested = True

        # Signal async coordinator to shutdown
        self._coordinator.shutdown()

        # Wait for background thread to finish
        if self._thread and self._thread.is_alive():
            _LOGGER.debug("Waiting for coordinator thread to finish")
            self._thread.join(timeout=5.0)

            if self._thread.is_alive():
                _LOGGER.warning("Coordinator thread did not finish in time")

    def start_background(self) -> None:
        """
        Start the coordinator in a background thread.

        This allows the coordinator to run without blocking the main thread.
        """
        if self._thread and self._thread.is_alive():
            _LOGGER.warning("Coordinator already running in background")
            return

        _LOGGER.info("Starting coordinator in background thread")

        self._thread = threading.Thread(
            target=self._run_in_thread,
            name="coordinator",
            daemon=True,
        )
        self._thread.start()

    def _run_in_thread(self) -> None:
        """Run the async coordinator in a separate thread."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._coordinator.run())
        except Exception:
            _LOGGER.exception("Error in coordinator thread")
        finally:
            if self._loop:
                self._loop.close()
            _LOGGER.info("Coordinator thread stopped")

    def _signal_handler(self, signum: int, frame: Any) -> None:  # kwonly: disable
        """
        Handle OS signals.

        Args:
            signum: Signal number
            frame: Current stack frame

        """
        signal_name = signal.Signals(signum).name
        _LOGGER.info("Received signal: %s", signal_name)
        self.shutdown()


# Alias for backward compatibility
MtecCoordinatorAsync = SyncMtecCoordinatorWrapper


def main() -> None:
    """Run sync coordinator wrapper."""
    coordinator = SyncMtecCoordinatorWrapper()

    try:
        coordinator.run()
    except KeyboardInterrupt:
        _LOGGER.info("Received keyboard interrupt")
        coordinator.shutdown()
    except Exception:
        _LOGGER.exception("Fatal error")
        raise


if __name__ == "__main__":
    main()
