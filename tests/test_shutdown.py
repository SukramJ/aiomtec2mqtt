"""Tests for shutdown manager."""

from __future__ import annotations

import signal
import threading
import time

from aiomtec2mqtt.shutdown import ShutdownManager, get_shutdown_manager


class TestShutdownManager:
    """Tests for ShutdownManager class."""

    def test_duplicate_shutdown_calls_idempotent(self) -> None:
        """Multiple shutdown calls should be idempotent."""
        manager = ShutdownManager()

        callback_count = [0]

        def callback() -> None:
            callback_count[0] += 1

        manager.register_callback(callback=callback)

        # Call shutdown multiple times
        manager.shutdown()
        manager.shutdown()
        manager.shutdown()

        # Callback should only execute once
        assert callback_count[0] == 1
        assert manager.should_shutdown()

    def test_initialization(self) -> None:
        """Shutdown manager should initialize in non-shutdown state."""
        manager = ShutdownManager()

        assert not manager.should_shutdown()

    def test_manual_shutdown(self) -> None:
        """Manual shutdown should set shutdown flag."""
        manager = ShutdownManager()

        assert not manager.should_shutdown()

        manager.shutdown()

        assert manager.should_shutdown()

    def test_register_signal_handlers(self) -> None:
        """Register signal handlers should not raise."""
        manager = ShutdownManager()

        # Should not raise
        manager.register_signal_handlers()

        # Cleanup: restore default handlers
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def test_reset(self) -> None:
        """Reset should clear shutdown state and callbacks."""
        manager = ShutdownManager()

        callback_executed = []

        def callback() -> None:
            callback_executed.append(1)

        manager.register_callback(callback=callback)
        manager.shutdown()

        assert manager.should_shutdown()
        assert len(callback_executed) == 1

        # Reset
        manager.reset()

        assert not manager.should_shutdown()

        # Trigger shutdown again - callback should not execute (was cleared)
        manager.shutdown()
        assert len(callback_executed) == 1  # Still 1, not 2

    def test_shutdown_callbacks_execute(self) -> None:
        """Shutdown callbacks should execute on shutdown."""
        manager = ShutdownManager()

        callback_executed = []

        def callback1() -> None:
            callback_executed.append(1)

        def callback2() -> None:
            callback_executed.append(2)

        manager.register_callback(callback=callback1)
        manager.register_callback(callback=callback2)

        manager.shutdown()

        # Callbacks should execute in order
        assert callback_executed == [1, 2]

    def test_shutdown_callbacks_handle_errors(self) -> None:
        """Shutdown should continue even if callback raises exception."""
        manager = ShutdownManager()

        callback_executed = []

        def failing_callback() -> None:
            callback_executed.append("fail")
            raise ValueError("Callback error")

        def success_callback() -> None:
            callback_executed.append("success")

        manager.register_callback(callback=failing_callback)
        manager.register_callback(callback=success_callback)

        # Should not raise, even though first callback fails
        manager.shutdown()

        # Both callbacks should have been attempted
        assert callback_executed == ["fail", "success"]

    def test_signal_handler(self) -> None:
        """Signal handler should trigger shutdown."""
        manager = ShutdownManager()

        assert not manager.should_shutdown()

        # Simulate signal (don't actually send signal to avoid issues)
        manager.signal_handler(signal_number=signal.SIGTERM, _frame=None)

        assert manager.should_shutdown()

    def test_wait_with_shutdown(self) -> None:
        """Wait should return True immediately when shutdown occurs."""
        manager = ShutdownManager()

        # Trigger shutdown in another thread after delay
        def trigger_shutdown() -> None:
            time.sleep(0.05)
            manager.shutdown()

        thread = threading.Thread(target=trigger_shutdown)
        thread.start()

        # Wait should return True when shutdown happens
        start = time.time()
        result = manager.wait(timeout=1.0)
        elapsed = time.time() - start

        assert result is True
        assert elapsed < 0.2  # Should return quickly, not wait full timeout

        thread.join()

    def test_wait_with_timeout_no_shutdown(self) -> None:
        """Wait should return False if timeout occurs without shutdown."""
        manager = ShutdownManager()

        # Wait for 0.1 seconds without shutdown
        start = time.time()
        result = manager.wait(timeout=0.1)
        elapsed = time.time() - start

        assert result is False  # Timeout occurred
        assert 0.08 <= elapsed <= 0.15  # Approximately 0.1s


class TestGlobalShutdownManager:
    """Tests for global shutdown manager singleton."""

    def test_singleton_returns_same_instance(self) -> None:
        """Multiple calls should return the same instance."""
        manager1 = get_shutdown_manager()
        manager2 = get_shutdown_manager()

        assert manager1 is manager2

    def test_singleton_thread_safe(self) -> None:
        """Singleton should be thread-safe."""
        managers = []

        def get_manager() -> None:
            managers.append(get_shutdown_manager())

        # Create multiple threads getting manager simultaneously
        threads = [threading.Thread(target=get_manager) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All should be the same instance
        assert len(managers) == 10
        assert all(m is managers[0] for m in managers)


class TestShutdownManagerIntegration:
    """Integration tests for shutdown manager."""

    def test_graceful_shutdown_with_cleanup(self) -> None:
        """Test graceful shutdown with cleanup callbacks."""
        manager = ShutdownManager()
        manager.reset()

        cleanup_order = []

        def cleanup_mqtt() -> None:
            cleanup_order.append("mqtt")

        def cleanup_modbus() -> None:
            cleanup_order.append("modbus")

        def cleanup_logging() -> None:
            cleanup_order.append("logging")

        # Register cleanup callbacks
        manager.register_callback(callback=cleanup_mqtt)
        manager.register_callback(callback=cleanup_modbus)
        manager.register_callback(callback=cleanup_logging)

        # Simulate main loop
        for i in range(3):
            if i == 2:
                manager.shutdown()

        # Verify cleanup happened in order
        assert cleanup_order == ["mqtt", "modbus", "logging"]

    def test_main_loop_pattern(self) -> None:
        """Test typical main loop pattern."""
        manager = ShutdownManager()
        manager.reset()  # Ensure clean state

        loop_iterations = 0
        max_iterations = 5

        # Simulate main loop
        while not manager.should_shutdown():
            loop_iterations += 1

            # Trigger shutdown after a few iterations
            if loop_iterations >= max_iterations:
                manager.shutdown()

            time.sleep(0.01)

        assert loop_iterations == max_iterations
        assert manager.should_shutdown()
