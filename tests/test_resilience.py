"""Tests for resilience patterns (backoff, circuit breaker, state machine)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aiomtec2mqtt.exceptions import ModbusTimeoutError
from aiomtec2mqtt.resilience import (
    BackoffConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    ConnectionState,
    ConnectionStateMachine,
    ExponentialBackoff,
)


class TestExponentialBackoff:
    """Tests for exponential backoff utility."""

    def test_backoff_max_retries(self) -> None:
        """Backoff should raise after max_retries."""
        config = BackoffConfig(max_retries=3, jitter=False)
        backoff = ExponentialBackoff(config=config)

        backoff.next_delay()  # Attempt 0
        backoff.next_delay()  # Attempt 1
        backoff.next_delay()  # Attempt 2

        # Fourth attempt should raise
        with pytest.raises(RuntimeError, match="Max retries .* exceeded"):
            backoff.next_delay()

    def test_backoff_reset(self) -> None:
        """Reset should restart backoff sequence."""
        backoff = ExponentialBackoff(config=BackoffConfig(jitter=False))

        backoff.next_delay()  # 1.0
        backoff.next_delay()  # 2.0
        backoff.next_delay()  # 4.0

        backoff.reset()

        # After reset, should start from beginning
        assert backoff.next_delay() == 1.0
        assert backoff.attempt == 1

    def test_backoff_respects_max_delay(self) -> None:
        """Backoff should not exceed max_delay."""
        config = BackoffConfig(initial_delay=1.0, max_delay=5.0, multiplier=2.0, max_retries=None)
        backoff = ExponentialBackoff(config=config)

        # Get many delays - should eventually cap at max_delay
        delays = [backoff.next_delay() for _ in range(10)]

        # All delays should be <= max_delay + jitter allowance
        for delay in delays:
            assert delay <= 6.5  # max_delay * 1.3 (accounting for jitter)

    def test_backoff_unlimited_retries(self) -> None:
        """Backoff with max_retries=None should never raise."""
        config = BackoffConfig(max_retries=None, jitter=False)
        backoff = ExponentialBackoff(config=config)

        # Should work for many attempts
        for _ in range(100):
            delay = backoff.next_delay()
            assert delay > 0  # Should always return valid delay

    def test_backoff_without_jitter(self) -> None:
        """Backoff without jitter should have exact values."""
        config = BackoffConfig(
            initial_delay=1.0,
            max_delay=60.0,
            multiplier=2.0,
            jitter=False,
        )
        backoff = ExponentialBackoff(config=config)

        # Without jitter, delays should be exact
        assert backoff.next_delay() == 1.0
        assert backoff.next_delay() == 2.0
        assert backoff.next_delay() == 4.0
        assert backoff.next_delay() == 8.0
        assert backoff.next_delay() == 16.0

    def test_default_backoff_sequence(self) -> None:
        """Verify exponential backoff sequence with default config."""
        backoff = ExponentialBackoff()

        # First delays should follow exponential pattern: 1, 2, 4, 8, 16...
        delays = [backoff.next_delay() for _ in range(5)]

        # With jitter enabled, check approximate ranges (±25%)
        assert 0.75 <= delays[0] <= 1.25  # ~1s
        assert 1.5 <= delays[1] <= 2.5  # ~2s
        assert 3.0 <= delays[2] <= 5.0  # ~4s
        assert 6.0 <= delays[3] <= 10.0  # ~8s
        assert 12.0 <= delays[4] <= 20.0  # ~16s


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    def test_circuit_opens_after_threshold_failures(self) -> None:
        """Circuit should open after failure_threshold retryable failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(name="test", config=config)

        def failing_func() -> None:
            raise ModbusTimeoutError(message="Timeout")

        # First 3 failures should open circuit
        for _ in range(3):
            with pytest.raises(ModbusTimeoutError):
                breaker.call(func=failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open
        assert breaker.stats.failed_calls == 3

    def test_circuit_starts_closed(self) -> None:
        """Circuit breaker should start in CLOSED state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open
        assert not breaker.is_half_open

    def test_circuit_transitions_to_half_open(self) -> None:
        """Circuit should transition from OPEN to HALF_OPEN after timeout."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        breaker = CircuitBreaker(name="test", config=config)

        def failing_func() -> None:
            raise ModbusTimeoutError(message="Timeout")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ModbusTimeoutError):
                breaker.call(func=failing_func)

        assert breaker.is_open

        # Wait for timeout (using internal method for testing)
        import time

        time.sleep(0.15)

        # Check state - should be HALF_OPEN now
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.is_half_open

    def test_half_open_closes_after_success_threshold(self) -> None:
        """HALF_OPEN circuit should close after success_threshold successes."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout=0.1,
            half_open_max_calls=2,  # Allow 2 test calls in HALF_OPEN
        )
        breaker = CircuitBreaker(name="test", config=config)

        def failing_func() -> None:
            raise ModbusTimeoutError(message="Timeout")

        def success_func() -> str:
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ModbusTimeoutError):
                breaker.call(func=failing_func)

        # Wait for HALF_OPEN
        import time

        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Successful calls should close circuit
        breaker.call(func=success_func)
        assert breaker.is_half_open  # Still half-open after 1 success

        breaker.call(func=success_func)
        assert breaker.is_closed  # Closed after 2 successes

    def test_half_open_reopens_on_failure(self) -> None:
        """HALF_OPEN circuit should reopen immediately on failure."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        breaker = CircuitBreaker(name="test", config=config)

        def failing_func() -> None:
            raise ModbusTimeoutError(message="Timeout")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ModbusTimeoutError):
                breaker.call(func=failing_func)

        # Wait for HALF_OPEN
        import time

        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Single failure should reopen
        with pytest.raises(ModbusTimeoutError):
            breaker.call(func=failing_func)

        assert breaker.is_open

    def test_manual_reset(self) -> None:
        """Manual reset should close the circuit."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(name="test", config=config)

        def failing_func() -> None:
            raise ModbusTimeoutError(message="Timeout")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ModbusTimeoutError):
                breaker.call(func=failing_func)

        assert breaker.is_open

        # Manual reset
        breaker.reset()
        assert breaker.is_closed

    def test_non_retryable_exceptions_dont_open_circuit(self) -> None:
        """Non-retryable exceptions should not trigger circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(name="test", config=config)

        def non_retryable_func() -> None:
            raise ValueError("Not retryable")

        # Multiple non-retryable failures should not open circuit
        for _ in range(5):
            with pytest.raises(ValueError):
                breaker.call(func=non_retryable_func)

        assert breaker.is_closed  # Circuit should still be closed

    def test_open_circuit_rejects_calls(self) -> None:
        """Open circuit should reject calls immediately."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(name="test", config=config)

        def failing_func() -> None:
            raise ModbusTimeoutError(message="Timeout")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ModbusTimeoutError):
                breaker.call(func=failing_func)

        assert breaker.is_open

        # Next call should be rejected without calling function
        with pytest.raises(CircuitBreakerOpenError, match="is OPEN"):
            breaker.call(func=failing_func)

        assert breaker.stats.rejected_calls == 1
        assert breaker.stats.failed_calls == 2  # Still 2, not 3

    def test_statistics_tracking(self) -> None:
        """Circuit breaker should track call statistics."""
        breaker = CircuitBreaker(name="test")

        def success_func() -> str:
            return "success"

        def failing_func() -> None:
            raise ModbusTimeoutError(message="Timeout")

        # Mix of success and failure
        breaker.call(func=success_func)
        breaker.call(func=success_func)
        with pytest.raises(ModbusTimeoutError):
            breaker.call(func=failing_func)

        stats = breaker.get_stats()
        assert stats.total_calls == 3
        assert stats.successful_calls == 2
        assert stats.failed_calls == 1
        assert stats.last_failure_time is not None

    def test_successful_calls_keep_circuit_closed(self) -> None:
        """Successful calls should keep circuit closed."""
        breaker = CircuitBreaker(name="test")

        def success_func() -> str:
            return "success"

        # Multiple successful calls
        for _ in range(10):
            result = breaker.call(func=success_func)
            assert result == "success"

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.successful_calls == 10
        assert breaker.stats.failed_calls == 0


class TestConnectionStateMachine:
    """Tests for connection state machine."""

    def test_connected_resets_attempts(self) -> None:
        """Successful connection should reset attempt counter."""
        sm = ConnectionStateMachine(name="test")

        sm.transition_to(new_state=ConnectionState.CONNECTING)
        sm.transition_to(new_state=ConnectionState.RECONNECTING)
        assert sm.info.reconnect_attempts == 1

        sm.transition_to(new_state=ConnectionState.CONNECTED)
        assert sm.info.reconnect_attempts == 0

    def test_error_message_stored(self) -> None:
        """Error state should store error message."""
        sm = ConnectionStateMachine(name="test")

        error_msg = "Connection refused"
        sm.transition_to(new_state=ConnectionState.ERROR, error=error_msg)

        assert sm.state == ConnectionState.ERROR
        assert sm.info.last_error == error_msg

    def test_invalid_state_transitions_ignored(self) -> None:
        """Invalid state transitions should be ignored."""
        sm = ConnectionStateMachine(name="test")

        # Try invalid transition: DISCONNECTED -> CONNECTED (must go through CONNECTING)
        sm.transition_to(new_state=ConnectionState.CONNECTED)

        # Should still be DISCONNECTED
        assert sm.state == ConnectionState.DISCONNECTED

    def test_reconnecting_increments_attempts(self) -> None:
        """Entering RECONNECTING should increment attempt counter."""
        sm = ConnectionStateMachine(name="test")

        sm.transition_to(new_state=ConnectionState.CONNECTING)
        sm.transition_to(new_state=ConnectionState.CONNECTED)
        sm.transition_to(new_state=ConnectionState.RECONNECTING)

        assert sm.info.reconnect_attempts == 1

        sm.transition_to(new_state=ConnectionState.RECONNECTING)
        assert sm.info.reconnect_attempts == 2

    def test_reset_to_disconnected(self) -> None:
        """Reset should return to DISCONNECTED state."""
        sm = ConnectionStateMachine(name="test")

        sm.transition_to(new_state=ConnectionState.CONNECTING)
        sm.transition_to(new_state=ConnectionState.CONNECTED)

        sm.reset()

        assert sm.state == ConnectionState.DISCONNECTED
        assert sm.info.reconnect_attempts == 0

    def test_starts_disconnected(self) -> None:
        """State machine should start in DISCONNECTED state."""
        sm = ConnectionStateMachine(name="test")
        assert sm.state == ConnectionState.DISCONNECTED
        assert sm.is_disconnected
        assert not sm.is_connected

    def test_state_change_callbacks(self) -> None:
        """State change callbacks should be called."""
        sm = ConnectionStateMachine(name="test")

        transitions: list[tuple[ConnectionState, ConnectionState]] = []

        def callback(old: ConnectionState, new: ConnectionState) -> None:
            transitions.append((old, new))

        sm.on_state_change(callback=callback)

        # Trigger some transitions
        sm.transition_to(new_state=ConnectionState.CONNECTING)
        sm.transition_to(new_state=ConnectionState.CONNECTED)

        assert len(transitions) == 2
        assert transitions[0] == (ConnectionState.DISCONNECTED, ConnectionState.CONNECTING)
        assert transitions[1] == (ConnectionState.CONNECTING, ConnectionState.CONNECTED)

    def test_timestamps_updated(self) -> None:
        """State machine should track connection timestamps."""
        sm = ConnectionStateMachine(name="test")

        # Connect
        sm.transition_to(new_state=ConnectionState.CONNECTING)
        sm.transition_to(new_state=ConnectionState.CONNECTED)

        assert sm.info.connected_at is not None
        assert sm.info.connected_at <= datetime.now(UTC)

        # Disconnect
        sm.transition_to(new_state=ConnectionState.DISCONNECTED)

        assert sm.info.disconnected_at is not None
        assert sm.info.disconnected_at >= sm.info.connected_at

    def test_valid_state_transitions(self) -> None:
        """Valid state transitions should work."""
        sm = ConnectionStateMachine(name="test")

        # DISCONNECTED -> CONNECTING
        sm.transition_to(new_state=ConnectionState.CONNECTING)
        assert sm.state == ConnectionState.CONNECTING

        # CONNECTING -> CONNECTED
        sm.transition_to(new_state=ConnectionState.CONNECTED)
        assert sm.state == ConnectionState.CONNECTED
        assert sm.is_connected

        # CONNECTED -> DISCONNECTED
        sm.transition_to(new_state=ConnectionState.DISCONNECTED)
        assert sm.state == ConnectionState.DISCONNECTED
