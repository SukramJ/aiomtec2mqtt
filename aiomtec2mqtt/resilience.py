"""
Resilience patterns for aiomtec2mqtt.

This module provides reliability patterns including:
- Exponential backoff with jitter
- Circuit breaker pattern
- Connection state machine

(c) 2026 by SukramJ
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import logging
import random
from typing import Any, Final

from aiomtec2mqtt.exceptions import RetryableException

_LOGGER: Final = logging.getLogger(__name__)


# ============================================================================
# Exponential Backoff
# ============================================================================


@dataclass
class BackoffConfig:
    """
    Configuration for exponential backoff.

    Attributes:
        initial_delay: Initial delay in seconds (default 1.0)
        max_delay: Maximum delay in seconds (default 60.0)
        multiplier: Multiplier for each retry (default 2.0)
        jitter: Add random jitter to delays (default True)
        max_retries: Maximum number of retries, None = unlimited (default 5)

    """

    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True
    max_retries: int | None = 5


class ExponentialBackoff:
    """
    Exponential backoff with jitter for retry logic.

    This class implements exponential backoff with optional jitter to prevent
    thundering herd problems. Each retry doubles the delay (configurable) up
    to a maximum delay.

    Example:
        backoff = ExponentialBackoff()
        for attempt in range(5):
            try:
                result = risky_operation()
                break
            except RetryableException:
                delay = backoff.next_delay()
                await asyncio.sleep(delay)

    """

    def __init__(self, *, config: BackoffConfig | None = None) -> None:
        """
        Initialize exponential backoff.

        Args:
            config: Backoff configuration, uses defaults if not provided

        """
        self.config = config or BackoffConfig()
        self._attempt = 0

    @property
    def attempt(self) -> int:
        """Current attempt number (0-indexed)."""
        return self._attempt

    def next_delay(self) -> float:
        """
        Calculate next delay with exponential backoff and jitter.

        Returns:
            Delay in seconds for the next retry

        Raises:
            RuntimeError: If max_retries is exceeded

        """
        if self.config.max_retries is not None and self._attempt >= self.config.max_retries:
            msg = f"Max retries ({self.config.max_retries}) exceeded"
            raise RuntimeError(msg)

        # Calculate exponential delay
        delay = min(
            self.config.initial_delay * (self.config.multiplier**self._attempt),
            self.config.max_delay,
        )

        # Add jitter if enabled (±25% random variation)
        if self.config.jitter:
            jitter_range = delay * 0.25
            delay = delay + random.uniform(-jitter_range, jitter_range)  # noqa: S311
            delay = max(0.1, delay)  # Ensure minimum 100ms

        self._attempt += 1
        return delay

    def reset(self) -> None:
        """Reset backoff state (call after successful operation)."""
        self._attempt = 0


# ============================================================================
# Circuit Breaker
# ============================================================================


class CircuitState(Enum):
    """
    Circuit breaker states.

    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing fast, requests rejected immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening circuit (default 5)
        success_threshold: Number of successes in HALF_OPEN to close circuit (default 2)
        timeout: Seconds to wait before entering HALF_OPEN from OPEN (default 60.0)
        half_open_max_calls: Max concurrent calls allowed in HALF_OPEN (default 1)

    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    half_open_max_calls: int = 1


@dataclass
class CircuitBreakerStats:
    """
    Circuit breaker statistics.

    Attributes:
        total_calls: Total number of calls
        successful_calls: Number of successful calls
        failed_calls: Number of failed calls
        rejected_calls: Number of calls rejected by open circuit
        last_failure_time: Timestamp of last failure
        last_state_change: Timestamp of last state change

    """

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: datetime | None = None
    last_state_change: datetime = field(default_factory=lambda: datetime.now(UTC))


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    The circuit breaker prevents cascading failures by failing fast when a
    service is experiencing issues. It has three states:

    - CLOSED (normal): All requests pass through
    - OPEN (failing): Requests fail immediately without calling service
    - HALF_OPEN (testing): Limited requests allowed to test recovery

    State Transitions:
    - CLOSED -> OPEN: After failure_threshold consecutive failures
    - OPEN -> HALF_OPEN: After timeout seconds
    - HALF_OPEN -> CLOSED: After success_threshold consecutive successes
    - HALF_OPEN -> OPEN: On any failure

    Example:
        breaker = CircuitBreaker(name="modbus")

        try:
            breaker.call(modbus_client.read_register, address=11000)
        except CircuitBreakerOpenError:
            logger.error("Circuit breaker open, service unavailable")

    """

    def __init__(
        self,
        *,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            name: Name for logging and identification
            config: Circuit breaker configuration

        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._opened_at: datetime | None = None
        self._half_open_calls = 0
        self.stats = CircuitBreakerStats()

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN

    @property
    def state(self) -> CircuitState:
        """Current circuit breaker state."""
        self._update_state()
        return self._state

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function protected by circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result of function call

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception raised by func

        """
        self.stats.total_calls += 1

        # Check if circuit is open
        current_state = self.state
        if current_state == CircuitState.OPEN:
            self.stats.rejected_calls += 1
            msg = f"Circuit breaker '{self.name}' is OPEN"
            raise CircuitBreakerOpenError(msg)

        # Check half-open call limit
        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                self.stats.rejected_calls += 1
                msg = f"Circuit breaker '{self.name}' HALF_OPEN call limit reached"
                raise CircuitBreakerOpenError(msg)
            self._half_open_calls += 1

        # Execute function
        try:
            result = func(*args, **kwargs)
        except Exception as ex:
            self._on_failure(exception=ex)
            raise
        else:
            self._on_success()
            return result

    def get_stats(self) -> CircuitBreakerStats:
        """
        Get circuit breaker statistics.

        Returns:
            Current statistics

        """
        return self.stats

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        _LOGGER.info("Circuit breaker '%s' manually reset", self.name)
        self._transition_to(new_state=CircuitState.CLOSED)

    def _on_failure(self, *, exception: Exception) -> None:
        """Handle failed call."""
        self.stats.failed_calls += 1
        self.stats.last_failure_time = datetime.now(UTC)
        self._last_failure_time = self.stats.last_failure_time

        # Only count retryable exceptions for circuit breaker logic
        if not isinstance(exception, RetryableException):
            _LOGGER.debug(
                "Circuit breaker '%s' ignoring non-retryable exception: %s",
                self.name,
                type(exception).__name__,
            )
            return

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in HALF_OPEN immediately reopens circuit
            self._transition_to(new_state=CircuitState.OPEN)
            _LOGGER.warning(
                "Circuit breaker '%s' reopened after failure in HALF_OPEN: %s",
                self.name,
                exception,
            )
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to(new_state=CircuitState.OPEN)
                _LOGGER.error(
                    "Circuit breaker '%s' opened after %d failures: %s",
                    self.name,
                    self._failure_count,
                    exception,
                )

    def _on_success(self) -> None:
        """Handle successful call."""
        self.stats.successful_calls += 1

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to(new_state=CircuitState.CLOSED)
                _LOGGER.info(
                    "Circuit breaker '%s' closed after %d successful tests",
                    self.name,
                    self._success_count,
                )
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success in CLOSED state
            self._failure_count = 0

    def _transition_to(self, *, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self.stats.last_state_change = datetime.now(UTC)

        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.now(UTC)
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0

        _LOGGER.info(
            "Circuit breaker '%s' transitioned: %s -> %s",
            self.name,
            old_state.value,
            new_state.value,
        )

    def _update_state(self) -> None:
        """Update circuit state based on timeout and current conditions."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            elapsed = (datetime.now(UTC) - self._opened_at).total_seconds()
            if elapsed >= self.config.timeout:
                self._transition_to(new_state=CircuitState.HALF_OPEN)
                _LOGGER.info(
                    "Circuit breaker '%s' entering HALF_OPEN state after %.1fs timeout",
                    self.name,
                    elapsed,
                )


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected."""


# ============================================================================
# Connection State Machine
# ============================================================================


class ConnectionState(Enum):
    """
    Connection states for state machine.

    - DISCONNECTED: Not connected
    - CONNECTING: Connection attempt in progress
    - CONNECTED: Successfully connected
    - RECONNECTING: Attempting to reconnect after failure
    - ERROR: Permanent error, no reconnection attempts
    """

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class ConnectionStateInfo:
    """
    Information about current connection state.

    Attributes:
        state: Current state
        connected_at: When connection was established (if CONNECTED)
        disconnected_at: When connection was lost (if DISCONNECTED/ERROR)
        reconnect_attempts: Number of reconnection attempts
        last_error: Last error message (if any)

    """

    state: ConnectionState
    connected_at: datetime | None = None
    disconnected_at: datetime | None = None
    reconnect_attempts: int = 0
    last_error: str | None = None


class ConnectionStateMachine:
    """
    State machine for connection lifecycle management.

    This class manages connection state transitions and provides hooks for
    state change notifications. It helps track connection status and
    reconnection attempts.

    Example:
        state_machine = ConnectionStateMachine(name="modbus")

        # Register callback for state changes
        state_machine.on_state_change(lambda old, new: print(f"{old} -> {new}"))

        # Transition states
        state_machine.transition_to(ConnectionState.CONNECTING)
        state_machine.transition_to(ConnectionState.CONNECTED)

    """

    def __init__(self, *, name: str) -> None:
        """
        Initialize connection state machine.

        Args:
            name: Name for logging and identification

        """
        self.name = name
        self._state = ConnectionState.DISCONNECTED
        self._info = ConnectionStateInfo(state=ConnectionState.DISCONNECTED)
        self._callbacks: list[Callable[[ConnectionState, ConnectionState], None]] = []
        _LOGGER.debug("Connection state machine '%s' initialized", self.name)

    @property
    def info(self) -> ConnectionStateInfo:
        """Get detailed state information."""
        return self._info

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def is_disconnected(self) -> bool:
        """Check if currently disconnected."""
        return self._state == ConnectionState.DISCONNECTED

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    def on_state_change(
        self,
        *,
        callback: Callable[[ConnectionState, ConnectionState], None],
    ) -> None:
        """
        Register callback for state changes.

        Args:
            callback: Function called with (old_state, new_state) on transitions

        """
        self._callbacks.append(callback)

    def reset(self) -> None:
        """Reset state machine to DISCONNECTED."""
        self.transition_to(new_state=ConnectionState.DISCONNECTED)
        self._info = ConnectionStateInfo(state=ConnectionState.DISCONNECTED)

    def transition_to(
        self,
        *,
        new_state: ConnectionState,
        error: str | None = None,
    ) -> None:
        """
        Transition to a new state.

        Args:
            new_state: Target state
            error: Optional error message (for ERROR state)

        """
        old_state = self._state

        # Validate state transition
        if not self._is_valid_transition(old_state=old_state, new_state=new_state):
            _LOGGER.warning(
                "Connection state machine '%s': Invalid transition %s -> %s",
                self.name,
                old_state.value,
                new_state.value,
            )
            return

        # Update state
        self._state = new_state
        self._info.state = new_state

        # Update state-specific information
        now = datetime.now(UTC)
        if new_state == ConnectionState.CONNECTED:
            self._info.connected_at = now
            self._info.reconnect_attempts = 0
            self._info.last_error = None
        elif new_state in (ConnectionState.DISCONNECTED, ConnectionState.ERROR):
            self._info.disconnected_at = now
            if error:
                self._info.last_error = error
        elif new_state == ConnectionState.RECONNECTING:
            self._info.reconnect_attempts += 1

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(old_state, new_state)
            except Exception as ex:  # noqa: BLE001
                _LOGGER.error(
                    "Connection state machine '%s': Callback error: %s",
                    self.name,
                    ex,
                )

        _LOGGER.info(
            "Connection state machine '%s': %s -> %s%s",
            self.name,
            old_state.value,
            new_state.value,
            f" (error: {error})" if error else "",
        )

    def _is_valid_transition(
        self,
        *,
        old_state: ConnectionState,
        new_state: ConnectionState,
    ) -> bool:
        """
        Check if state transition is valid.

        Valid transitions:
        - DISCONNECTED -> CONNECTING, ERROR
        - CONNECTING -> CONNECTED, DISCONNECTED, RECONNECTING, ERROR
        - CONNECTED -> DISCONNECTED, RECONNECTING, ERROR
        - RECONNECTING -> CONNECTED, DISCONNECTED, ERROR
        - ERROR -> DISCONNECTED (recovery attempt)

        Args:
            old_state: Current state
            new_state: Target state

        Returns:
            True if transition is valid

        """
        # Allow staying in same state
        if old_state == new_state:
            return True

        valid_transitions = {
            ConnectionState.DISCONNECTED: {
                ConnectionState.CONNECTING,
                ConnectionState.ERROR,
            },
            ConnectionState.CONNECTING: {
                ConnectionState.CONNECTED,
                ConnectionState.DISCONNECTED,
                ConnectionState.RECONNECTING,
                ConnectionState.ERROR,
            },
            ConnectionState.CONNECTED: {
                ConnectionState.DISCONNECTED,
                ConnectionState.RECONNECTING,
                ConnectionState.ERROR,
            },
            ConnectionState.RECONNECTING: {
                ConnectionState.CONNECTED,
                ConnectionState.DISCONNECTED,
                ConnectionState.ERROR,
            },
            ConnectionState.ERROR: {
                ConnectionState.DISCONNECTED,  # Allow recovery attempt
            },
        }

        return new_state in valid_transitions.get(old_state, set())
