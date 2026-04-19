"""
Reusable decorators for retry, measurement and callback plumbing.

The decorators here are deliberately thin wrappers around the primitives
defined in `resilience.py` and `prometheus_metrics.py`. They exist so call
sites can opt into a cross-cutting behaviour with a single line instead of
repeating try/except/time blocks. Both sync and async callables are
supported; the correct variant is selected from the wrapped function.

(c) 2026 by SukramJ
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import wraps
import inspect
import logging
import time
from typing import ParamSpec, TypeVar, cast

from aiomtec2mqtt.exceptions import RetryableException
from aiomtec2mqtt.resilience import BackoffConfig, ExponentialBackoff

__all__ = ["callback", "measure", "retry"]

_LOGGER = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

# Signature for metric sinks used by @measure. The recorder receives the
# metric label, the elapsed wall-clock time in seconds, and a success flag.
MeasureRecorder = Callable[[str, float, bool], None]


def retry(
    *,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    multiplier: float = 2.0,
    jitter: bool = True,
    exceptions: type[BaseException] | tuple[type[BaseException], ...] = RetryableException,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Retry the decorated callable with exponential backoff.

    Works for both sync and async callables. Only exceptions matching
    `exceptions` trigger a retry; everything else propagates immediately.
    After `max_retries` failures the last exception is re-raised.

    Args:
        max_retries: Maximum number of additional attempts after the first call.
        initial_delay: Delay before the first retry in seconds.
        max_delay: Upper bound for the computed delay in seconds.
        multiplier: Growth factor applied between retries.
        jitter: If True, apply ±25 % random jitter to each delay.
        exceptions: Exception type(s) that should trigger a retry.

    Returns:
        A decorator that preserves the wrapped callable's signature.

    """
    # `max_retries=None` on the config keeps `ExponentialBackoff` from
    # raising its own RuntimeError. The decorator does the attempt
    # accounting itself so the original exception is propagated.
    config = BackoffConfig(
        initial_delay=initial_delay,
        max_delay=max_delay,
        multiplier=multiplier,
        jitter=jitter,
        max_retries=None,
    )

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                backoff = ExponentialBackoff(config=config)
                attempt = 0
                while True:
                    try:
                        return cast("_R", await func(*args, **kwargs))
                    except exceptions as ex:
                        attempt += 1
                        if attempt > max_retries:
                            _LOGGER.error(
                                "retry: %s exhausted %d attempts: %s",
                                func.__qualname__,
                                attempt,
                                ex,
                            )
                            raise
                        delay = backoff.next_delay()
                        _LOGGER.warning(
                            "retry: %s attempt %d failed (%s); sleeping %.2fs",
                            func.__qualname__,
                            attempt,
                            type(ex).__name__,
                            delay,
                        )
                        await asyncio.sleep(delay)

            return cast("Callable[_P, _R]", async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            backoff = ExponentialBackoff(config=config)
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as ex:
                    attempt += 1
                    if attempt > max_retries:
                        _LOGGER.error(
                            "retry: %s exhausted %d attempts: %s",
                            func.__qualname__,
                            attempt,
                            ex,
                        )
                        raise
                    delay = backoff.next_delay()
                    _LOGGER.warning(
                        "retry: %s attempt %d failed (%s); sleeping %.2fs",
                        func.__qualname__,
                        attempt,
                        type(ex).__name__,
                        delay,
                    )
                    time.sleep(delay)

        return sync_wrapper

    return decorator


def measure(
    *,
    recorder: MeasureRecorder,
    name: str | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Record the wall-clock duration of a call via `recorder`.

    Works for both sync and async callables. The recorder is invoked
    exactly once per call, in a `finally` clause, regardless of whether
    the wrapped function raised. On exception the `success` flag is False
    and the original exception propagates.

    Args:
        recorder: Callable that receives `(label, duration_seconds, success)`.
        name: Metric label to pass to the recorder. Defaults to the wrapped
            function's qualified name.

    Returns:
        A decorator that preserves the wrapped callable's signature.

    """

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        label = name or func.__qualname__

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                start = time.perf_counter()
                success = True
                try:
                    return cast("_R", await func(*args, **kwargs))
                except Exception:
                    success = False
                    raise
                finally:
                    recorder(label, time.perf_counter() - start, success)

            return cast("Callable[_P, _R]", async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            start = time.perf_counter()
            success = True
            try:
                return func(*args, **kwargs)
            except Exception:
                success = False
                raise
            finally:
                recorder(label, time.perf_counter() - start, success)

        return sync_wrapper

    return decorator


def callback(func: Callable[_P, _R]) -> Callable[_P, _R | None]:  # noqa: UP047  # kwonly: disable
    """
    Mark a function as a callback handler.

    Exceptions raised inside the wrapped function are logged at ERROR level
    and swallowed. The decorated callable returns `None` on failure so a
    faulty observer cannot disrupt the dispatcher that invoked it.

    Args:
        func: The callable to protect.

    Returns:
        A wrapper returning `R | None` (sync or async mirroring `func`).

    """
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R | None:
            try:
                return cast("_R", await func(*args, **kwargs))
            except Exception as ex:  # noqa: BLE001
                _LOGGER.error(
                    "callback %s raised %s: %s",
                    func.__qualname__,
                    type(ex).__name__,
                    ex,
                )
                return None

        return cast("Callable[_P, _R | None]", async_wrapper)

    @wraps(func)
    def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R | None:
        try:
            return func(*args, **kwargs)
        except Exception as ex:  # noqa: BLE001
            _LOGGER.error(
                "callback %s raised %s: %s",
                func.__qualname__,
                type(ex).__name__,
                ex,
            )
            return None

    return sync_wrapper
