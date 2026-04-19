"""Tests for aiomtec2mqtt.decorators."""

from __future__ import annotations

import asyncio
import logging

import pytest

from aiomtec2mqtt.decorators import callback, measure, retry
from aiomtec2mqtt.exceptions import ModbusConnectionError, ModbusDeviceError


class TestRetrySync:
    def test_custom_exception_types(self) -> None:
        attempts = {"n": 0}

        @retry(
            max_retries=2,
            initial_delay=0.001,
            max_delay=0.002,
            jitter=False,
            exceptions=(ValueError,),
        )
        def op() -> str:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ValueError("nope")
            return "ok"

        assert op() == "ok"
        assert attempts["n"] == 2

    def test_logs_error_on_exhaustion(self, caplog: pytest.LogCaptureFixture) -> None:
        @retry(max_retries=1, initial_delay=0.001, max_delay=0.002, jitter=False)
        def op() -> None:
            raise ModbusConnectionError(message="x")

        with (
            caplog.at_level(logging.ERROR, logger="aiomtec2mqtt.decorators"),
            pytest.raises(ModbusConnectionError),
        ):
            op()
        assert any("exhausted" in rec.message for rec in caplog.records)

    def test_logs_warning_on_retry(self, caplog: pytest.LogCaptureFixture) -> None:
        @retry(max_retries=1, initial_delay=0.001, max_delay=0.002, jitter=False)
        def op() -> int:
            if not getattr(op, "done", False):
                op.done = True  # type: ignore[attr-defined]
                raise ModbusConnectionError(message="x")
            return 7

        with caplog.at_level(logging.WARNING, logger="aiomtec2mqtt.decorators"):
            assert op() == 7
        assert any("attempt 1 failed" in rec.message for rec in caplog.records)

    def test_non_matching_exception_is_not_retried(self) -> None:
        attempts = {"n": 0}

        @retry(max_retries=5, initial_delay=0.001, max_delay=0.002, jitter=False)
        def op() -> None:
            attempts["n"] += 1
            # ModbusDeviceError is not a RetryableException
            raise ModbusDeviceError(message="hard")

        with pytest.raises(ModbusDeviceError):
            op()
        assert attempts["n"] == 1

    def test_reraises_after_exhaustion(self) -> None:
        @retry(max_retries=2, initial_delay=0.001, max_delay=0.002, jitter=False)
        def op() -> None:
            raise ModbusConnectionError(message="boom")

        with pytest.raises(ModbusConnectionError):
            op()

    def test_retries_until_success(self) -> None:
        attempts = {"n": 0}

        @retry(max_retries=3, initial_delay=0.001, max_delay=0.002, jitter=False)
        def op() -> int:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ModbusConnectionError(message="transient")
            return attempts["n"]

        assert op() == 3
        assert attempts["n"] == 3

    def test_returns_value_on_first_success(self) -> None:
        @retry(max_retries=3, initial_delay=0.001, max_delay=0.002, jitter=False)
        def op() -> int:
            return 42

        assert op() == 42


class TestRetryAsync:
    async def test_non_matching_exception_is_not_retried(self) -> None:
        attempts = {"n": 0}

        @retry(max_retries=5, initial_delay=0.001, max_delay=0.002, jitter=False)
        async def op() -> None:
            attempts["n"] += 1
            raise ModbusDeviceError(message="hard")

        with pytest.raises(ModbusDeviceError):
            await op()
        assert attempts["n"] == 1

    async def test_reraises_after_exhaustion(self) -> None:
        @retry(max_retries=2, initial_delay=0.001, max_delay=0.002, jitter=False)
        async def op() -> None:
            raise ModbusConnectionError(message="boom")

        with pytest.raises(ModbusConnectionError):
            await op()

    async def test_retries_until_success(self) -> None:
        attempts = {"n": 0}

        @retry(max_retries=3, initial_delay=0.001, max_delay=0.002, jitter=False)
        async def op() -> int:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ModbusConnectionError(message="transient")
            return attempts["n"]

        assert await op() == 3
        assert attempts["n"] == 3

    async def test_returns_value_on_first_success(self) -> None:
        @retry(max_retries=3, initial_delay=0.001, max_delay=0.002, jitter=False)
        async def op() -> int:
            return 42

        assert await op() == 42

    async def test_uses_asyncio_sleep(self) -> None:
        # Smoke test — if the async variant accidentally fell back to
        # time.sleep, the event loop would block and this test would time
        # out under pytest's default async settings.
        attempts = {"n": 0}

        @retry(max_retries=2, initial_delay=0.001, max_delay=0.002, jitter=False)
        async def op() -> int:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ModbusConnectionError(message="x")
            await asyncio.sleep(0)
            return attempts["n"]

        assert await op() == 2


class TestMeasureSync:
    def test_default_label_uses_qualname(self) -> None:
        events: list[tuple[str, float, bool]] = []

        def rec(label: str, duration: float, success: bool) -> None:
            events.append((label, duration, success))

        @measure(recorder=rec)
        def my_named_function() -> int:
            return 1

        assert my_named_function() == 1
        assert events[0][0].endswith("my_named_function")

    def test_records_duration_on_success(self) -> None:
        events: list[tuple[str, float, bool]] = []

        def rec(label: str, duration: float, success: bool) -> None:
            events.append((label, duration, success))

        @measure(recorder=rec, name="foo")
        def op(x: int) -> int:
            return x * 2

        assert op(3) == 6
        assert len(events) == 1
        label, duration, success = events[0]
        assert label == "foo"
        assert duration >= 0.0
        assert success is True

    def test_records_failure_on_exception(self) -> None:
        events: list[tuple[str, float, bool]] = []

        def rec(label: str, duration: float, success: bool) -> None:
            events.append((label, duration, success))

        @measure(recorder=rec, name="boom")
        def op() -> None:
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            op()
        assert len(events) == 1
        assert events[0][0] == "boom"
        assert events[0][2] is False


class TestMeasureAsync:
    async def test_records_duration_on_success(self) -> None:
        events: list[tuple[str, float, bool]] = []

        def rec(label: str, duration: float, success: bool) -> None:
            events.append((label, duration, success))

        @measure(recorder=rec, name="async_foo")
        async def op(x: int) -> int:
            await asyncio.sleep(0)
            return x * 2

        assert await op(5) == 10
        assert events[0][0] == "async_foo"
        assert events[0][2] is True

    async def test_records_failure_on_exception(self) -> None:
        events: list[tuple[str, float, bool]] = []

        def rec(label: str, duration: float, success: bool) -> None:
            events.append((label, duration, success))

        @measure(recorder=rec, name="async_boom")
        async def op() -> None:
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            await op()
        assert events[0][2] is False


class TestCallbackSync:
    def test_returns_value_on_success(self) -> None:
        @callback
        def handler(x: int) -> int:
            return x + 1

        assert handler(3) == 4

    def test_swallows_exception_and_returns_none(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        @callback
        def handler() -> int:
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger="aiomtec2mqtt.decorators"):
            result = handler()
        assert result is None
        assert any("raised RuntimeError" in rec.message for rec in caplog.records)


class TestCallbackAsync:
    async def test_returns_value_on_success(self) -> None:
        @callback
        async def handler(x: int) -> int:
            await asyncio.sleep(0)
            return x + 1

        assert await handler(3) == 4

    async def test_swallows_exception_and_returns_none(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        @callback
        async def handler() -> int:
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger="aiomtec2mqtt.decorators"):
            result = await handler()
        assert result is None
        assert any("raised RuntimeError" in rec.message for rec in caplog.records)
