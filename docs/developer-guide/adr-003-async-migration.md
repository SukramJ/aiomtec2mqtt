# ADR-003: Migration to `asyncio` with `TaskGroup` as orchestrator

- **Status**: Accepted
- **Date**: 2026-04-17
- **Context**: ROADMAP 6.8 — document the async migration retroactively.

## Context

The original codebase (`MTECmqtt` by croedel) used a synchronous main loop
with `time.sleep`-based refresh intervals. Modbus polling, MQTT publish,
and health tracking ran sequentially in a single thread. Consequences:

- A slow Modbus read blocked all other refreshes.
- MQTT reconnects waited for the next polling tick instead of reacting
  immediately.
- Tests had to mock real `sleep` calls or rely on `freezegun`.
- Polling several inverters in parallel was not practical.

## Decision

Full migration to `asyncio` (Python 3.13+) with `asyncio.TaskGroup`
(PEP 654) as the central orchestrator inside `AsyncMtecCoordinator.run()`.
Every refresh stream (`now`, `day`, `total`, `config`, `static`, health,
backup) runs as its own task inside the task group.

Accompanying replacements:

- `pymodbus.client.AsyncModbusTcpClient` (instead of the sync client with
  a thread pool).
- `aiomqtt` (instead of `paho-mqtt` with callback threads).
- Signal handlers are registered via `loop.add_signal_handler`, not
  `signal.signal` — clean shutdown in an async context.

For consumers that must stay sync (legacy scripts, possibly future HA
add-ons with sync hooks) there is `sync_coordinator_wrapper.py`, which
runs an event loop in a dedicated thread. The wrapper is `@deprecated`-
flagged and will be removed with the next major release.

## Rationale

- **`TaskGroup` instead of `asyncio.gather`**: TaskGroup propagates
  exceptions in a structured way (`ExceptionGroup`), cancels sibling
  tasks correctly, and allows deterministic cleanup. `gather` with
  `return_exceptions=True` would swallow errors.
- **One TaskGroup per coordinator instance**: no global loop state, easy
  to test, every coordinator instance owns its lifecycle.
- **Backward-compat wrapper**: transitional only. The strategic path is
  "everything async". The sync wrapper carries a deprecation warning
  (see ROADMAP 1.1).
- **Python 3.13+ as the floor**: `TaskGroup` plus `asyncio.timeout()`
  context manager only really come together from 3.11, and 3.13 adds
  `asyncio.runners` improvements and `taskgroup` performance gains.

## Constraints

- Inside coroutines, **never** call `time.sleep` — always use
  `asyncio.sleep`.
- Long CPU operations (formula evaluator, JSON serialization of large
  topics) stay synchronous because they are sub-millisecond. If ever
  needed: `asyncio.to_thread`, **not** `loop.run_in_executor` with a
  custom pool.
- No global `asyncio.Queue`s shared between coordinator instances —
  tests would interfere with each other.
- Tests use `pytest-asyncio` with `asyncio_mode = "auto"` (see
  `pyproject.toml`).

## Alternatives rejected

- **`anyio` instead of bare `asyncio`**: would unlock Trio
  compatibility, but every dependency (`pymodbus`, `aiomqtt`) is
  asyncio-only. Extra abstraction without a payoff.
- **Threads instead of async**: simpler to port, but a thread per
  inverter + a thread per MQTT subscription scales poorly and
  cancellation gets fuzzy.
- **Stay sync + `concurrent.futures`**: just shifts the problem — pool
  exhaustion instead of loop blocking.

## Consequences

- Polling intervals are honoured even when one Modbus read takes 5 s —
  other streams keep running undisturbed.
- Shutdown is deterministic: `KeyboardInterrupt` → TaskGroup cancels all
  children → `aclose()` hooks of the clients run → process exits without
  a hanging thread.
- The test suite is now async; tests that had to remain sync run via
  `asyncio.run()` in a helper.
- New code must be async. Sync PRs are not accepted, except for pure CPU
  helpers without I/O.
