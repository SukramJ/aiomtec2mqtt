# ADR-002: Resilience design — Backoff + Circuit Breaker + Connection State Machine

- **Status**: Accepted
- **Date**: 2026-04-17
- **Context**: ROADMAP 6.8 — back-fill ADRs for the load-bearing architecture
  decisions.

## Context

The bridge talks to two unreliable network peers:

- **M-TEC inverter** (Modbus TCP/RTU): firmware bugs, network glitches,
  Wi-Fi drops on the Espressif adapter, power cuts in the breaker box.
- **MQTT broker** (local or remote): restarts, auth changes, network outages.

Without a resilience layer the bridge would either crash-loop on every
transient failure or — worse — hang silently (e.g. TCP handshake without a
timeout).

## Decision

`aiomtec2mqtt/resilience.py` exposes three orthogonal primitives that are
applied wherever I/O hits an external system:

1. **`ExponentialBackoff`** — exponential backoff with jitter, capped at
   `max_delay`, optionally limited by `max_retries`.
2. **`CircuitBreaker`** — Closed → Open → Half-Open transitions based on
   failure rate; prevents hammering an already-failed peer.
3. **`ConnectionStateMachine`** — explicit `CONNECTED`, `DISCONNECTED`,
   `RECONNECTING`, `FAILED` states; observable from the outside
   (Prometheus + `HealthCheck`).

## Rationale

- **Separation of concerns**: backoff decides _when_ to retry, the circuit
  breaker decides _whether_ to try at all, and the state machine makes the
  status observable. Combined they are sturdier than a monolithic
  "reconnect-with-retry".
- **Jitter**: multiple bridges in a single household (e.g. several rooms
  with separate Energybutlers) would reconnect-storm in lockstep without
  jitter as soon as the broker comes back. Jitter spreads the load.
- **Half-open probe**: prevents the circuit breaker from staying open
  forever without burdening the hot path with permanent probes.
- **Hand-rolled instead of `tenacity`/`backoff`**: both libraries are
  excellent, but the codebase is small enough (~300 LoC for all three
  primitives combined) that the extra dependency does not justify the
  audit overhead. The hand-rolled design also gives us
  `CircuitBreakerStats` and `ConnectionStateInfo` that can be wired
  straight into Prometheus.

## Constraints

- **Do not** retry non-idempotent calls in the hot path
  (e.g. `write_registers` for control registers) — only read Modbus calls
  and MQTT publishes go through the backoff.
- Circuit-breaker configuration stays _per component_ (Modbus vs. MQTT),
  not global — Modbus inverters are slower than MQTT brokers and need
  longer `recovery_timeout` values.
- `ConnectionStateMachine` transitions must be logged + exported via
  Prometheus _before_ the actual reconnect, otherwise monitoring only
  sees reconnect storms after they are already over.

## Alternatives rejected

- **`tenacity` instead of in-house**: extra dependency without a clear
  win, and bespoke stats hooks would still be required.
- **Backoff only, no circuit breaker**: during long outages this would
  produce a flood of log spam and CPU load with no chance of a successful
  connect.
- **Single global circuit breaker for all peers**: would conflate MQTT and
  Modbus failures — a Modbus outage would block MQTT publishes even
  though the broker is reachable.

## Consequences

- `HealthCheck.report_*` must fire on every successful I/O call,
  otherwise the `stale` status never resets.
- Tests must skip backoff delays via `time.monotonic` patching or
  `freezegun` — see `tests/test_resilience.py`.
- New I/O paths (e.g. a future HTTP integration) must reuse the existing
  primitives instead of bringing their own retry loops.
