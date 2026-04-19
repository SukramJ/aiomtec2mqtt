# ADR-005: No separate `aiomtec2mqtt_storage` package

- **Status**: Accepted
- **Date**: 2026-04-19
- **Context**: ROADMAP 1.8 — "`aiomtec2mqtt_storage` as a separate package
  for caching/persistence (if needed)"

## Context

`aiohomematic` keeps `hahomematic_support` and a separate caching/storage
layer because it owns long-lived device state — paramset descriptions, device
descriptions, link metadata — that is expensive to fetch over XML-RPC and
must survive process restarts. Pulling that into its own package lets the
core stay slim and lets test code reuse the same storage primitives.

`aiomtec2mqtt` does not have a comparable need:

- The bridge polls Modbus on a fixed schedule and pushes outbound MQTT
  messages. It owns no durable state of its own.
- Register definitions are static (`registers.yaml`), loaded at startup,
  and addressed by ID — no caching layer earns its keep.
- Home Assistant's MQTT discovery is the system of record for the entity
  graph. The bridge re-publishes discovery on every connect; there is
  nothing to persist between restarts.
- Daily/total energy counters live on the inverter itself. The bridge
  reads them, it does not aggregate them.
- The replay framework added in 3.10 (`session_replay.py`) covers the one
  legitimate persistence concern — capturing real Modbus traces for
  offline reproduction — without needing a separate distribution.

## Decision

**Do not** create an `aiomtec2mqtt_storage` package. Treat 1.8 as
deliberately not-needed and revisit only if the triggers below appear.

## Rationale

- **No state to store**: the bridge is a stateless pump. A storage
  package without consumers is dead weight on the release pipeline (one
  more version to bump, one more wheel to publish, one more PyPI project
  to maintain).
- **Splitting has a cost**: the dual-release plumbing for
  `aiomtec2mqtt-test-support` (3.12 + 7.5) was justified because tests
  in _downstream_ projects need those fixtures. A storage package would
  have no downstream consumer today.
- **Replay covers the recorded-state use case**: `SessionRecorder` /
  `SessionPlayer` already give us deterministic playback against real
  inverter traces. That is the only persistence concern the project has
  surfaced in practice.
- **YAGNI**: the roadmap entry itself hedged with "(if needed)". It is
  not needed.

## Constraints

Revisit this decision if any of the following becomes true:

- The bridge starts owning state that must survive restarts
  (e.g. cross-boot energy aggregation, write-rate limiting with a
  persistent budget, MQTT message dedup across reconnects).
- A second downstream tool wants to reuse a caching/storage primitive
  that we develop in-process — at that point a `support`-style
  distribution may be cheaper than copy-paste.
- We adopt a long-running queue/buffer (e.g. for offline MQTT broker
  outages) that benefits from durable storage. Today the resilience
  layer drops on the floor and reconnects; if that changes, storage
  follows.

## Alternatives considered

- **Build the package speculatively**: rejected — adds release/CI
  surface for zero current consumers; YAGNI.
- **Inline a `storage/` sub-package without splitting the distribution**:
  also rejected for now — there is no code to put in it. The 1.4
  sub-package refactor already covers the structural concern; an empty
  `storage/` would just be a placeholder.
