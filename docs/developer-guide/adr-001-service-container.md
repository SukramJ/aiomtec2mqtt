# ADR-001: Keep `ServiceContainer` alongside constructor injection

- **Status**: Accepted
- **Date**: 2026-04-17
- **Context**: ROADMAP 1.7 — "Evaluate `ServiceContainer` against plain factory
  functions — simplify if overhead exceeds benefit"

## Context

`aiomtec2mqtt/container.py` ships a small, hand-rolled DI container
(`ServiceContainer`) with singleton and factory registrations. The production
coordinator (`AsyncMtecCoordinator`) does **not** use it — it wires its
dependencies directly via constructor injection. The container is consumed by:

- `aiomtec2mqtt/testing.py::create_test_container` — convenience for tests
  that want fakes wired together.
- Future integrations (HA add-on, custom deployments) that may want to swap
  real clients for fakes at runtime.

The question: remove the container (reduce surface) or keep it?

## Decision

**Keep** `ServiceContainer`. Document its scope so it does not sprawl.

## Rationale

- **Low overhead**: ~200 LoC, fully tested. Removing it saves little.
- **Test ergonomics**: `create_test_container` composes fakes in one call —
  rewriting each test to assemble fakes inline would be churn.
- **Integration ergonomics**: downstream users (HA add-on, custom wiring) get
  a single, discoverable hook instead of monkey-patching imports.
- **Consistency with `aiohomematic`**: contributors familiar with that
  codebase find a matching pattern.

## Constraints

- Do **not** grow the container into a full IoC framework
  (no auto-wiring, no scopes beyond singleton/factory).
- New components should expose plain constructor injection as the primary
  API; container registration is optional syntactic sugar.
- The coordinator must remain usable **without** the container for the
  simplest deployment path.

## Alternatives considered

- **Remove container + plain factories**: Cleaner, but forces every test to
  repeat the same fake-assembly code.
- **Adopt `dependency-injector` / `injector`**: Too heavy for a single-box
  bridge; adds one more package to audit.
