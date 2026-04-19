# Developer Guide

This guide is for contributors who want to work on `aiomtec2mqtt` itself —
fixing bugs, adding registers, integrating new inverters, or hardening the
CI/CD pipeline.

1. [Setup](setup.md) — clone the repo, install dev deps, run prek hooks.
2. [Architecture](architecture.md) — module layout and control flow.
3. [Testing](testing.md) — test layout, coverage, markers, contract tests.
4. [Release Process](release-process.md) — how tags and changelogs are produced.

## Architecture Decision Records (ADRs)

- [ADR-001 — Keep `ServiceContainer`](adr-001-service-container.md)
- [ADR-002 — Resilience design (backoff + circuit breaker)](adr-002-resilience-design.md)
- [ADR-003 — Async migration to `asyncio.TaskGroup`](adr-003-async-migration.md)
- [ADR-004 — Builder pattern for `CoordinatorConfig`](adr-004-coordinator-config-builder.md)
- [ADR-005 — No separate storage package](adr-005-no-storage-package.md)
- [ADR-006 — Defer Python 3.14 floor](adr-006-defer-python-314-floor.md)

## Quality history

- [Quality Roadmap (history)](quality-roadmap.md) — audit trail of how
  the project reached parity with `aiohomematic` (68 / 68 closed,
  2026-04-19). Open new realignment items at the bottom of the relevant
  section.
