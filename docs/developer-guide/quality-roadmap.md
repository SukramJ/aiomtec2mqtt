# Quality Roadmap — aiomtec2mqtt

**Status (2026-04-19): baseline reached — 68 / 68 closed.** This document
is no longer an open work-tracker; it is the audit trail of how the
project arrived at parity with [`aiohomematic`](https://github.com/SukramJ/aiohomematic)
across architecture, tests, CI/CD, security, documentation, and release.

**How to use this document going forward**:

- **Read it as history**: every closed line carries the date and any
  notable context. Several ADRs (`adr-001`, `adr-005`, `adr-006`)
  reference items by number — those references stay stable.
- **Keep it as the realignment anchor**: the benchmark project keeps
  moving. The "Review cadence" block below schedules the next
  realignment pass; new items get appended at the bottom of the
  relevant section with a fresh date and re-open the area's score.
- **Do not delete closed lines**: the History table at the end
  summarises by date, but the per-line `— YYYY-MM-DD (note)` is the
  only record of _why_ a particular item was closed the way it was.

**Legend**: `P0` = quick win (days) · `P1` = medium (weeks) · `P2` =
large (months) · `[ ]` open · `[x]` done · `[~]` in progress.

---

## Overview — progress per theme area

| Area                      | P0          | P1          | P2          | Total       |
| ------------------------- | ----------- | ----------- | ----------- | ----------- |
| 1. Architecture           | 3 / 3       | 4 / 4       | 2 / 2       | 9 / 9       |
| 2. Code quality & typing  | 4 / 4       | 3 / 3       | 2 / 2       | 9 / 9       |
| 3. Tests                  | 5 / 5       | 4 / 4       | 3 / 3       | 12 / 12     |
| 4. CI/CD & tooling        | 5 / 5       | 4 / 4       | 2 / 2       | 11 / 11     |
| 5. Security               | 4 / 4       | 3 / 3       | 2 / 2       | 9 / 9       |
| 6. Documentation          | 2 / 2       | 5 / 5       | 3 / 3       | 10 / 10     |
| 7. Dependencies & release | 3 / 3       | 3 / 3       | 2 / 2       | 8 / 8       |
| **Sum**                   | **26 / 26** | **26 / 26** | **16 / 16** | **68 / 68** |

---

## 1. Architecture & module structure

### P0 — quick wins

- [x] 1.1 Verify external usage of `sync_coordinator_wrapper.py`; if unused, plan removal — 2026-04-17 (only used internally in tests/docs; deprecation warning added)
- [x] 1.2 Define `__all__` exports across every public module (make the API boundary visible) — 2026-04-17
- [x] 1.3 Add `py.typed` marker file (PEP 561) for downstream mypy users — 2026-04-17 (already present + included in package-data)

### P1 — medium term

- [x] 1.4 Add sub-packages: `client/` (modbus + mqtt), `model/` (register, pydantic), `coordinator/`, `integrations/` (hass, prometheus), `support/` (exceptions, logging, container, resilience) — 2026-04-17 (re-export packages, flat modules kept for backward compat)
- [x] 1.5 `interfaces/` package with `Protocol` classes for `ModbusClientProtocol`, `MqttClientProtocol`, `CoordinatorProtocol` — 2026-04-17 (`aiomtec2mqtt/interfaces/__init__.py` re-exports from `protocols.py`)
- [x] 1.6 Builder pattern for the coordinator config (analogous to `CentralConfigBuilder`) — 2026-04-17 (`aiomtec2mqtt/coordinator_config.py`)
- [x] 1.7 Evaluate `ServiceContainer` against plain factory functions — simplify if overhead exceeds benefit — 2026-04-17 (decision: keep, ADR-001 documents the rationale)

### P2 — long term

- [x] 1.8 `aiomtec2mqtt_storage` as a separate package for caching/persistence (if needed) — 2026-04-19 (deferred — see ADR-005: bridge process is stateless, replay framework already covers recorded-state use case; revisit if cross-restart aggregation/buffering becomes a requirement)
- [x] 1.9 Decorator layer for callback/retry/metrics (analogous to `aiohomematic/decorators/`) — 2026-04-19 (`aiomtec2mqtt/decorators/__init__.py` exposes `@retry`, `@measure`, `@callback` for both sync and async callables, layered on top of `resilience.ExponentialBackoff`; 21 tests in `tests/test_decorators.py` cover backoff exhaustion, exception filtering, recorder semantics, and callback exception swallowing — adoption at call sites is an incremental follow-up, library is ready)

---

## 2. Code quality & typing

### P0 — quick wins

- [x] 2.1 Align ruff rules: add the missing groups from aiohomematic (FLY, INP, PYI, SLOT, TRY) — 2026-04-17 (already enabled)
- [x] 2.2 Document the line-length decision (99 vs 120) in `CLAUDE.md` or `docs/` — 2026-04-17
- [x] 2.3 Consolidate the mypy config from `setup.cfg` into `pyproject.toml` (single source) — 2026-04-17 (mypy.ini → [tool.mypy])
- [x] 2.4 Verify `warn_unreachable = true` and `warn_return_any = true` (aiohomematic has both) — 2026-04-17 (both set)

### P1 — medium term

- [x] 2.5 Enable pylint plugins `pylint_strict_informational` + `pylint_per_file_ignores` — 2026-04-17 (already active in pyproject.toml load-plugins)
- [x] 2.6 Define per-file ignores for tests (`D100–D103`) and scripts (`T20`) — 2026-04-17 (`[tool.ruff.lint.per-file-ignores]`)
- [x] 2.7 Set McCabe complexity to a max of 25 + enforce in CI — 2026-04-17 (`[tool.ruff.lint.mccabe] max-complexity = 25`)

### P2 — long term

- [x] 2.8 Python 3.14 as the floor (once the aiohomematic path is confirmed) — 2026-04-19 (deferred — see ADR-006: aiohomematic trigger met, but the Pi/Synology/HAOS user base tracks Debian, which won't ship 3.14 as system Python before Debian 14 "Forky" ~mid-2027; CI already exercises 3.14 + 3.14t, no 3.14-only feature on the backlog. Revisit triggers + mechanical bump checklist documented in the ADR.)
- [x] 2.9 Prepare free-threaded 3.14t support (orjson fallback etc.) — 2026-04-19 (`aiomtec2mqtt/_json.py` wraps `dumps`/`loads` with a stdlib fallback when orjson is missing — exactly the case on 3.14t today; `hass_int.py` switched from stdlib `json.dumps` to the wrapper so the orjson fast-path lands automatically once a free-threaded wheel ships; CI matrix carries the `3.14t` experimental entry; strategy and exceptions documented in `docs/developer-guide/setup.md`. `session_replay.py` deliberately stays on stdlib `json` because it needs `indent=2`, which orjson does not support.)

---

## 3. Tests

### P0 — quick wins

- [x] 3.1 Set the coverage threshold `fail_under = 85` in `pyproject.toml` and block in CI — 2026-04-17
- [x] 3.2 Adopt `pytest-xdist`: `-n auto --dist loadscope` — 2026-04-17
- [x] 3.3 Enable branch coverage (`branch = true` in the coverage config) — 2026-04-17
- [x] 3.4 Define test markers: `slow`, `benchmark`, `integration` — 2026-04-17
- [x] 3.5 Add `freezegun` for time-dependent tests — 2026-04-17 (already in requirements_test.txt)

### P1 — medium term

- [x] 3.6 Mock Modbus server (analogous to `MockXmlRpcServer`) for integration tests — 2026-04-17 (`aiomtec2mqtt/mock_modbus_server.py` + `tests/test_mock_modbus_server.py`)
- [x] 3.7 Contract tests: `registers.yaml` ↔ `register_models.py` (completeness, types, addresses) — 2026-04-17 (`tests/test_registers_contract.py`)
- [x] 3.8 Teardown hook with global `patch.stopall()` in conftest — 2026-04-17 (autouse fixture in `tests/conftest.py`)
- [x] 3.9 Double the test count (target: ~500 test functions) — focus on `async_coordinator`, `hass_int` — 2026-04-19 (576 tests passing; +45 in `test_async_coordinator_extended.py` covering health-check loop, watchdogs, write queue, HASS discovery/birth, polling cancellation, pseudo-register edges)

### P2 — long term

- [x] 3.10 Replay framework with real Modbus traces (ZIP-based, `SessionPlayer` pattern) — 2026-04-19 (`aiomtec2mqtt/session_replay.py` ships `SessionRecorder` + `SessionPlayer`; ZIP/`manifest.json` format with frame timeline, address-keyed register snapshots; 24 tests in `tests/test_session_replay.py` cover load/advance/loop and end-to-end record→replay)
- [x] 3.11 Benchmarks for polling performance (`pytest-benchmark`) — 2026-04-19 (`tests/test_benchmarks.py` covers `_format_value`, `_process_register_value`, `_convert_code`, JSON dumps/loads; opt-in via `pytest -p no:xdist -m benchmark`)
- [x] 3.12 Separate `aiomtec2mqtt_test_support` package, published to PyPI — 2026-04-19 (`test_support/` subdir with own `pyproject.toml`, src-layout, `aiomtec2mqtt_test_support` package re-exporting `transports`/`fakes`/`fixtures`/`assertions`; pytest plugin auto-registered via `pytest11` entry point; 9 smoke tests; main package excludes `test_support*` from its build) — _unblocks 7.5_

---

## 4. CI/CD & tooling

### P0 — quick wins

- [x] 4.1 Dependabot: weekly instead of monthly, grouping (`pre-commit-hooks`, `test-dependencies`, `prod-dependencies`) — 2026-04-17
- [x] 4.2 Add pre-commit hooks: `bandit`, `codespell`, `yamllint` — 2026-04-17 (already configured)
- [x] 4.3 Replace the `run-in-env.sh` wrapper with `prek` or stock `pre-commit` — 2026-04-17 (local hooks migrated to `language: system`)
- [x] 4.4 Extend the CI matrix to `3.14t` (free-threaded) — initially `continue-on-error` — 2026-04-17
- [x] 4.5 Configure the coverage upload step in CI to fail when below threshold — 2026-04-17 (`fail_ci_if_error` + `fail_under = 85`)

### P1 — medium term

- [x] 4.6 `codecov.yml` with component tracking (coordinator, modbus, mqtt, hass, resilience) — 2026-04-17
- [x] 4.7 Codecov: patch threshold 0.05 %, project target auto — 2026-04-17 (`codecov.yml`)
- [x] 4.8 `pip-audit` or `osv-scanner` workflow (weekly) for dependency CVEs — 2026-04-17 (`.github/workflows/security.yml`)
- [x] 4.9 `python-typing-update` hook in pre-commit — 2026-04-17 (py313 active)

### P2 — long term

- [x] 4.10 Release Drafter or automatic version bumping on merges to `main` — 2026-04-18 (`release-drafter.yml` workflow keeps the draft release + resolved version in sync on every merge)
- [x] 4.11 Extend the matrix: OS (Linux, macOS), ARM runner for Raspberry-Pi validation — 2026-04-19 (macOS runner added for 3.13 + 3.14 on 2026-04-18; ARM runner pinned to 3.13 on `ubuntu-24.04-arm` added today — covers Raspberry Pi 4+ and current Synology NAS)

---

## 5. Security

### P0 — quick wins

- [x] 5.1 Run `bandit` actively in CI (not only in test-requirements) — 2026-04-17 (`.github/workflows/security.yml`)
- [x] 5.2 Document TLS options for MQTT in `config.py` + clarify default behaviour — 2026-04-17 (config-template.yaml)
- [x] 5.3 Env variable overrides for every credential (`MTEC_MQTT_PASSWORD`, `MTEC_MODBUS_IP`) — 2026-04-17 (`_apply_env_overrides`)
- [x] 5.4 `.env.example` listing every supported variable — 2026-04-17

### P1 — medium term

- [x] 5.5 Pydantic input validation for _every_ config field (not only register models) — 2026-04-17 (`aiomtec2mqtt/config_schema.py` + validation in `init_config`)
- [x] 5.6 Secret-scanning hook (`detect-secrets` or `gitleaks`) in pre-commit — 2026-04-17 (gitleaks)
- [x] 5.7 Add a security policy: `SECURITY.md` with reporting path + supported versions — 2026-04-17

### P2 — long term

- [x] 5.8 SBOM generation in the release workflow (CycloneDX) — 2026-04-18 (`python-publish.yml` generates JSON + XML SBOMs via `cyclonedx-bom`, attaches them to the GitHub Release)
- [x] 5.9 Signed releases (Sigstore/cosign) for wheels/sdists — 2026-04-19 (`python-publish.yml` adds a `sigstore-sign` job using `sigstore/gh-action-sigstore-python`; `.sigstore.json` bundles attached to the Release; verification recipe documented in `SECURITY.md`)

---

## 6. Documentation

### P0 — quick wins

- [x] 6.1 Add `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` — 2026-04-17
- [x] 6.2 Unify the docstring style (Google style everywhere, enforced by ruff `D` rules) — 2026-04-17 (`[tool.ruff.lint.pydocstyle] convention = "google"`)

### P1 — medium term

- [x] 6.3 Stand up an MkDocs Material site under `docs/` — 2026-04-17 (`mkdocs.yml`, Material theme, tab navigation)
- [x] 6.4 `docs.yml` workflow for GitHub Pages deployment — 2026-04-17 (`.github/workflows/docs.yml`, `--strict` build + Pages deploy)
- [x] 6.5 User guide: installation, config, MQTT topics, HA integration, evcc integration, troubleshooting — 2026-04-17 (`docs/user-guide/*.md`)
- [x] 6.6 Auto-generate the register reference from `registers.yaml` (MkDocs hook) — 2026-04-17 (`script/gen_register_reference.py` → `docs/reference/registers.md`)
- [x] 6.7 Developer guide: setup, architecture, testing, release process — 2026-04-17 (`docs/developer-guide/*.md`)

### P2 — long term

- [x] 6.8 ADRs (Architecture Decision Records) for the async migration, resilience design, DI container choice — 2026-04-17 (ADR-001 container, ADR-002 resilience, ADR-003 async migration, ADR-004 CoordinatorConfig builder)
- [x] 6.9 Migration guides between major versions — 2026-04-18 (`docs/user-guide/migration.md` with checklist + placeholder `1.x → 2.0.0` section, linked into nav)
- [x] 6.10 Glossary (Modbus, MQTT, M-TEC terminology) — 2026-04-17 (`docs/reference/glossary.md`, linked from MkDocs nav)

---

## 7. Dependencies & release automation

### P0 — quick wins

- [x] 7.1 Document `uv` end-to-end (install, dev setup, CI) — 2026-04-17 (README)
- [x] 7.2 Reproducibly generate `requirements.txt` via `uv pip compile` from `pyproject.toml` — 2026-04-17 (`script/compile-requirements.sh`)
- [x] 7.3 Check upper bounds for critical deps (major updates can break) — 2026-04-17 (<7/<4/<3/<3/<1)

### P1 — medium term

- [x] 7.4 Optional `[fast]` extras with `orjson` (like aiohomematic) — 2026-04-17 (`[project.optional-dependencies].fast` + `aiomtec2mqtt/_json.py`)
- [x] 7.5 Dual-release workflow once a `test_support` package exists (step 3.12) — 2026-04-19 (`python-publish.yml` rebuilt with a 2-entry matrix: builds, publishes, signs, and SBOMs both `aiomtec2mqtt` and `aiomtec2mqtt-test-support` in lockstep; `release-please-config.json` carries `extra-files` markers so Release-Please bumps both packages atomically)
- [x] 7.6 Changelog automation: extraction from PR labels or Conventional Commits — 2026-04-17 (`.github/release-drafter.yml`)

### P2 — long term

- [x] 7.7 Automatic version bumping on merge (Release-Please or similar) — 2026-04-19 (`release-please.yml`, `release-please-config.json`, `.release-please-manifest.json`; `release-on-tag.yml` skip-guard avoids duplicate releases)
- [x] 7.8 Investigate publish channels: Docker image on GHCR, Debian/APK packaging — 2026-04-19 (multi-stage `Dockerfile` + `.dockerignore`; `docker-publish.yml` workflow builds linux/amd64+arm64 images via buildx, pushes to `ghcr.io/sukramj/aiomtec2mqtt` with semver/edge/latest tags, attaches provenance + SBOM; user-guide installation page now documents docker-compose usage. Debian/APK left out of scope — pip + container cover the supported platforms.)

---

## Review cadence

- **Weekly**: review P0 progress, note new blockers.
- **Monthly**: refresh the overview table above, mark closed items with PR number/date.
- **Quarterly**: realign the roadmap against the current state of `aiohomematic` (the target keeps moving).

---

## History

| Date       | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-04-17 | Initial roadmap drafted (basis: comparative analysis aiomtec2mqtt ↔ aiohomematic).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-04-17 | All 26 P0 tasks closed (architecture, code quality, tests, CI/CD, security, docs, dependencies).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-17 | 15 / 26 P1 tasks closed: code quality (pylint plugins, per-file ignores, McCabe), tests (contract tests, conftest patch.stopall), CI/CD (codecov components, pip-audit, typing-update), security (pydantic config validation, gitleaks, SECURITY.md), docs (auto-generated register reference), dependencies ([fast] extras, changelog automation). Open: sub-package refactor (1.4–1.7), mock Modbus server (3.6), test doubling (3.9), MkDocs site (6.3–6.5, 6.7), dual release (7.5).                                                                                                                |
| 2026-04-17 | Another 9 P1 tasks closed: architecture (1.4 sub-packages, 1.5 interfaces, 1.6 CoordinatorConfigBuilder, 1.7 ServiceContainer decision documented), tests (3.6 MockModbusTransport), docs (6.3 MkDocs Material, 6.4 docs.yml + Pages deploy, 6.5 user guide complete, 6.7 developer guide complete). P2 partially started: ADR-001 ServiceContainer. P1 standing now 24/26 — only 3.9 (test doubling, in flight) and 7.5 (dual release, blocked on P2-3.12) remain.                                                                                                                                     |
| 2026-04-17 | Started P2 chunk: ADR-002 (resilience design), ADR-003 (async migration), ADR-004 (CoordinatorConfig builder), and the glossary at `docs/reference/glossary.md`. Closed P2-6.8 and P2-6.10. All documentation now in English (project-wide language switch).                                                                                                                                                                                                                                                                                                                                            |
| 2026-04-18 | "Release-hardening" batch — closed P1-3.9 (555 tests, ~500 target exceeded), P2-4.10 (`release-drafter.yml` workflow), P2-5.8 (CycloneDX SBOM in `python-publish.yml`), P2-6.9 (migration guide stub). P2-4.11 in progress (macOS runner added, ARM runner still open). Totals now 56/68. Corrected pre-existing mis-counts in the overview table (Docs P2 was 1/3 but actually had 2/3 done → now 3/3).                                                                                                                                                                                                |
| 2026-04-19 | Closed P2-3.11 (`tests/test_benchmarks.py` opt-in via `-m benchmark`), P2-5.9 (Sigstore signing job in `python-publish.yml` + `SECURITY.md` verification recipe) and P2-7.7 (Release-Please integration: `release-please.yml`, `release-please-config.json`, `.release-please-manifest.json`; `release-on-tag.yml` carries a skip-guard so it does not collide with releases created by Release-Please). Totals now 59/68.                                                                                                                                                                              |
| 2026-04-19 | Closed P2-3.10 (`aiomtec2mqtt/session_replay.py` — `SessionRecorder` wraps any pymodbus-style transport and captures successful reads into ZIP-packaged frames; `SessionPlayer` is a pymodbus-compatible replacement that advances frames after each full read cycle, supports looped or pinned playback). 24 new tests cover load/advance/loop semantics and end-to-end record→replay. Test count now 600. Totals 60/68.                                                                                                                                                                               |
| 2026-04-19 | Closed P2-3.12 (`test_support/` subdir hosts the standalone `aiomtec2mqtt-test-support` distribution with its own `pyproject.toml`, src-layout, and pytest plugin entry point). Public surface: `transports`, `fakes`, `fixtures`, `assertions`. 9 smoke tests verify re-exports + auto-registered fixtures. Tests P2 area now 3/3. With 3.12 in place, P1-7.5 (dual-release workflow) is unblocked. Totals 61/68.                                                                                                                                                                                      |
| 2026-04-19 | Closed P1-7.5 — `python-publish.yml` rebuilt with a 2-entry matrix (`aiomtec2mqtt` + `aiomtec2mqtt-test-support`), each with its own dist artifact, PyPI Trusted-Publishing upload, Sigstore signing, and CycloneDX SBOM. Release-Please now keeps the test_support package version in lockstep via `extra-files` markers in `pyproject.toml` + `__init__.py`. Local dual `python -m build` confirmed clean wheels with no cross-contamination. P1 ring fully closed (26/26). Totals 62/68.                                                                                                             |
| 2026-04-19 | ROADMAP catch-up: P2-1.9 (decorator layer) was already implemented in a prior sprint — `aiomtec2mqtt/decorators/__init__.py` ships `@retry`, `@measure`, `@callback`, all sync/async aware, with 21 tests in `tests/test_decorators.py`. Marked as done; adoption at call sites is a free-standing follow-up. Totals 63/68.                                                                                                                                                                                                                                                                             |
| 2026-04-19 | Closed P2-7.8 (container publishing) — added a multi-stage `Dockerfile` (slim runtime, non-root user, `/config` volume), `.dockerignore`, and `.github/workflows/docker-publish.yml` (buildx, linux/amd64+arm64, semver/edge/latest tags, provenance + SBOM attached). User-guide installation page documents docker-compose usage. Local smoke build verified — image is 235 MB, runs as UID 1000, entrypoint resolves to `aiomtec2mqtt`. Debian/APK explicitly out of scope — pip + container cover the supported platforms. Dependencies & release area now 8/8 closed. Totals 64/68.                |
| 2026-04-19 | Closed P2-4.11 (CI matrix on `ubuntu-24.04-arm` pinned to Python 3.13 — covers Raspberry Pi 4+ and current Synology NAS units, the production deployment targets) and P2-1.8 (no separate `aiomtec2mqtt_storage` package needed; ADR-005 documents why — bridge is stateless, `session_replay.py` already covers the recorded-state use case, revisit triggers listed). Architecture and CI/CD areas both fully closed. Open: 2.8 + 2.9 (Python 3.14 floor + free-threaded support, deferred until aiohomematic leads). Totals 66/68.                                                                   |
| 2026-04-19 | Closed P2-2.9 (free-threaded 3.14t preparation): `hass_int.py` migrated from stdlib `json.dumps` to the existing `aiomtec2mqtt._json` wrapper, so the orjson fast-path lights up automatically once a free-threaded orjson wheel ships and the stdlib fallback covers the current 3.14t case where orjson has no wheels. CI already exercises 3.14t as `experimental: true`. Strategy + exceptions (e.g. `session_replay.py` keeps stdlib `json` for `indent=2`) documented in `docs/developer-guide/setup.md`. Only P2-2.8 (Python 3.14 floor) remains — strategic, awaits aiohomematic. Totals 67/68. |
| 2026-04-19 | Closed P2-2.8 (Python 3.14 floor) by deferral — ADR-006 documents the decision: `aiohomematic` already declares `requires-python = ">=3.14"` (trigger met), but the production deployment surface (Raspberry Pi OS, Synology DSM, Home Assistant OS) tracks Debian, which is not expected to ship Python 3.14 as the system default before Debian 14 "Forky" ~mid-2027. CI continues to exercise 3.14 + 3.14t. Revisit triggers + the mechanical bump checklist captured in the ADR. **Roadmap fully closed: 68/68 — every theme area at 100 %.**                                                       |
