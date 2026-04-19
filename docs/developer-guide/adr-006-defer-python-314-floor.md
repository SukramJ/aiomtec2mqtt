# ADR-006: Defer Python 3.14 floor until Debian stable ships it

- **Status**: Accepted
- **Date**: 2026-04-19
- **Context**: ROADMAP 2.8 — "Python 3.14 as the floor (once the
  aiohomematic path is confirmed)"

## Context

The trigger condition in roadmap entry 2.8 is met: `aiohomematic`
upstream (the project we explicitly use as the quality benchmark)
declares `requires-python = ">=3.14"` in its `pyproject.toml`. Bumping
our own floor would keep the two codebases aligned and let us adopt
3.14-only language features (PEP 750 t-strings, deferred annotation
evaluation defaults, the new `except*` improvements, the cleaner
`asyncio.TaskGroup` ergonomics).

The cost is borne by users, not us. The dominant production targets
for `aiomtec2mqtt` are:

- **Raspberry Pi 4+** running Raspberry Pi OS (Debian-derived).
- **Synology NAS** units running DSM, which packages a Debian-derived
  Python.
- **Home Assistant OS / supervised**, which tracks the version of
  Python shipped in its base image.

Each of these tracks Debian's release cadence. Debian 13 "Trixie"
shipped August 2025 with Python 3.13 as its system Python. Debian 14
"Forky" — the first release that will ship 3.14 — is not expected
before mid-to-late 2027. Forcing 3.14 today means a non-trivial slice
of users would have to install Python from a third-party source (deadsnakes
PPA, pyenv, conda) before they can run `pip install aiomtec2mqtt`. For
a single-purpose bridge that runs on appliance-class hardware, that is
a friction point disproportionate to the upside.

## Decision

**Defer** the 3.14 floor bump. Keep `requires-python = ">=3.13"` and
continue to test 3.14 (and 3.14t) in CI. Re-evaluate when Debian
stable ships Python 3.14 as its default system Python.

## Rationale

- **User impact dominates**: the project's core value is "drop-in
  bridge that runs on the SBC you already own". Requiring users to
  upgrade Python out of band breaks that promise.
- **CI coverage is already there**: the test matrix includes 3.13
  (production), 3.14 (next), and 3.14t (free-threaded experimental).
  We catch 3.14-incompatible code at PR time without forcing the
  floor up.
- **No 3.14-only feature on the must-have list**: nothing in the
  current backlog requires a 3.14-only language feature. The bump
  would be cosmetic alignment with `aiohomematic`, not a capability
  unlock.
- **Symmetric with 1.8 (ADR-005)**: we already declined to add a
  storage package "if needed". The 2.8 entry was hedged the same way
  ("once the aiohomematic path is confirmed") — meeting the trigger
  is necessary but not sufficient.
- **Library JSON wrapper covers the orjson-on-3.14t gap**: the
  `aiomtec2mqtt._json` shim already lets us run on 3.14 / 3.14t
  without `orjson` wheels. The 3.13 → 3.14 bump would not eliminate
  that shim — `orjson` free-threaded wheels are the gating factor,
  not the floor version.

## Constraints

Revisit this decision when **any** of the following becomes true:

- Debian stable (or Raspberry Pi OS, whichever moves first) ships
  Python 3.14 as the system Python by default.
- Home Assistant OS bumps its base image to Python 3.14.
- A 3.14-only language feature lands on the implementation roadmap
  with a strong enough rationale to justify the user-side migration.
- An external dependency we ship with (`pymodbus`, `aiomqtt`,
  `pydantic`) drops 3.13 from its supported versions, forcing the
  bump regardless.

When the bump happens, the migration is mechanical:

- `pyproject.toml`: `requires-python`, classifiers, ruff
  `target-version`, mypy `python_version`.
- CI matrices in `test-run.yaml`, `python-publish.yml`,
  `security.yml`, `docs.yml`.
- `Dockerfile` base image (`python:3.13-slim` → `python:3.14-slim`).
- pre-commit `python-typing-update` from `py313` to `py314`.
- Documentation (`README.md`, `docs/user-guide/installation.md`,
  `docs/developer-guide/setup.md`, `CLAUDE.md`).
- `docs/user-guide/migration.md` — populate the `1.x → 2.0` section
  with the version bump as the headline breaking change.

## Alternatives considered

- **Bump now (match `aiohomematic`)**: rejected — see "User impact"
  above. Strict alignment with the upstream benchmark is not worth
  forcing every Pi user to install Python out of band.
- **Drop 3.13 from CI but keep `requires-python = ">=3.13"`**:
  rejected — the runtime metadata would lie about what we test.
- **Add a `[modern]` extra that requires 3.14 for orjson + other
  optionals**: rejected — adds cognitive load and a second supported
  matrix without solving the actual user problem (system Python on
  the SBC is too old).
