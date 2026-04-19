# Developer Setup

## Prerequisites

- Python **3.13+**
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- Git + [`prek`](https://github.com/j178/prek) (pre-commit-compatible runner; pulled in by `requirements_test.txt`)

## Clone & install

```bash
git clone https://github.com/SukramJ/aiomtec2mqtt.git
cd aiomtec2mqtt
uv venv
source .venv/bin/activate
uv pip install -e ".[fast]"
uv pip install -r requirements_test.txt
```

## Git hooks (prek)

`prek` is a Rust-based, drop-in replacement for `pre-commit`. It reads the
existing `.pre-commit-config.yaml` unchanged, so the hook list and pinned
versions are shared between local runs and CI.

```bash
prek install              # wires up Git hooks
prek run --all-files      # run the full hook set on demand
```

Hooks include:

- `ruff` — linter + formatter (also replaces `black`/`isort`).
- `mypy` — strict typing (see `pyproject.toml` `[tool.mypy]`).
- `bandit`, `codespell`, `yamllint`, `gitleaks` — security & doc hygiene.
- `pylint` — additional static analysis (strict + per-file-ignores plugins).
- `python-typing-update` — keeps type annotations modern (`py313+`).

## Run tests

```bash
pytest                                # parallel (xdist) with coverage gate
pytest tests/test_config_schema.py    # focused run
pytest -m "not slow"                  # skip slow markers
```

## Regenerate pinned requirements

```bash
./script/compile-requirements.sh
```

Uses `uv pip compile` against `pyproject.toml` to produce a reproducible
`requirements.txt` with all transitive pins.

## Regenerate register reference

```bash
PYTHONPATH=. python script/gen_register_reference.py
PYTHONPATH=. python script/gen_register_reference.py --check   # CI mode
```

## Free-threaded Python (3.14t) support

CI runs an extra matrix entry on `ubuntu-latest` with `python-version: "3.14t"` (PEP 703 free-threaded build), marked `experimental: true` so
its result does not gate merges. The application is structured to run
on it today:

- `aiomtec2mqtt/_json.py` exposes `dumps`/`loads` with an optional
  `orjson` fast path. Free-threaded wheels for `orjson` are not yet
  published — the wrapper transparently falls back to stdlib `json`
  when the import fails. Hot-path call sites (`hass_int.py`) use the
  wrapper, **not** stdlib `json` directly, so an upstream orjson
  free-threaded wheel turns into a free perf gain with no code change.
- The bridge holds no module-level mutable state outside the explicit
  coordinator/client objects, so the GIL-removal change of intent
  (cooperative locking instead of implicit serialisation) does not
  expose silent races in our code.
- Direct C-extension dependencies that may lack free-threaded wheels
  (`pymodbus`, `aiomqtt`, `pydantic`, `prometheus-client`,
  `cyclonedx-bom`) are tracked upstream. When a wheel is missing,
  `pip install` falls back to building from sdist.

If you add new hot-path JSON serialisation, route it through
`aiomtec2mqtt._json` rather than importing stdlib `json` directly. The
exception is code that needs `indent=`, `default=`, or other arguments
orjson does not support (e.g. `session_replay.py`, which writes
human-readable manifests with `indent=2`) — keep stdlib `json` there.

## Build the docs locally

```bash
uv pip install mkdocs-material
mkdocs serve
```

Then open <http://127.0.0.1:8000/>. The docs hot-reload while you edit.
