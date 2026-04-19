#!/usr/bin/env bash
# Activate the project venv (./.venv) if present, then exec the given
# command. Used by prek hooks (mypy/pylint) so they find project deps
# when prek is invoked outside an activated shell (e.g. as a Git
# pre-commit hook). In CI the deps are installed directly into the
# system Python, so missing `.venv` falls back to PATH lookup.
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
activate="${repo_root}/.venv/bin/activate"

if [ -f "${activate}" ]; then
  # shellcheck disable=SC1090
  . "${activate}"
fi

exec "$@"
