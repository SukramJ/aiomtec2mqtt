#!/usr/bin/env bash
# Activate the project venv (./.venv) and exec the given command.
# Used by prek hooks (mypy/pylint) so they find project deps when prek
# is invoked outside an activated shell (e.g. as a Git pre-commit hook).
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
activate="${repo_root}/.venv/bin/activate"

if [ ! -f "${activate}" ]; then
  echo "error: ${activate} not found — run 'script/setup' first" >&2
  exit 1
fi

# shellcheck disable=SC1090
. "${activate}"

exec "$@"
