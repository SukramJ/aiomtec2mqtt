#!/usr/bin/env sh
# Reproducibly regenerate requirements.txt from pyproject.toml using uv.
# Requires: https://docs.astral.sh/uv/
set -eu

cd "$(git rev-parse --show-toplevel)"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not installed. See https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv pip compile pyproject.toml -o requirements.txt
echo "requirements.txt regenerated from pyproject.toml"
