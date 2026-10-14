#!/bin/sh
# Starts Obj Remover on macOS / Linux. Needs `uv` (https://docs.astral.sh/uv/).
cd "$(dirname "$0")" || exit 1
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
uv sync --quiet
exec uv run python server.py
