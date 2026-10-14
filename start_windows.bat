@echo off
cd /d "%~dp0"
where uv >nul 2>nul || (
  echo uv is not installed. Install it from https://docs.astral.sh/uv/ and run this file again.
  pause
  exit /b 1
)
uv sync --quiet
uv run python server.py
pause
