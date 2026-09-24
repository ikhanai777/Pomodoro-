#!/usr/bin/env sh
# Starts the DepthForge server (Linux/macOS). Setup: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd "$(dirname "$0")"
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
exec "$PY" app.py
