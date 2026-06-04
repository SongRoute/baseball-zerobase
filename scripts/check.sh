#!/usr/bin/env bash
set -euo pipefail

uv run ruff check .
uv run pyright src tests
uv run pytest
