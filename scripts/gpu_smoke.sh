#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

reject_locked_path() {
  local value="$1"
  case "$value" in
    *data/locked*|*2025_postseason*|*postseason*|*game_type=F*|*game_type=D*|*game_type=L*|*game_type=W*)
      echo "Refusing locked or excluded input token: $value" >&2
      exit 2
      ;;
  esac

  if [[ "$value" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] \
    && [[ "$value" > "2026-03-24" ]] \
    && [[ "$value" < "2026-06-01" ]]; then
    echo "Refusing locked 2026 date: $value (locked range is 2026-03-25..2026-05-31)" >&2
    exit 2
  fi
}

reject_locked_args() {
  for value in "$@"; do
    reject_locked_path "$value"
  done
}

reject_locked_args "$@"

echo "== Repository =="
git status --short --branch

echo "== GPU diagnostics =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "nvidia-smi not found; continuing because current baseline training is CPU-compatible."
fi

echo "== Python environment =="
command -v uv >/dev/null 2>&1 || {
  echo "uv is required. Install it before running this script." >&2
  exit 127
}
uv --version

echo "== Dependency sync =="
uv sync

echo "== Quality gate =="
scripts/check.sh

echo "== Synthetic pipeline smoke =="
uv run baseball-zerobase pipeline-smoke
