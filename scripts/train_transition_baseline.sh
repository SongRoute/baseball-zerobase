#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/train_transition_baseline.sh DATASET_PARQUET [LABEL]

Fit and evaluate the deterministic shared transition baseline on a dev dataset.
Example:
  scripts/train_transition_baseline.sh \
    data/processed/dev_dataset/role=dev_regular/dev_dataset_2022-04-07_2022-04-10.parquet
USAGE
}

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

sanitize_label() {
  local label
  label="$(printf '%s' "$1" | LC_ALL=C tr -c 'A-Za-z0-9_.-' '_' | sed 's/^_*//;s/_*$//')"
  if [[ -z "$label" ]]; then
    echo "label cannot be empty after sanitization" >&2
    exit 2
  fi
  printf '%s\n' "$label"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

DATASET_PATH="$1"
DEFAULT_LABEL="$(basename "$DATASET_PATH" .parquet)"
LABEL="$(sanitize_label "${2:-$DEFAULT_LABEL}")"
reject_locked_args "$DATASET_PATH" "$LABEL"

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset parquet does not exist: $DATASET_PATH" >&2
  exit 2
fi

command -v uv >/dev/null 2>&1 || {
  echo "uv is required. Install it before running this script." >&2
  exit 127
}

mkdir -p artifacts/models/transition reports/generated/transition

MODEL_PATH="artifacts/models/transition/v0_${LABEL}.json"
REPORT_PATH="reports/generated/transition/v0_${LABEL}.json"

echo "== Fit transition baseline =="
uv run baseball-zerobase fit-transition-model \
  --dataset "$DATASET_PATH" \
  --output "$MODEL_PATH"

echo "== Evaluate transition baseline =="
uv run baseball-zerobase evaluate-transition-model \
  --dataset "$DATASET_PATH" \
  --model "$MODEL_PATH" \
  --report "$REPORT_PATH"

cat <<EOF
== Outputs ==
model=$MODEL_PATH
report=$REPORT_PATH
EOF
