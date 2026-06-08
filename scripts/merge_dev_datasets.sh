#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/merge_dev_datasets.sh LABEL INPUT_PARQUET INPUT_PARQUET [INPUT_PARQUET...]

Merge development regular-season datasets with schema and locked-data guards.
Example:
  scripts/merge_dev_datasets.sh 2022_2024_regular \
    data/processed/dev_dataset/role=dev_regular/dev_dataset_2022_regular.parquet \
    data/processed/dev_dataset/role=dev_regular/dev_dataset_2023_regular.parquet \
    data/processed/dev_dataset/role=dev_regular/dev_dataset_2024_regular.parquet
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

if [[ $# -lt 3 ]]; then
  usage
  exit 2
fi

LABEL="$(sanitize_label "$1")"
shift
INPUTS=("$@")

reject_locked_args "$LABEL" "${INPUTS[@]}"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required. Install it before running this script." >&2
  exit 127
}

DATASET_PATH="data/processed/dev_dataset/role=dev_regular/dev_dataset_${LABEL}.parquet"
VALIDATION_REPORT="reports/generated/validation/dev_dataset_${LABEL}.json"
reject_locked_args "$DATASET_PATH" "$VALIDATION_REPORT"

mkdir -p data/processed/dev_dataset/role=dev_regular reports/generated/validation

echo "== Merge development regular-season datasets =="
# The CLI helper rejects schema mismatch and writes the manifest with write_development_dataset.
uv run baseball-zerobase merge-dev-datasets \
  --output-parquet "$DATASET_PATH" \
  "$LABEL" \
  "${INPUTS[@]}"

echo "== Validate merged development regular-season dataset =="
uv run baseball-zerobase validate-dataset \
  --input "$DATASET_PATH" \
  --report "$VALIDATION_REPORT"

cat <<EOF
== Outputs ==
dataset=$DATASET_PATH
validation_report=$VALIDATION_REPORT
EOF
