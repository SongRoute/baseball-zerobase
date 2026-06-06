#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/report_transition_diagnostics.sh DATASET_PARQUET REPORT_JSON

Write transition_diagnostics JSON and a full-scale training markdown summary.
Example:
  scripts/report_transition_diagnostics.sh \
    data/processed/dev_dataset/role=dev_regular/dev_dataset_2022_2024_regular.parquet \
    reports/generated/diagnostics/transition_diagnostics_2022_2024_regular.json
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

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

DATASET_PATH="$1"
DIAGNOSTICS_REPORT="$2"
reject_locked_args "$DATASET_PATH" "$DIAGNOSTICS_REPORT"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required. Install it before running this script." >&2
  exit 127
}

mkdir -p "$(dirname "$DIAGNOSTICS_REPORT")" reports/generated/training

echo "== Write transition diagnostics =="
uv run baseball-zerobase report-transition-diagnostics \
  --dataset "$DATASET_PATH" \
  --report "$DIAGNOSTICS_REPORT"

DATASET_BASENAME="$(basename "$DATASET_PATH" .parquet)"
LABEL="${DATASET_BASENAME#dev_dataset_}"
RUN_DATE="$(date +%Y%m%d)"
TRAINING_REPORT="reports/generated/training/full_scale_${LABEL}_${RUN_DATE}.md"

export DATASET_PATH DIAGNOSTICS_REPORT LABEL RUN_DATE TRAINING_REPORT
echo "== Write full-scale training summary =="
uv run python - <<'PY'
import json
import os
from datetime import date
from pathlib import Path

dataset_path = os.environ["DATASET_PATH"]
diagnostics_path = Path(os.environ["DIAGNOSTICS_REPORT"])
label = os.environ["LABEL"]
training_report = Path(os.environ["TRAINING_REPORT"])
payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))

validation_path = Path(f"reports/generated/validation/dev_dataset_{label}.json")
transition_path = Path(f"reports/generated/transition/v0_{label}.json")
validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
transition = json.loads(transition_path.read_text(encoding="utf-8")) if transition_path.exists() else {}

def top_items(name, limit=8):
    values = payload.get(name, {})
    if not isinstance(values, dict):
        return []
    return sorted(values.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]

lines = [
    f"# {label} Full-Scale Training Diagnostics",
    "",
    f"Date: {date.today().isoformat()}",
    f"Dataset: `{dataset_path}`",
    "",
    "## Commands",
    "",
    "```bash",
    f"scripts/report_transition_diagnostics.sh {dataset_path} {diagnostics_path}",
    "```",
    "",
    "## Dataset Validation",
    "",
    f"- row_count: {validation.get('row_count', payload.get('row_count'))}",
    f"- locked_row_count: {validation.get('locked_row_count', 'not_available')}",
    f"- timestamp_join_rate: {validation.get('timestamp_join_rate', 'not_available')}",
    "",
    "## Transition Baseline Metrics",
    "",
    f"- log_loss: {transition.get('log_loss', 'not_available')}",
    f"- expected_calibration_error: {transition.get('expected_calibration_error', 'not_available')}",
    f"- rare_outcome_recall: {transition.get('rare_outcome_recall', 'not_available')}",
    f"- home_run_recall: {transition.get('home_run_recall', 'not_available')}",
    f"- support_min: {transition.get('support_min', 'not_available')}",
    f"- support_max: {transition.get('support_max', 'not_available')}",
    "",
    "## Diagnostics",
    "",
    f"- row_count: {payload.get('row_count')}",
    f"- pitcher_pitch_type_owned_true_rate: {payload.get('pitcher_pitch_type_owned_true_rate')}",
    f"- pitch_type_distribution_top: {top_items('pitch_type_distribution')}",
    f"- relative_zone_distribution_top: {top_items('relative_zone_distribution')}",
    f"- batter_weakness_archetype_distribution: {payload.get('batter_weakness_archetype_distribution')}",
    f"- batter_threat_score_bucket_distribution: {payload.get('batter_threat_score_bucket_distribution')}",
    f"- pitcher_profile_reliability_weight_bucket_distribution: {payload.get('pitcher_profile_reliability_weight_bucket_distribution')}",
    f"- profile_feature_null_rates: {payload.get('profile_feature_null_rates')}",
    f"- daily_state_count_summary: {payload.get('daily_state_count_summary')}",
    f"- label_outcome_distribution: {payload.get('label_outcome_distribution')}",
    "",
    "## Milestone 5 Preconditions",
    "",
    "- Evaluate the full candidate pitch type x 13-zone grid.",
    "- Do not remove candidates based on zone frequency or perceived reachability.",
    "- Use only features available before the target pitch.",
    "- Use the shared transition model for candidate scoring.",
    "- Include compact explanations in recommendation output.",
]

training_report.parent.mkdir(parents=True, exist_ok=True)
training_report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
print(training_report)
PY

cat <<EOF
== Outputs ==
diagnostics_report=$DIAGNOSTICS_REPORT
training_report=$TRAINING_REPORT
EOF
