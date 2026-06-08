#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/build_dev_regular_dataset.sh START_DATE END_DATE [LABEL]

Build a development-only regular-season dataset from Statcast and MLB StatsAPI feeds.
Example:
  scripts/build_dev_regular_dataset.sh 2022-04-07 2022-04-10
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

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

START_DATE="$1"
END_DATE="$2"
LABEL="$(sanitize_label "${3:-${START_DATE}_${END_DATE}}")"
reject_locked_args "$START_DATE" "$END_DATE" "$LABEL"

if [[ "$START_DATE" > "$END_DATE" ]]; then
  echo "START_DATE must be on or before END_DATE" >&2
  exit 2
fi

command -v uv >/dev/null 2>&1 || {
  echo "uv is required. Install it before running this script." >&2
  exit 127
}

mkdir -p data/work \
  data/processed/snapshots/role=dev_regular \
  data/processed/profiles/role=dev_regular \
  data/processed/dev_dataset/role=dev_regular \
  reports/generated/validation

echo "== Download Statcast dev regular range with resumable chunks =="
STATCAST_CHUNK_DAYS="${STATCAST_CHUNK_DAYS:-7}"
STATCAST_CHUNK_ROOT="data/raw/statcast_chunks/role=dev_regular/start=${START_DATE}_end=${END_DATE}"
echo "Statcast chunk cache: $STATCAST_CHUNK_ROOT (chunk_days=$STATCAST_CHUNK_DAYS)"
STATCAST_PATH="$(
  uv run baseball-zerobase download-statcast-dev-regular-chunked \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --chunk-days "$STATCAST_CHUNK_DAYS" | tail -n 1
)"
reject_locked_path "$STATCAST_PATH"
echo "Statcast parquet: $STATCAST_PATH"

GAME_PKS_PATH="data/work/game_pks_${LABEL}.parquet"
MISSING_GAME_PKS_PATH="data/work/game_pks_${LABEL}_missing_normalized.parquet"
PITCH_EVENTS_PATH="data/work/pitch_events_${LABEL}.parquet"
PREPARED_PATH="data/work/prepared_pitch_${LABEL}.parquet"
SNAPSHOTS_PATH="data/processed/snapshots/role=dev_regular/snapshots_${LABEL}.parquet"
PROFILED_SNAPSHOTS_PATH="data/processed/snapshots/role=dev_regular/snapshots_${LABEL}_profiled.parquet"
PITCHER_PROFILES_PATH="data/processed/profiles/role=dev_regular/pitcher_profiles_${LABEL}.parquet"
BATTER_PROFILES_PATH="data/processed/profiles/role=dev_regular/batter_profiles_${LABEL}.parquet"
DAILY_STATE_PATH="data/processed/profiles/role=dev_regular/daily_state_${LABEL}.parquet"
DATASET_PATH="data/processed/dev_dataset/role=dev_regular/dev_dataset_${LABEL}.parquet"
VALIDATION_REPORT="reports/generated/validation/dev_dataset_${LABEL}.json"

export STATCAST_PATH GAME_PKS_PATH
echo "== Derive game_pk list =="
uv run python - <<'PY'
import os
from pathlib import Path

import polars as pl

statcast_path = Path(os.environ["STATCAST_PATH"])
game_pks_path = Path(os.environ["GAME_PKS_PATH"])
frame = pl.read_parquet(statcast_path)
roles = set(frame.get_column("game_type").cast(pl.String).unique().to_list())
if roles != {"R"}:
    raise SystemExit(f"expected only regular-season game_type R rows, found {sorted(roles)}")
game_pks = frame.select(pl.col("game_pk").drop_nulls().cast(pl.Int64).unique().sort())
if game_pks.is_empty():
    raise SystemExit("no game_pk values found in Statcast partition")
game_pks_path.parent.mkdir(parents=True, exist_ok=True)
game_pks.write_parquet(game_pks_path)
print(f"Wrote {game_pks.height} game_pk values to {game_pks_path}")
PY

export MISSING_GAME_PKS_PATH
echo "== Check normalized game feed cache =="
MISSING_GAME_COUNT="$(
  uv run python - <<'PY'
import os
from pathlib import Path

import polars as pl

game_pks_path = Path(os.environ["GAME_PKS_PATH"])
missing_game_pks_path = Path(os.environ["MISSING_GAME_PKS_PATH"])
game_pks = (
    pl.read_parquet(game_pks_path)
    .get_column("game_pk")
    .cast(pl.Int64)
    .to_list()
)
missing = []
for game_pk in game_pks:
    base = Path("data/normalized/games") / f"game_pk={int(game_pk)}"
    if not (base / "game.parquet").exists() or not (base / "pitch_events.parquet").exists():
        missing.append(int(game_pk))

missing_game_pks_path.parent.mkdir(parents=True, exist_ok=True)
pl.DataFrame({"game_pk": missing}, schema={"game_pk": pl.Int64}).write_parquet(
    missing_game_pks_path
)
print(len(missing))
PY
)"

if [[ "$MISSING_GAME_COUNT" -gt 0 ]]; then
  echo "== Download $MISSING_GAME_COUNT missing normalized game feeds =="
  uv run baseball-zerobase download-games --game-pks-parquet "$MISSING_GAME_PKS_PATH"
else
  echo "== No missing normalized game feeds; skipping download-games =="
fi

export PITCH_EVENTS_PATH PREPARED_PATH
echo "== Merge normalized feeds and attach starter/lineup context =="
uv run python - <<'PY'
import os
from pathlib import Path

import polars as pl

from baseball_zerobase.data.starter_lineup import attach_starter_and_lineup_context

statcast_path = Path(os.environ["STATCAST_PATH"])
game_pks_path = Path(os.environ["GAME_PKS_PATH"])
pitch_events_path = Path(os.environ["PITCH_EVENTS_PATH"])
prepared_path = Path(os.environ["PREPARED_PATH"])

game_pks = (
    pl.read_parquet(game_pks_path)
    .get_column("game_pk")
    .cast(pl.Int64)
    .to_list()
)
game_frames = []
event_frames = []
missing = []
for game_pk in game_pks:
    base = Path("data/normalized/games") / f"game_pk={int(game_pk)}"
    game_path = base / "game.parquet"
    event_path = base / "pitch_events.parquet"
    if not game_path.exists() or not event_path.exists():
        missing.append(str(base))
        continue
    game_frames.append(pl.read_parquet(game_path))
    event_frames.append(pl.read_parquet(event_path))

if missing:
    raise SystemExit("missing normalized game feed outputs: " + ", ".join(missing[:10]))
if not game_frames or not event_frames:
    raise SystemExit("no normalized game feed outputs found")

normalized_games = pl.concat(game_frames, how="vertical")
pitch_events = pl.concat(event_frames, how="vertical")
statcast = pl.read_parquet(statcast_path)
prepared = attach_starter_and_lineup_context(statcast, normalized_games)

pitch_events_path.parent.mkdir(parents=True, exist_ok=True)
prepared_path.parent.mkdir(parents=True, exist_ok=True)
pitch_events.write_parquet(pitch_events_path)
prepared.write_parquet(prepared_path)
print(f"Wrote {pitch_events.height} pitch events to {pitch_events_path}")
print(f"Wrote {prepared.height} prepared Statcast rows to {prepared_path}")
PY

echo "== Build snapshots =="
uv run baseball-zerobase build-snapshots \
  --prepared-pitch-parquet "$PREPARED_PATH" \
  --normalized-pitch-events-parquet "$PITCH_EVENTS_PATH" \
  --output-parquet "$SNAPSHOTS_PATH"

echo "== Build separate profile artifacts =="
uv run baseball-zerobase build-pitcher-profiles \
  --input "$SNAPSHOTS_PATH" \
  --output-parquet "$PITCHER_PROFILES_PATH"
uv run baseball-zerobase build-batter-profiles \
  --input "$SNAPSHOTS_PATH" \
  --output-parquet "$BATTER_PROFILES_PATH"
uv run baseball-zerobase build-daily-state \
  --input "$SNAPSHOTS_PATH" \
  --output-parquet "$DAILY_STATE_PATH"

export SNAPSHOTS_PATH PROFILED_SNAPSHOTS_PATH
echo "== Build profiled snapshots for training =="
uv run python - <<'PY'
import os
from pathlib import Path

import polars as pl

from baseball_zerobase.data.snapshots import write_snapshot_dataset
from baseball_zerobase.profiles.batter_archetype import add_batter_archetypes
from baseball_zerobase.profiles.batter_threat import add_batter_threat
from baseball_zerobase.profiles.daily_state import add_daily_state
from baseball_zerobase.profiles.pitcher import add_pitcher_profiles

snapshots_path = Path(os.environ["SNAPSHOTS_PATH"])
profiled_path = Path(os.environ["PROFILED_SNAPSHOTS_PATH"])
snapshots = pl.read_parquet(snapshots_path)
profiled = add_pitcher_profiles(snapshots)
profiled = add_batter_archetypes(profiled)
profiled = add_batter_threat(profiled)
profiled = add_daily_state(profiled)
manifest = write_snapshot_dataset(
    profiled,
    profiled_path,
    source="baseball-zerobase.profiled-snapshots",
    request={"snapshots": str(snapshots_path.resolve())},
)
print(f"Wrote {profiled.height} profiled snapshots to {profiled_path} ({manifest.path})")
PY

echo "== Build development dataset from profiled snapshots =="
uv run baseball-zerobase build-dev-dataset \
  --snapshots-parquet "$PROFILED_SNAPSHOTS_PATH" \
  --output-parquet "$DATASET_PATH" \
  --min-prior-pitches 0

echo "== Validate development dataset =="
uv run baseball-zerobase validate-dataset \
  --input "$DATASET_PATH" \
  --report "$VALIDATION_REPORT"

cat <<EOF
== Outputs ==
game_pks=$GAME_PKS_PATH
pitch_events=$PITCH_EVENTS_PATH
prepared_pitch=$PREPARED_PATH
snapshots=$SNAPSHOTS_PATH
profiled_snapshots=$PROFILED_SNAPSHOTS_PATH
dataset=$DATASET_PATH
validation_report=$VALIDATION_REPORT
EOF
