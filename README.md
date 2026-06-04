# Baseball Zerobase

Baseball Zerobase is a clean-room rewrite for a leakage-safe MLB starting-pitcher
strategy research pipeline. This repository must not use code, model weights, or
processed data from the prior project.

Milestone 1-2 is limited to data foundations and empirical baselines: immutable
source acquisition, starter and lineup context, pre-pitch snapshots, as-of starter
eligibility, empirical behavior and transition baselines, inning simulation, and
rolling validation. Neural models, recommendations, API serving, and web UI work
are out of scope.

## Data Partitions

Development work uses only 2022-2025 MLB regular season data.

Locked partitions are never used during development:

- 2025 postseason games where `game_type` is `F`, `D`, `L`, or `W`
- 2026 regular season games from `2026-03-25` through `2026-05-31`

Do not run locked tests or read locked data paths while developing. Locked
evaluation is reserved for a separate final review step.

## Install And Quality

```bash
uv sync
uv run baseball-zerobase version
scripts/check.sh
```

`scripts/check.sh` runs the required local quality gate:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright src tests
uv run pytest -q
```

## Smoke Test

Use the synthetic pipeline smoke command for fast local validation. It never
downloads network data and is safe for automated tests.

```bash
uv run baseball-zerobase pipeline-smoke
```

The command prints a baseline metric summary including transition negative log
likelihood and simulation truncation rate.

## Small-Range Acquisition

Network acquisition is manual development work, not part of automated tests. Use
small date ranges first:

```bash
uv run baseball-zerobase download-statcast \
  --start 2022-04-07 \
  --end 2022-04-10
```

That writes an immutable development partition such as:

```text
data/raw/statcast/start=2022-04-07_end=2022-04-10.parquet
```

After extracting a development-only list of `game_pk` values to a local parquet,
download the matching MLB StatsAPI game feeds:

```bash
uv run baseball-zerobase download-games \
  --game-pks-parquet data/work/game_pks_2022-04-07_2022-04-10.parquet
```

Full-season downloads are long-running, network-dependent operations. They must
never be part of automated tests.

## Pipeline Commands

The persisted development pipeline is:

```bash
uv run baseball-zerobase download-statcast \
  --start 2022-04-07 \
  --end 2022-04-10

uv run baseball-zerobase download-games \
  --game-pks-parquet data/work/game_pks_2022-04-07_2022-04-10.parquet

uv run baseball-zerobase build-snapshots \
  --prepared-pitch-parquet data/work/prepared_pitch.parquet \
  --normalized-pitch-events-parquet data/normalized/games/game_pk=123456/pitch_events.parquet \
  --output-parquet data/processed/snapshots/role=dev_regular/snapshots.parquet

uv run baseball-zerobase build-dev-dataset \
  --snapshots-parquet data/processed/snapshots/role=dev_regular/snapshots.parquet \
  --output-parquet data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet

uv run baseball-zerobase validate-dataset \
  --input data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet \
  --report reports/generated/validation/dev_dataset.json

uv run baseball-zerobase evaluate-rolling \
  --dataset data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet \
  --output-dir reports/generated/rolling
```

`prepared_pitch.parquet` must contain development regular-season rows with
starter and stable-lineup context attached. Keep all paths out of `data/locked`.

## Language Review

User-facing English Markdown documents must have Korean review counterparts.
Maintain `README.md` and `README.ko.md` together.
