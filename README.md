# Baseball Zerobase

Baseball Zerobase is a clean-room rewrite for a leakage-safe MLB starting-pitcher
strategy research pipeline. This repository must not use code, model weights, or
processed data from the prior project.

Milestone 1-5 covers data foundations, empirical baselines, deterministic
personalization features, a contract-first shared transition model, and a
transition-proxy pitch recommendation CLI:
immutable source acquisition, starter and lineup context, pre-pitch snapshots,
as-of starter eligibility, pitcher profiles, batter weakness archetypes, batter
threat scores, same-day state summaries, empirical behavior and transition
baselines, inning simulation, rolling validation, and legal transition
probability distributions. Neural models, full inning-value simulation for
recommendations, API serving, and web UI work are out of scope.

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

uv run baseball-zerobase build-pitcher-profiles \
  --input data/processed/snapshots/role=dev_regular/snapshots.parquet \
  --output-parquet data/processed/profiles/role=dev_regular/pitcher_profiles.parquet

uv run baseball-zerobase build-batter-profiles \
  --input data/processed/snapshots/role=dev_regular/snapshots.parquet \
  --output-parquet data/processed/profiles/role=dev_regular/batter_profiles.parquet

uv run baseball-zerobase build-daily-state \
  --input data/processed/snapshots/role=dev_regular/snapshots.parquet \
  --output-parquet data/processed/profiles/role=dev_regular/daily_state.parquet

uv run baseball-zerobase build-dev-dataset \
  --snapshots-parquet data/processed/snapshots/role=dev_regular/snapshots.parquet \
  --output-parquet data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet

uv run baseball-zerobase validate-dataset \
  --input data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet \
  --report reports/generated/validation/dev_dataset.json

uv run baseball-zerobase evaluate-rolling \
  --dataset data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet \
  --output-dir reports/generated/rolling

uv run baseball-zerobase fit-transition-model \
  --dataset data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet \
  --output artifacts/models/transition/v0.json

uv run baseball-zerobase evaluate-transition-model \
  --dataset data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet \
  --model artifacts/models/transition/v0.json \
  --report reports/generated/transition/v0.json

uv run baseball-zerobase recommend-pitches \
  --input data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet \
  --model artifacts/models/transition/v0.json \
  --pitch-types FF,SL,CH \
  --row-index 0 \
  --top-k 10 \
  --output reports/generated/recommendations/pitch.json
```

`prepared_pitch.parquet` must contain development regular-season rows with
starter and stable-lineup context attached. Keep all paths out of `data/locked`.

## Personalization Features

Milestone 3 feature builders are deterministic transforms over development
snapshots. Pitcher profiles exclude the target game and use prior two-season
pitch history with early-season shrinkage flags. Batter weakness archetypes use
only response tendencies such as chase, whiff, and called-strike rates. Batter
threat scores use terminal outcome proxies such as reach, extra-base, home-run,
and strikeout rates. Daily state uses only same-game rows before the target
pitch.

Every Milestone 3 feature family writes an `*_as_of_timestamp` column. Validation
rejects feature timestamps that are null or not strictly before the target
`pitch_timestamp`.

## Shared Transition Model

Milestone 4 adds a deterministic shared transition model v0. It fits smoothed
development-only transition counts, excludes target label columns from model
features, emits normalized legal `TransitionAtom` distributions, and can be used
directly by the inning simulator. Component evaluation reports transition log
loss, calibration error, rare-outcome recall, and Korean summary text.

## Pitch Recommendations

Milestone 5 adds `recommend-pitches`. The command reads one profiled snapshot or
development dataset row, overwrites the row action for every candidate, scores
the full `pitch_type x 13 relative_zone` grid with the shared transition model,
and writes ranked JSON recommendations with compact explanations.

Zone frequency, action support, behavior-model probabilities, and perceived
reachability never remove first-pitch candidates. Target-row actual
`pitch_type`, actual `relative_zone`, outcome labels, post-pitch state, and raw
current-pitch measurements are replay labels only; they are not serving features.
The current ranking value is `transition_proxy`, an immediate transition-risk
score, not simulated expected inning runs.

## GPU Training Preparation

Use the GPU training runbook when preparing a remote server for dev-only dataset
generation and transition baseline training:

- English: `docs/gpu-training-runbook.md`
- Korean review: `docs/gpu-training-runbook.ko.md`

The runbook includes the server smoke, small real-data smoke, 2022 full-season
expansion, 2022-2024 expansion, diagnostics, and Milestone 5 recommendation
readiness steps.

## Language Review

User-facing English Markdown documents must have Korean review counterparts.
Maintain `README.md` and `README.ko.md` together.
