# GPU Training Runbook

This runbook prepares a GPU server to reproduce the development-only training
pipeline. Codex may run on the server, but the current Milestone 4 transition
baseline is CPU-compatible. The GPU server is still useful because it is the
intended host for larger data preparation, future neural training, and long
running experiments.

## GPU Server Quickstart

Run these commands after receiving the server:

```bash
git clone git@github.com:SongRoute/baseball-zerobase.git
cd baseball-zerobase
uv sync
scripts/gpu_smoke.sh
scripts/build_dev_regular_dataset.sh 2022-04-07 2022-04-10
scripts/train_transition_baseline.sh \
  data/processed/dev_dataset/role=dev_regular/dev_dataset_2022-04-07_2022-04-10.parquet
```

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If Codex CLI is needed on the server:

```bash
npm install -g @openai/codex
codex login --device-auth
codex --cd "$PWD" --sandbox workspace-write --ask-for-approval on-request
```

## Data Guardrails

Development runs may use only regular-season development data. Do not read or
write training inputs under `data/locked`.

Never use these locked partitions during development:

- 2025 postseason games with `game_type` `F`, `D`, `L`, or `W`
- 2026 regular-season games from `2026-03-25` through `2026-05-31`

The helper scripts reject obvious locked paths, postseason labels, and locked
2026 dates. The repository validation code still remains the source of truth for
dataset role checks.

## Stage 1: Server Smoke

Purpose: verify the server, Python environment, repo quality gate, and synthetic
pipeline.

```bash
scripts/gpu_smoke.sh
```

Expected outcome:

- `uv sync` completes
- `scripts/check.sh` passes
- `uv run baseball-zerobase pipeline-smoke` passes
- GPU diagnostics are printed if `nvidia-smi` is available

## Stage 2: Small Dev Dataset

Purpose: prove real data acquisition and profile-aware training work before
requesting full-season downloads.

```bash
scripts/build_dev_regular_dataset.sh 2022-04-07 2022-04-10
scripts/train_transition_baseline.sh \
  data/processed/dev_dataset/role=dev_regular/dev_dataset_2022-04-07_2022-04-10.parquet
```

Primary outputs:

```text
data/work/game_pks_2022-04-07_2022-04-10.parquet
data/work/pitch_events_2022-04-07_2022-04-10.parquet
data/work/prepared_pitch_2022-04-07_2022-04-10.parquet
data/processed/snapshots/role=dev_regular/snapshots_2022-04-07_2022-04-10_profiled.parquet
data/processed/dev_dataset/role=dev_regular/dev_dataset_2022-04-07_2022-04-10.parquet
artifacts/models/transition/v0_dev_dataset_2022-04-07_2022-04-10.json
reports/generated/transition/v0_dev_dataset_2022-04-07_2022-04-10.json
```

The dataset is built from profiled snapshots, so pitcher profile, batter
weakness, batter threat, and daily state features are available to the shared
transition model.

## Stage 3: 2022 Full Regular Season

After the small dev dataset succeeds, build the 2022 regular-season range:

```bash
scripts/build_dev_regular_dataset.sh 2022-04-07 2022-10-05 2022_regular
scripts/train_transition_baseline.sh \
  data/processed/dev_dataset/role=dev_regular/dev_dataset_2022_regular.parquet \
  2022_regular
```

Review:

- `reports/generated/validation/dev_dataset_2022_regular.json`
- `reports/generated/transition/v0_2022_regular.json`

Do not make this an automated test. It is network-dependent and long-running.

## Stage 4: 2022-2024 Expansion

Build annual datasets independently first:

```bash
scripts/build_dev_regular_dataset.sh 2022-04-07 2022-10-05 2022_regular
scripts/build_dev_regular_dataset.sh 2023-03-30 2023-10-01 2023_regular
scripts/build_dev_regular_dataset.sh 2024-03-28 2024-09-30 2024_regular
```

Next implementation step: add a checked merge script for annual dev datasets.
The intended merged output is:

```text
data/processed/dev_dataset/role=dev_regular/dev_dataset_2022_2024_regular.parquet
reports/generated/validation/dev_dataset_2022_2024_regular.json
artifacts/models/transition/v0_2022_2024_regular.json
reports/generated/transition/v0_2022_2024_regular.json
```

The merge script should validate each annual input and the merged output before
training.

## Stage 5: Diagnostics Before Recommendations

Before Milestone 5, add diagnostics for:

- log loss by count, base-out state, pitch type, and relative zone
- rare outcome recall by outcome class
- home run recall
- expected calibration error
- profile coverage
- early-season shrinkage rate
- null rate for personalization features

Use these reports to decide whether the deterministic transition baseline is
ready for recommendation ranking or needs a Milestone 4.5 smoothing pass.

## Stage 6: Milestone 5 Recommendation Engine

The next feature milestone should evaluate every pitch type and every 13-zone
relative coordinate candidate for a target snapshot. Do not drop candidates
because of historical zone frequency or perceived reachability.

Expected future command shape:

```bash
uv run baseball-zerobase recommend-pitches \
  --snapshot data/processed/snapshots/role=dev_regular/snapshots_2022_regular_profiled.parquet \
  --model artifacts/models/transition/v0_2022_regular.json \
  --output reports/generated/recommendations/sample_recommendations.json
```

Recommendation outputs should include the ranked candidates, expected transition
value, and a compact explanation using pitcher ownership, batter weakness,
batter threat, and daily state features.
