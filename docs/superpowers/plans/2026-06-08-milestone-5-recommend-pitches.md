# Milestone 5 Recommend Pitches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leakage-safe `recommend-pitches` CLI that ranks candidate `pitch_type x 13 relative_zone` actions with a shared transition model artifact.

**Architecture:** Add a small `inference/` package that owns candidate-grid generation, serving-row guards, candidate context construction, transparent transition-risk scoring, and JSON report serialization. The CLI stays thin: it loads a profiled snapshot or development dataset row, loads `SharedTransitionModelV0`, resolves pitch types, calls the recommender, and writes ranked recommendations.

**Tech Stack:** Python 3.12, Polars, Typer, Pydantic transition contracts, pytest, Ruff, Pyright.

---

## Planned File Structure

```text
src/baseball_zerobase/
  cli.py
  inference/
    __init__.py
    candidates.py
    recommender.py
    schemas.py
tests/
  inference/
    test_candidates.py
    test_recommender.py
  test_cli_smoke.py
docs/superpowers/plans/
  2026-06-08-milestone-5-recommend-pitches.md
README.md
README.ko.md
```

## Contract Decisions

- Candidate zones are always `tuple(zone.value for zone in RelativeZone)`. No zone is removed for historical frequency, perceived reachability, command skill, or support.
- Candidate pitch types come from `--pitch-types` first, then a prior-only list column such as `pitcher_owned_pitch_types` or `eligible_pitch_types`, then the row `pitch_type` as a single-action fallback for smoke use.
- Candidate contexts overwrite the row action with the candidate `pitch_type` and `relative_zone`.
- Target label columns stay excluded through `transition_context_from_row`.
- Candidate `pitcher_pitch_type_owned` is recomputed from `pitcher_owned_pitch_types` when that prior-only list is present. The recommender does not reuse an actual-row owned flag across all candidates.
- Raw current-pitch measurement columns are rejected at serving: `zone`, `plate_x`, `plate_z`, `release_speed`, `pfx_x`, `pfx_z`, `release_pos_x`, `release_pos_z`, and `release_extension`.
- Row-level physical profile means that were computed for the actual target pitch type are dropped from candidate contexts unless a future as-of profile map is provided. Milestone 5 v0 therefore uses ownership, batter archetype, threat, count/state, handedness, and daily-state features that are not candidate physical measurements.
- The v0 score is an immediate transition-proxy score, not simulated expected inning runs:

```text
score =
  expected_runs_scored
  + p_unfavorable
  - p_favorable
```

`p_favorable` covers atoms that add outs, advance strikes, or end the half-inning.
`p_unfavorable` covers atoms that score runs, add ball progress, or produce on-base
damage such as singles, extra-base hits, walks, HBP, and reach-other outcomes. Lower
score ranks better for the pitcher. The report includes the score parts, support,
top transition atoms, strike/ball/reach/home-run probabilities, pitch-type ownership
evidence, model artifact hash, `value_type: "transition_proxy"`, and a `zone_filtering`
field set to `disabled`.

## Task 1: Failing Recommendation Tests

**Files:**
- Create: `tests/inference/test_candidates.py`
- Create: `tests/inference/test_recommender.py`

- [ ] **Step 1: Write candidate-grid test**

```python
def test_candidate_grid_uses_every_relative_zone_for_each_pitch_type() -> None:
    candidates = generate_candidate_grid(["FF", "SL"])

    assert len(candidates) == 2 * len(RelativeZone)
    assert {candidate.relative_zone for candidate in candidates} == {
        zone.value for zone in RelativeZone
    }
```

- [ ] **Step 2: Write full-grid recommendation test**

```python
def test_recommendations_score_full_pitch_type_zone_grid_without_zone_filtering() -> None:
    model = SharedTransitionModelV0(min_support=1, prior_weight=0.0).fit(
        _transition_training_frame(),
        training_manifest_hash="synthetic:m5",
    )

    report = recommend_pitches(
        model,
        _target_row(),
        pitch_types=["FF", "SL"],
        top_k=None,
    )

    assert report.candidate_count == 26
    assert report.zone_filtering == "disabled"
    assert {item.pitch_type for item in report.recommendations} == {"FF", "SL"}
    assert {
        item.relative_zone for item in report.recommendations if item.pitch_type == "FF"
    } == {zone.value for zone in RelativeZone}
```

- [ ] **Step 3: Write scoring and leakage guard tests**

```python
def test_transition_risk_score_prefers_strike_distribution_to_home_run_distribution() -> None:
    model = SharedTransitionModelV0(min_support=1, prior_weight=0.0).fit(
        _transition_training_frame(),
        training_manifest_hash="synthetic:m5",
    )

    report = recommend_pitches(model, _target_row(), pitch_types=["FF", "SL"], top_k=None)
    ff_middle = _find(report, "FF", "middle_middle")
    sl_middle = _find(report, "SL", "middle_middle")

    assert ff_middle.ranking_score < sl_middle.ranking_score


def test_candidate_owned_flag_is_recomputed_from_prior_pitch_type_list() -> None:
    model = SharedTransitionModelV0(min_support=1, prior_weight=0.0).fit(
        _transition_training_frame(),
        training_manifest_hash="synthetic:m5",
    )
    row = {**_target_row(), "pitch_type": "SL", "pitcher_pitch_type_owned": True}

    report = recommend_pitches(model, row, pitch_types=["FF", "SL"], top_k=None)

    assert _find(report, "FF", "middle_middle").explanation["pitcher_pitch_type_owned"] is True
    assert _find(report, "SL", "middle_middle").explanation["pitcher_pitch_type_owned"] is False


def test_recommendations_reject_feature_timestamps_at_or_after_target_pitch() -> None:
    model = SharedTransitionModelV0(min_support=1, prior_weight=0.0).fit(
        _transition_training_frame(),
        training_manifest_hash="synthetic:m5",
    )
    row = {**_target_row(), "batter_profile_as_of_timestamp": _target_row()["pitch_timestamp"]}

    with pytest.raises(ValueError, match="must be before pitch_timestamp"):
        recommend_pitches(model, row, pitch_types=["FF"], top_k=None)


def test_target_labels_and_actual_action_do_not_change_recommendations() -> None:
    model = SharedTransitionModelV0(min_support=1, prior_weight=0.0).fit(
        _transition_training_frame(),
        training_manifest_hash="synthetic:m5",
    )
    original = recommend_pitches(model, _target_row(), pitch_types=["FF", "SL"], top_k=None)
    mutated = recommend_pitches(
        model,
        {
            **_target_row(),
            "pitch_type": "CU",
            "relative_zone": "chase_high",
            "outcome": "home_run",
            "runs_scored": 4,
            "balls_after": 0,
            "strikes_after": 0,
            "outs_after": 0,
            "runners_after": 0,
            "plate_appearance_ended": True,
            "half_inning_ended": False,
        },
        pitch_types=["FF", "SL"],
        top_k=None,
    )

    assert mutated.to_dict()["recommendations"] == original.to_dict()["recommendations"]


@pytest.mark.parametrize("column", ["zone", "plate_x", "plate_z", "release_speed", "pfx_x"])
def test_recommendations_reject_raw_current_pitch_measurements(column: str) -> None:
    model = SharedTransitionModelV0(min_support=1, prior_weight=0.0).fit(
        _transition_training_frame(),
        training_manifest_hash="synthetic:m5",
    )
    row = {**_target_row(), column: 1}

    with pytest.raises(ValueError, match="serving input cannot include current-pitch measurement"):
        recommend_pitches(model, row, pitch_types=["FF"], top_k=None)
```

- [ ] **Step 4: Verify red**

Run:

```bash
uv run pytest tests/inference/test_candidates.py tests/inference/test_recommender.py -q
```

Expected: FAIL because `baseball_zerobase.inference` does not exist.

## Task 2: Minimal Inference Package

**Files:**
- Create: `src/baseball_zerobase/inference/__init__.py`
- Create: `src/baseball_zerobase/inference/candidates.py`
- Create: `src/baseball_zerobase/inference/recommender.py`
- Create: `src/baseball_zerobase/inference/schemas.py`

- [ ] **Step 1: Implement dataclasses and candidate generation**

```python
@dataclass(frozen=True, slots=True)
class PitchCandidate:
    pitch_type: str
    relative_zone: str


def generate_candidate_grid(pitch_types: Iterable[str]) -> tuple[PitchCandidate, ...]:
    unique_pitch_types = _normalize_pitch_types(pitch_types)
    return tuple(
        PitchCandidate(pitch_type, zone.value)
        for pitch_type in unique_pitch_types
        for zone in RelativeZone
    )
```

- [ ] **Step 2: Implement scoring and report serialization**

```python
def recommend_pitches(
    model: SharedTransitionModelV0,
    row: Mapping[str, Any],
    *,
    pitch_types: Iterable[str] | None = None,
    top_k: int | None = 10,
    top_outcomes: int = 5,
) -> RecommendationReport:
    ...
```

Each candidate builds a `TransitionContext`, calls `model.predict_distribution(context)`,
scores the full distribution, and sorts by `(ranking_score, pitch_type, relative_zone)`.

- [ ] **Step 3: Verify green**

Run:

```bash
uv run pytest tests/inference/test_candidates.py tests/inference/test_recommender.py -q
```

Expected: PASS.

## Task 3: Failing CLI Smoke Test

**Files:**
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write CLI test**

```python
def test_recommend_pitches_cli_writes_ranked_recommendations(tmp_path) -> None:
    dataset = tmp_path / "data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet"
    dataset.parent.mkdir(parents=True)
    _transition_cli_frame().write_parquet(dataset)
    model_path = tmp_path / "artifacts/models/transition/v0.json"
    output_path = tmp_path / "reports/generated/recommendations/pitch.json"
    config = tmp_path / "configs/base.yaml"

    fit_result = CliRunner().invoke(...)
    assert fit_result.exit_code == 0

    recommend_result = CliRunner().invoke(
        app,
        [
            "recommend-pitches",
            "--input",
            str(dataset),
            "--model",
            str(model_path),
            "--pitch-types",
            "FF,SL",
            "--row-index",
            "0",
            "--top-k",
            "5",
            "--output",
            str(output_path),
            "--config",
            str(config),
        ],
    )

    assert recommend_result.exit_code == 0, recommend_result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 26
    assert payload["zone_filtering"] == "disabled"
    assert len(payload["recommendations"]) == 5
```

- [ ] **Step 2: Verify red**

Run:

```bash
uv run pytest tests/test_cli_smoke.py::test_recommend_pitches_cli_writes_ranked_recommendations -q
```

Expected: FAIL because the CLI command does not exist.

## Task 4: CLI Implementation

**Files:**
- Modify: `src/baseball_zerobase/cli.py`

- [ ] **Step 1: Add imports**

```python
from baseball_zerobase.inference.recommender import recommend_pitches
```

- [ ] **Step 2: Add `recommend-pitches` command**

```python
@app.command("recommend-pitches")
def recommend_pitches_command(
    input_path: Path = typer.Option(..., "--input"),
    model_path: Path = typer.Option(..., "--model"),
    output: Path | None = typer.Option(None, "--output"),
    row_index: int = typer.Option(0, min=0),
    pitch_types: str | None = typer.Option(None, "--pitch-types"),
    top_k: int | None = typer.Option(10, min=1),
    top_outcomes: int = typer.Option(5, min=1),
    config: Path = typer.Option(Path("configs/base.yaml")),
) -> None:
    ...
```

The command reads `.parquet` or `.json`, rejects locked data paths with `require_dev_input`,
loads the model artifact, emits JSON to `--output` or stdout, and reports the output path.

- [ ] **Step 3: Verify CLI green**

Run:

```bash
uv run pytest tests/test_cli_smoke.py::test_recommend_pitches_cli_writes_ranked_recommendations -q
```

Expected: PASS.

## Task 5: Docs and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `README.ko.md`

- [ ] **Step 1: Update README commands and Milestone text**

Add `recommend-pitches` after `evaluate-transition-model` and document:

```bash
uv run baseball-zerobase recommend-pitches \
  --input data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet \
  --model artifacts/models/transition/v0.json \
  --pitch-types FF,SL,CH \
  --row-index 0 \
  --top-k 10 \
  --output reports/generated/recommendations/pitch.json
```

- [ ] **Step 2: Run quality gates**

```bash
uv run pytest tests/inference/test_candidates.py tests/inference/test_recommender.py tests/test_cli_smoke.py -q
scripts/check.sh
git diff --check
```

Expected: all commands PASS.

- [ ] **Step 3: Review and commit**

Review:

```bash
git status --short
git diff -- src tests README.md README.ko.md docs/superpowers/plans/2026-06-08-milestone-5-recommend-pitches.md
```

Commit:

```bash
git add src/baseball_zerobase/inference src/baseball_zerobase/cli.py tests/inference tests/test_cli_smoke.py README.md README.ko.md docs/superpowers/plans/2026-06-08-milestone-5-recommend-pitches.md
git commit -m "feat: add pitch recommendation CLI"
```
