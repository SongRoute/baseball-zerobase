# Milestone 4 Shared Transition Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a contract-first shared transition model v0 that emits legal `TransitionAtom` probability distributions for candidate `pitch_type × batter-relative 13-zone` actions.

**Architecture:** Milestone 4 introduces `src/baseball_zerobase/models/` with small, testable modules for context extraction, head labels, legal-transition masking, smoothed transition counts, calibration metrics, and model artifacts. The v0 implementation uses deterministic smoothed empirical heads and backoff buckets so the simulator and evaluation pipeline can validate probability contracts before neural or LightGBM models are introduced.

**Tech Stack:** Python 3.12, Polars, Pydantic dataclasses/contracts, NumPy, Typer, pytest, Ruff, Pyright.

---

## Planned File Structure

```text
src/baseball_zerobase/
  models/
    __init__.py
    transition.py
    transition_context.py
    transition_heads.py
    transition_artifact.py
    calibration.py
  cli.py
  evaluation/
    transition.py
tests/
  models/
    test_transition_context.py
    test_transition_heads.py
    test_transition_model.py
    test_transition_artifact.py
    test_calibration.py
  evaluation/
    test_transition_evaluation.py
  test_cli_smoke.py
README.md
README.ko.md
```

## Phase 1: Transition Context and Feature Exclusion

**Files:**
- Create: `src/baseball_zerobase/models/__init__.py`
- Create: `src/baseball_zerobase/models/transition_context.py`
- Create: `tests/models/test_transition_context.py`

- [ ] **Step 1: Write failing tests**

```python
def test_context_excludes_target_label_columns() -> None:
    frame = pl.DataFrame({...})
    context = transition_context_from_row(frame.row(0, named=True))
    assert "outcome" not in context.features
    assert "runs_scored" not in context.features
    assert "balls_after" not in context.features
    assert context.pitch_type == "FF"
    assert context.relative_zone == "middle_middle"
```

Run:

```bash
uv run pytest tests/models/test_transition_context.py -q
```

Expected: FAIL because `baseball_zerobase.models.transition_context` does not exist.

- [ ] **Step 2: Implement minimal context extraction**

Create a frozen `TransitionContext` dataclass with:

```python
pitch_type: str
relative_zone: str
balls: int
strikes: int
outs: int
runners: int
stand: str | None
p_throws: str | None
features: Mapping[str, object]
```

`transition_context_from_row(row)` must exclude label columns:

```python
LABEL_COLUMNS = {
    "outcome",
    "balls_after",
    "strikes_after",
    "outs_after",
    "runners_after",
    "runs_scored",
    "plate_appearance_ended",
    "half_inning_ended",
    "terminal_reason",
}
```

- [ ] **Step 3: Verify Phase 1**

```bash
uv run pytest tests/models/test_transition_context.py -q
```

Expected: PASS.

## Phase 2: Head Labels and Legal Transition Masks

**Files:**
- Create: `src/baseball_zerobase/models/transition_heads.py`
- Create: `tests/models/test_transition_heads.py`

- [ ] **Step 1: Write failing tests**

```python
def test_head_labels_map_swinging_strike_to_swing_no_contact() -> None:
    labels = head_labels_from_row({"outcome": "swinging_strike"})
    assert labels.swing == "swing"
    assert labels.contact == "no_contact"


def test_legal_transition_mask_rejects_decreasing_outs() -> None:
    atom = TransitionAtom(...)
    assert not is_legal_transition(context, atom)
```

Run:

```bash
uv run pytest tests/models/test_transition_heads.py -q
```

Expected: FAIL because `transition_heads.py` does not exist.

- [ ] **Step 2: Implement head labels**

Create a frozen `TransitionHeadLabels` dataclass with fields:

```python
swing: str
contact: str | None
called_result: str | None
contact_result: str | None
batted_ball: str | None
plate_appearance: str
```

Implement `head_labels_from_row(row)` using the existing `OutcomeLabel` values.

- [ ] **Step 3: Implement legal transition checks**

Implement:

```python
def is_legal_transition(context: TransitionContext, atom: TransitionAtom) -> bool:
    ...
```

It must enforce probability-support invariants from the design:
`outs_after >= outs`, `outs_after <= 3`, non-terminal count monotonicity, terminal
count reset, and valid runner tuple length.

- [ ] **Step 4: Verify Phase 2**

```bash
uv run pytest tests/models/test_transition_heads.py -q
```

Expected: PASS.

## Phase 3: Smoothed Shared Transition Model v0

**Files:**
- Create: `src/baseball_zerobase/models/transition.py`
- Create: `tests/models/test_transition_model.py`

- [ ] **Step 1: Write failing tests**

```python
def test_transition_model_predicts_normalized_legal_distribution() -> None:
    model = SharedTransitionModelV0(min_support=1, prior_weight=1.0)
    model.fit(training_frame(), training_manifest_hash="synthetic:m4")
    distribution = model.predict_distribution(context_from_target())
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert all(probability >= 0 for probability in distribution.values())
    assert all(is_legal_transition(context_from_target(), atom) for atom in distribution)
```

Run:

```bash
uv run pytest tests/models/test_transition_model.py -q
```

Expected: FAIL because `SharedTransitionModelV0` does not exist.

- [ ] **Step 2: Implement model state**

Create `SharedTransitionModelV0` with:

```python
min_support: int = 1
prior_weight: float = 1.0
training_manifest_hash: str | None = None
```

Use backoff keys in this order:

```text
personalized_action
state_action
count_action
action
global
```

Personalized buckets should use coarse existing feature values such as
`pitcher_pitch_type_owned`, `batter_weakness_archetype`, and binned
`batter_threat_score`.

- [ ] **Step 3: Implement fit and prediction**

`fit(frame, training_manifest_hash=...)` collects legal `TransitionAtom` counts.
`predict_distribution(context)` selects the first supported backoff level and
smooths with the broader level. It must never drop rare legal outcomes because
they are sparse.

- [ ] **Step 4: Implement sampling and log probability**

Expose:

```python
sample(rng, context) -> TransitionAtom
log_probability(actual_atom, context) -> float
support(context) -> int
```

Use deterministic key ordering so serialization and tests are stable.

- [ ] **Step 5: Verify Phase 3**

```bash
uv run pytest tests/models/test_transition_model.py -q
```

Expected: PASS.

## Phase 4: Artifact Serialization and Loading

**Files:**
- Create: `src/baseball_zerobase/models/transition_artifact.py`
- Create: `tests/models/test_transition_artifact.py`

- [ ] **Step 1: Write failing tests**

```python
def test_transition_artifact_round_trip_preserves_predictions(tmp_path: Path) -> None:
    model = fitted_model()
    path = tmp_path / "transition_model.json"
    write_transition_artifact(model, path)
    loaded = read_transition_artifact(path)
    assert loaded.predict_distribution(context()) == model.predict_distribution(context())
```

Run:

```bash
uv run pytest tests/models/test_transition_artifact.py -q
```

Expected: FAIL because artifact helpers do not exist.

- [ ] **Step 2: Implement deterministic artifact helpers**

Artifact JSON must include:

```text
model_type
version
training_manifest_hash
feature_columns
backoff_order
smoothing_settings
counts
```

Write JSON with `sort_keys=True` and stable separators.

- [ ] **Step 3: Verify Phase 4**

```bash
uv run pytest tests/models/test_transition_artifact.py -q
```

Expected: PASS.

## Phase 5: Calibration and Component Evaluation

**Files:**
- Create: `src/baseball_zerobase/models/calibration.py`
- Create: `src/baseball_zerobase/evaluation/transition.py`
- Create: `tests/models/test_calibration.py`
- Create: `tests/evaluation/test_transition_evaluation.py`

- [ ] **Step 1: Write failing metric tests**

```python
def test_expected_calibration_error_groups_probabilities() -> None:
    value = expected_calibration_error(
        probabilities=[0.1, 0.8],
        outcomes=[False, True],
        bins=2,
    )
    assert value >= 0
```

Run:

```bash
uv run pytest tests/models/test_calibration.py -q
```

Expected: FAIL because calibration metrics do not exist.

- [ ] **Step 2: Implement metrics**

Implement log loss, Brier score, expected calibration error, support counts, and
rare-outcome recall for home runs, extra-base hits, walks, strikeouts, and HBP.

- [ ] **Step 3: Write failing component evaluation test**

```python
def test_evaluate_transition_model_reports_korean_summary() -> None:
    report = evaluate_transition_model(fitted_model(), validation_frame())
    assert report.row_count > 0
    assert "korean_summary" in report.to_dict()
```

Run:

```bash
uv run pytest tests/evaluation/test_transition_evaluation.py -q
```

Expected: FAIL because transition evaluation does not exist.

- [ ] **Step 4: Implement evaluation report**

Create a frozen report dataclass with `to_dict()`. Include English metric fields
and a concise Korean summary string for generated report compatibility.

- [ ] **Step 5: Verify Phase 5**

```bash
uv run pytest tests/models/test_calibration.py tests/evaluation/test_transition_evaluation.py -q
```

Expected: PASS.

## Phase 6: CLI, Documentation, and Final Verification

**Files:**
- Modify: `src/baseball_zerobase/cli.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `README.md`
- Modify: `README.ko.md`

- [ ] **Step 1: Write failing CLI tests**

```python
def test_transition_model_cli_fit_and_evaluate(tmp_path: Path) -> None:
    dataset = tmp_path / "data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet"
    write_transition_training_frame(dataset)
    model_path = tmp_path / "artifacts/models/transition/v0.json"
    report_path = tmp_path / "reports/generated/transition/v0.json"
    fit_result = CliRunner().invoke(app, ["fit-transition-model", "--dataset", str(dataset), "--output", str(model_path)])
    assert fit_result.exit_code == 0
    eval_result = CliRunner().invoke(app, ["evaluate-transition-model", "--dataset", str(dataset), "--model", str(model_path), "--report", str(report_path)])
    assert eval_result.exit_code == 0
```

Run:

```bash
uv run pytest tests/test_cli_smoke.py::test_transition_model_cli_fit_and_evaluate -q
```

Expected: FAIL because the commands do not exist.

- [ ] **Step 2: Implement CLI commands**

Add:

```text
fit-transition-model
evaluate-transition-model
```

Both commands must use `require_dev_input(...)`, reject locked paths, avoid
network access, and write deterministic artifacts/reports.

- [ ] **Step 3: Update docs**

Update README and README.ko together with the new commands and Milestone 4 scope.

- [ ] **Step 4: Final verification**

Run:

```bash
uv run pytest tests/models tests/evaluation/test_transition_evaluation.py tests/test_cli_smoke.py -q
uv run baseball-zerobase pipeline-smoke
scripts/check.sh
git diff --check
```

Expected: all commands PASS.

## Korean Review / 한글 검토본

### 목표

Milestone 4는 추천 엔진이 아니라 공유 전이모델 v0를 만든다. 핵심 산출물은 후보
행동에 대해 합법적인 `TransitionAtom` 확률분포를 내는 모델 계약, artifact,
component evaluation, CLI이다.

### Phase 요약

1. 전이모델 context를 만들고 target label column을 feature에서 제외한다.
2. swing/contact/called/contact-result/batted-ball/PA/runner head label과 legal mask를 만든다.
3. smoothed empirical shared transition model v0를 구현한다.
4. deterministic artifact 저장/로드를 만든다.
5. calibration과 rare outcome component evaluation을 만든다.
6. CLI와 README/README.ko를 갱신하고 전체 검증을 통과시킨다.

### 주의사항

- locked data와 `/Users/song/Projects/baseball`은 사용하지 않는다.
- target pitch 이후 정보는 feature에 넣지 않는다.
- zone 빈도나 도달 가능성으로 후보를 제거하지 않는다.
- rare outcome은 제거하지 않고 smoothing/backoff로 처리한다.
- PyTorch/LightGBM은 이번 v0 이후에 얹을 수 있다.
