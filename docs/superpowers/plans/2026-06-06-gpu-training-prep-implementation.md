# GPU Training Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible scripts and bilingual runbooks so a GPU server can prepare dev-only datasets and run transition baseline training with a few commands.

**Architecture:** Keep the GPU preparation surface as shell scripts under `scripts/` and user-facing documentation under `docs/`. The scripts call existing `baseball-zerobase` CLI commands, enforce development-only path guards, and avoid adding new model logic.

**Tech Stack:** Bash, Python 3.12 via `uv`, Polars, existing Typer CLI, pytest contract tests.

---

### Task 1: Script And Runbook Contracts

**Files:**
- Create: `tests/test_gpu_training_prep.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_prep_scripts_exist_and_are_executable() -> None:
    for path in (
        ROOT / "scripts/gpu_smoke.sh",
        ROOT / "scripts/build_dev_regular_dataset.sh",
        ROOT / "scripts/train_transition_baseline.sh",
    ):
        assert path.exists()
        assert path.stat().st_mode & 0o111


def test_gpu_prep_scripts_refuse_locked_paths() -> None:
    for path in (
        ROOT / "scripts/gpu_smoke.sh",
        ROOT / "scripts/build_dev_regular_dataset.sh",
        ROOT / "scripts/train_transition_baseline.sh",
    ):
        text = path.read_text(encoding="utf-8")
        assert "data/locked" in text
        assert "reject_locked_path" in text or "reject_locked_args" in text


def test_gpu_training_runbooks_are_bilingual_and_include_next_stage() -> None:
    english = ROOT / "docs/gpu-training-runbook.md"
    korean = ROOT / "docs/gpu-training-runbook.ko.md"
    assert english.exists()
    assert korean.exists()
    assert "GPU Server Quickstart" in english.read_text(encoding="utf-8")
    assert "GPU 서버 빠른 시작" in korean.read_text(encoding="utf-8")
    assert "2022-2024" in english.read_text(encoding="utf-8")
    assert "2022-2024" in korean.read_text(encoding="utf-8")
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_gpu_training_prep.py -q`

Expected: FAIL because the scripts and runbooks do not exist.

### Task 2: Add GPU Preparation Scripts

**Files:**
- Create: `scripts/gpu_smoke.sh`
- Create: `scripts/build_dev_regular_dataset.sh`
- Create: `scripts/train_transition_baseline.sh`

- [ ] **Step 1: Implement script guards and command flow**

`scripts/gpu_smoke.sh` should:
- run from the repository root
- reject arguments containing `data/locked`, 2025 postseason labels, or the 2026 locked date range
- run `uv sync`, `scripts/check.sh`, `uv run baseball-zerobase pipeline-smoke`
- print GPU diagnostics when `nvidia-smi` is available

`scripts/build_dev_regular_dataset.sh` should:
- accept `START_DATE END_DATE [LABEL]`
- reject locked dates and locked path tokens
- run download-statcast and derive `game_pks` from the downloaded parquet
- run download-games
- merge normalized `pitch_events.parquet` files
- attach starter/lineup context with existing `attach_starter_and_lineup_context`
- run snapshot/profile/dataset/validation CLI commands

`scripts/train_transition_baseline.sh` should:
- accept `DATASET_PARQUET [LABEL]`
- reject locked paths
- run transition fit/eval
- write artifacts and reports with the label

- [ ] **Step 2: Verify GREEN for script contracts**

Run: `uv run pytest tests/test_gpu_training_prep.py -q`

Expected: PASS.

### Task 3: Add Bilingual Runbooks

**Files:**
- Create: `docs/gpu-training-runbook.md`
- Create: `docs/gpu-training-runbook.ko.md`
- Modify: `README.md`
- Modify: `README.ko.md`

- [ ] **Step 1: Document server quickstart**

Include:
- Codex CLI setup reminder
- `uv` installation check
- GPU diagnostics
- small-range smoke commands
- 2022 full regular season expansion
- 2022-2024 merge/training roadmap
- locked data restrictions

- [ ] **Step 2: Link from README and Korean README**

Add a short section pointing to the English and Korean runbooks.

### Task 4: Final Verification And Commit

**Files:**
- All files above

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/test_gpu_training_prep.py -q`

Expected: PASS.

- [ ] **Step 2: Run project gate**

Run: `scripts/check.sh`

Expected: PASS.

- [ ] **Step 3: Run shell syntax checks**

Run: `bash -n scripts/gpu_smoke.sh scripts/build_dev_regular_dataset.sh scripts/train_transition_baseline.sh`

Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add docs/gpu-training-runbook.md docs/gpu-training-runbook.ko.md \
  docs/superpowers/plans/2026-06-06-gpu-training-prep-implementation.md \
  README.md README.ko.md \
  scripts/gpu_smoke.sh scripts/build_dev_regular_dataset.sh scripts/train_transition_baseline.sh \
  tests/test_gpu_training_prep.py
git commit -m "chore: prepare gpu training workflow"
```
