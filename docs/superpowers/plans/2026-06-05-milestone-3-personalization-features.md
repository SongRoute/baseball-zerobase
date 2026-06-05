# Milestone 3 Personalization Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add leakage-safe starter pitcher profiles, batter weakness archetypes, batter threat scores, and same-day state features for future shared transition models.

**Architecture:** Milestone 3 adds deterministic feature builders under `src/baseball_zerobase/profiles/`. Each builder accepts development snapshot rows, computes as-of features from earlier information only, and returns augmented rows for the existing dataset/validation pipeline. No independent pitcher models, neural training, recommendation ranking, locked-test execution, or zone reachability pruning is introduced.

**Tech Stack:** Python 3.12, Polars, Typer, pytest, Ruff, Pyright.

---

## English Plan

### Phase 1: Profile Package and Pitcher Profile Core

**Files:**
- Create: `src/baseball_zerobase/profiles/__init__.py`
- Create: `src/baseball_zerobase/profiles/pitcher.py`
- Create: `tests/profiles/test_pitcher.py`

- [x] Write failing tests for prior-only pitcher aggregation, same-game exclusion, owned pitch type thresholds, early-season shrinkage flags, and optional physical profile shrinkage.
- [x] Implement `add_pitcher_profiles(...)` as a pure transform.
- [x] Verify target rows never use the target pitch or later same-game pitches.
- [x] Run `uv run pytest tests/profiles/test_pitcher.py -q`.

### Phase 2: Batter Weakness Archetypes

**Files:**
- Create: `src/baseball_zerobase/profiles/batter_archetype.py`
- Create: `tests/profiles/test_batter_archetype.py`

- [x] Write failing tests that weakness uses response tendencies only, excludes current pitch, returns neutral for low sample, and separates chase/whiff/called-strike profiles from absolute production.
- [x] Implement deterministic archetypes: `neutral_unknown`, `chase_vulnerable`, `whiff_vulnerable`, `called_strike_vulnerable`, `balanced`.
- [x] Run `uv run pytest tests/profiles/test_batter_archetype.py -q`.

### Phase 3: Batter Threat Scores

**Files:**
- Create: `src/baseball_zerobase/profiles/batter_threat.py`
- Create: `tests/profiles/test_batter_threat.py`

- [x] Write failing tests for shrunk reach, extra-base, home-run, strikeout, and run-value proxy rates.
- [x] Implement `add_batter_threat(...)` with league-prior shrinkage and confidence based on prior plate appearances.
- [x] Confirm threat can use terminal outcome proxies while weakness cannot.
- [x] Run `uv run pytest tests/profiles/test_batter_threat.py -q`.

### Phase 4: Daily State and Feature Provenance Guards

**Files:**
- Create: `src/baseball_zerobase/profiles/daily_state.py`
- Create: `tests/profiles/test_daily_state.py`
- Modify: `src/baseball_zerobase/data/validation.py`
- Modify: `tests/data/test_validation.py`

- [x] Write failing tests for first-pitch zero state, prior-only same-day counts, no current-pitch outcome leakage, and feature timestamp validation.
- [x] Implement `add_daily_state(...)` from sorted snapshot rows using only rows with completed information before the target row.
- [x] Extend validation to reject profile/day-state timestamp columns that are null or not strictly before `pitch_timestamp`.
- [x] Run `uv run pytest tests/profiles/test_daily_state.py tests/data/test_validation.py -q`.

### Phase 5: CLI, Pipeline Smoke, and Documentation

**Files:**
- Modify: `src/baseball_zerobase/cli.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `README.md`
- Modify: `README.ko.md`

- [x] Write failing CLI smoke tests for `build-pitcher-profiles`, `build-batter-profiles`, and `build-daily-state`.
- [x] Implement commands using existing `require_dev_input(...)` path guards and immutable parquet writes with manifests.
- [x] Update pipeline smoke to include Milestone 3 feature transforms before starter eligibility and dataset building.
- [x] Update English README and Korean review README together.
- [x] Run `uv run pytest tests/test_cli_smoke.py -q` and `uv run baseball-zerobase pipeline-smoke`.

### Phase 6: Final Verification and Commit

**Files:**
- All Milestone 3 files.

- [x] Run `scripts/check.sh`.
- [x] Run CLI smoke commands on synthetic test parquet outputs.
- [x] Run `git diff --check`.
- [x] Review `git diff` directly for leakage, locked-data, action-space, and language-document drift.
- [x] Commit with `feat: add milestone 3 personalization features`.

---

## Korean Review / 한글 검토본

### 목표

Milestone 3는 다음 공유 전이모델에 들어갈 개인화 feature를 만든다. 투수별 독립 모델, 추천 랭킹, 복잡한 ML 학습, locked test 실행은 하지 않는다.

### 구현 단위

1. 선발투수 프로필: 이전 경기/이전 시즌 정보를 이용해 보유 구종, 구종별 표본, 사용률, 물리 특성 축소 추정값, 시즌 초 shrinkage flag를 만든다.
2. 타자 약점 유형: chase, whiff, called-strike 같은 반응 성향만 사용한다. 타율, 홈런, 장타 같은 절대 공격력은 넣지 않는다.
3. 타자 위협도: 출루/장타/홈런/삼진/간단한 run-value proxy를 축소 추정한다.
4. 당일 누적 상태: 현재 투구 전까지 같은 경기에서 관측된 투구 수, 결과 요약, 타자별 prior exposure를 만든다.
5. CLI와 문서: `build-pitcher-profiles`, `build-batter-profiles`, `build-daily-state`를 추가하고 README/README.ko를 함께 갱신한다.

### 누수 방지 원칙

- 모든 feature timestamp는 target `pitch_timestamp`보다 엄격히 이전이어야 한다.
- 장기 pitcher profile은 현재 경기 투구를 제외한다.
- daily state만 현재 경기의 이전 투구를 사용한다.
- current pitch의 구속, 움직임, 도착 위치, outcome은 target row feature에 넣지 않는다.
- locked partition과 `data/locked/`는 개발 중 읽지 않는다.

### 검증 기준

- 새 profile test가 target pitch 제외와 shrinkage를 확인한다.
- validation test가 `*_as_of_timestamp >= pitch_timestamp`를 거부한다.
- CLI smoke가 locked path guard와 재현 가능한 parquet 출력을 확인한다.
- `scripts/check.sh`, 관련 CLI smoke, `git diff --check`가 통과해야 한다.
