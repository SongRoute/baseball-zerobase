# Milestone 4 Shared Transition Model Design

- Date: 2026-06-05
- Status: Draft design for user review
- Scope: Milestone 4 only
- Prior milestones:
  - Milestone 1-2: data foundation, empirical baselines, inning simulator, rolling validation
  - Milestone 3: deterministic personalization features under `profiles/`

## 1. Goal

Milestone 4 builds the first shared transition-model layer for the recommendation
pipeline. The deliverable is not a recommendation engine. It is a reproducible,
testable model contract that predicts legal next-pitch transition distributions
from pre-pitch state, candidate action, pitcher profile features, batter
weakness/threat features, and daily-state features.

The implementation should be strong enough to evaluate component quality and feed
the inning simulator in later milestones, while remaining simple enough to audit
for leakage and baseball-state invariants.

## 2. Recommended Approach

Use a contract-first transition model v0.

The v0 model should expose the same interface expected from later hierarchical
models:

```text
predict_distribution(context, action) -> {TransitionAtom: probability}
sample(context, action, rng) -> TransitionAtom
log_probability(actual_atom, context, action) -> float
support(context, action) -> int
```

Internally, v0 can use smoothed empirical heads and calibration tables rather than
deep learning. This keeps Milestone 4 focused on:

- feature input contracts
- conditional head decomposition
- probability normalization
- impossible-transition masking
- calibration evaluation
- deterministic serialization
- simulator compatibility

PyTorch or LightGBM models can be introduced after these contracts are proven.

## 3. Non-Goals

Milestone 4 must not implement:

- pitch recommendation ranking
- `single-decision value`
- replay reports
- API or web UI
- locked-test execution
- policy improvement
- pitcher-specific independent production models
- candidate pruning by zone frequency or reachability

LightGBM and pitcher-specific models may be documented as future baselines, but
they should not be required to complete this milestone.

## 4. Inputs

The transition model uses only rows that pass the development dataset contract.
Every feature must be known before the target pitch:

- game state: balls, strikes, outs, runners, inning, score differential
- matchup state: batter stand, pitcher throws
- action: candidate `pitch_type × batter-relative 13-zone`
- pitcher personalization: Milestone 3 pitcher profile columns
- batter personalization: Milestone 3 weakness and threat columns
- day state: Milestone 3 same-game prior-state columns
- label: observed `TransitionAtom` built from the target pitch outcome

The label columns remain labels. They are not model inputs for the same target
row. In particular, target-row `outcome`, `*_after`, `runs_scored`,
`plate_appearance_ended`, and `half_inning_ended` are excluded from feature
vectors.

## 5. Conditional Heads

The approved product design names six baseball-process heads plus runner
advancement. Milestone 4 v0 should implement these as explicit contracts even if
some heads are backed by simple tables:

1. `Swing Head`
   - Domain: `swing`, `take`
   - Uses `outcome` mapping during training only.
2. `Contact Head`
   - Domain: `contact`, `no_contact`
   - Conditional on swing.
3. `Called Result Head`
   - Domain: `ball`, `called_strike`, `hit_by_pitch`
   - Conditional on no swing.
4. `Contact Result Head`
   - Domain: `foul`, `in_play`
   - Conditional on contact.
5. `Batted Ball Head`
   - Domain: `in_play_out`, `single`, `double`, `triple`, `home_run`, `reach_other`
   - Conditional on in-play contact.
6. `Plate Appearance Head`
   - Domain: `continues`, `walk`, `strikeout`, `terminal_in_play`, `hbp`, `reach_other`
   - Enforces count reset on terminal plate appearances.
7. `Runner Advancement Head`
   - Domain: legal `(outs_after, runners_after, runs_scored)` states.
   - Enforces no decrease in outs and no impossible base masks.

The public model output remains a distribution over full `TransitionAtom` values,
because the existing simulator consumes `TransitionAtom`.

## 6. Probability and State Invariants

Every prediction must satisfy:

- probabilities are finite
- probabilities are non-negative
- probabilities sum to 1 within a small tolerance
- impossible transitions have probability 0
- `outs_after >= outs`
- `outs_after <= 3`
- non-terminal pitches cannot decrease balls or strikes
- terminal plate appearances reset balls and strikes to 0
- half-inning terminal states are explicitly represented

If no legal transition has support, the model backs off to a broader legal
context. It must not return an empty distribution for a valid development action.

## 7. Backoff and Shrinkage

The model uses shared data before personalized contexts:

1. full personalized context and action
2. game-state plus action
3. count plus action
4. action
5. global legal transition distribution

Pitcher profile, batter archetype, batter threat, and daily-state features are
used as conditioning buckets in v0 rather than as continuous learned embeddings.
This gives the later shared model a stable contract without pretending that v0 is
already a neural model.

Shrinkage should be explicit:

```text
smoothed_count = observed_count + prior_weight * prior_probability
```

Prior sources are selected from broader backoff levels. Sparse rare outcomes
should be smoothed, not dropped.

## 8. Calibration

Milestone 4 adds component calibration reports before recommendation ranking.

Required metrics:

- log loss
- Brier score for binary heads
- expected calibration error
- head-level support counts
- rare outcome recall for home runs, extra-base hits, walks, strikeouts, and HBP

Calibration must run on rolling folds already supported by the repository. Locked
test partitions remain unavailable during development.

## 9. Serialization

The transition model must serialize deterministically to JSON or another stable
artifact with:

- model type and version
- training manifest hash
- feature column list
- head definitions
- smoothing settings
- backoff order
- calibration settings
- learned counts or parameters

Loading the artifact and predicting with the same input must produce the same
distribution.

## 10. CLI Surface

Recommended commands:

```text
fit-transition-model
evaluate-transition-model
```

`fit-transition-model` writes a model artifact and manifest. It accepts only
development-safe input paths and rejects `data/locked/`.

`evaluate-transition-model` writes component metrics and Korean summary text in
the generated report. It evaluates rolling or holdout development folds only.

## 11. Testing Requirements

Milestone 4 tests should cover:

- feature column exclusion for target labels
- probability sum invariant
- impossible-transition masking
- deterministic fit/serialize/load/predict
- legal sampling for simulator use
- rare-outcome smoothing
- calibration metric calculations
- rolling fold integration
- locked-path rejection
- no candidate pruning by zone frequency

## 12. Acceptance Criteria

Milestone 4 is complete when:

- a transition model artifact can be fit from development rows
- the model can produce legal `TransitionAtom` distributions for candidate actions
- the inning simulator can use the fitted model without changing the action-space contract
- component reports include calibration and rare-outcome metrics
- `scripts/check.sh` passes
- CLI smoke tests pass without network or locked data
- English user-facing docs and Korean review text are both maintained

## Korean Review / 한글 검토본

### 목표

Milestone 4는 추천 엔진이 아니라 공유 전이모델 계약을 만든다. 출력은 기존
시뮬레이터가 소비하는 `TransitionAtom` 확률분포이며, 모든 입력 feature는 target
pitch 이전 정보여야 한다.

### 추천 방향

처음부터 PyTorch 모델을 만들지 않고, 계약 우선 v0를 만든다. v0는 smoothed empirical
head와 backoff table을 사용해도 된다. 중요한 것은 확률 합, 불가능 전이 마스킹,
보정 지표, serialization, simulator 호환성을 먼저 검증하는 것이다.

### 범위 제외

추천 순위, replay report, API, web UI, locked test 실행, zone 빈도 기반 후보 제거,
투수별 독립 production model은 Milestone 4 범위가 아니다.

### 핵심 검증

- target pitch의 outcome과 transition label을 feature로 쓰지 않는다.
- 모든 feature timestamp는 target pitch보다 이전이다.
- 확률분포는 finite/non-negative/sum-to-1을 만족한다.
- 불가능한 상태 전이는 0 확률이다.
- rare outcome은 제거하지 않고 smoothing/backoff로 처리한다.
- development 중 `data/locked/`와 locked partition은 읽지 않는다.
