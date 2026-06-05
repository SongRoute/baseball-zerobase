# GPU 학습 런북

이 문서는 GPU 서버에서 개발 전용 학습 파이프라인을 재현하기 위한 절차입니다.
Codex는 서버에서 명령을 실행하는 역할이며, 현재 Milestone 4 전이 baseline은
CPU에서도 실행됩니다. GPU 서버는 더 큰 데이터 준비, 이후 신경망 학습, 장시간
실험을 위한 실행 환경으로 사용합니다.

## GPU 서버 빠른 시작

서버를 받은 뒤 다음 명령을 실행합니다:

```bash
git clone git@github.com:SongRoute/baseball-zerobase.git
cd baseball-zerobase
uv sync
scripts/gpu_smoke.sh
scripts/build_dev_regular_dataset.sh 2022-04-07 2022-04-10
scripts/train_transition_baseline.sh \
  data/processed/dev_dataset/role=dev_regular/dev_dataset_2022-04-07_2022-04-10.parquet
```

`uv`가 없다면 먼저 설치합니다:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

서버에서 Codex CLI를 사용할 경우:

```bash
npm install -g @openai/codex
codex login --device-auth
codex --cd "$PWD" --sandbox workspace-write --ask-for-approval on-request
```

## 데이터 가드레일

개발 실행은 정규시즌 개발 데이터만 사용합니다. 학습 입력은 `data/locked` 아래에서
읽거나 쓰면 안 됩니다.

개발 중 절대 사용하지 않는 locked 파티션:

- `game_type`이 `F`, `D`, `L`, `W`인 2025 포스트시즌 경기
- `2026-03-25`부터 `2026-05-31`까지의 2026 정규시즌 경기

보조 스크립트는 명백한 locked 경로, postseason label, locked 2026 날짜를 거부합니다.
최종 dataset role 검증의 기준은 저장소의 validation 코드입니다.

## Stage 1: 서버 스모크

목표: 서버, Python 환경, repo 품질 게이트, 합성 파이프라인이 동작하는지 확인합니다.

```bash
scripts/gpu_smoke.sh
```

성공 기준:

- `uv sync` 완료
- `scripts/check.sh` 통과
- `uv run baseball-zerobase pipeline-smoke` 통과
- `nvidia-smi`가 있으면 GPU 진단 정보 출력

## Stage 2: 작은 개발 데이터셋

목표: full season 다운로드 전에 실제 데이터 수집과 profile-aware 학습이 동작하는지
확인합니다.

```bash
scripts/build_dev_regular_dataset.sh 2022-04-07 2022-04-10
scripts/train_transition_baseline.sh \
  data/processed/dev_dataset/role=dev_regular/dev_dataset_2022-04-07_2022-04-10.parquet
```

주요 산출물:

```text
data/work/game_pks_2022-04-07_2022-04-10.parquet
data/work/pitch_events_2022-04-07_2022-04-10.parquet
data/work/prepared_pitch_2022-04-07_2022-04-10.parquet
data/processed/snapshots/role=dev_regular/snapshots_2022-04-07_2022-04-10_profiled.parquet
data/processed/dev_dataset/role=dev_regular/dev_dataset_2022-04-07_2022-04-10.parquet
artifacts/models/transition/v0_dev_dataset_2022-04-07_2022-04-10.json
reports/generated/transition/v0_dev_dataset_2022-04-07_2022-04-10.json
```

dataset은 profiled snapshot에서 생성되므로 투수 프로필, 타자 약점, 타자 위협도,
당일 누적 상태 feature가 공유 전이모델 입력에 포함됩니다.

## Stage 3: 2022 전체 정규시즌

작은 개발 데이터셋이 성공하면 2022 정규시즌 범위를 생성합니다:

```bash
scripts/build_dev_regular_dataset.sh 2022-04-07 2022-10-05 2022_regular
scripts/train_transition_baseline.sh \
  data/processed/dev_dataset/role=dev_regular/dev_dataset_2022_regular.parquet \
  2022_regular
```

검토 대상:

- `reports/generated/validation/dev_dataset_2022_regular.json`
- `reports/generated/transition/v0_2022_regular.json`

이 단계는 네트워크 의존성과 실행 시간이 있으므로 자동 테스트에 넣지 않습니다.

## Stage 4: 2022-2024 확장

먼저 연도별 dataset을 독립적으로 생성합니다:

```bash
scripts/build_dev_regular_dataset.sh 2022-04-07 2022-10-05 2022_regular
scripts/build_dev_regular_dataset.sh 2023-03-30 2023-10-01 2023_regular
scripts/build_dev_regular_dataset.sh 2024-03-28 2024-09-30 2024_regular
```

다음 구현 단계는 연도별 dev dataset을 검증 후 병합하는 script를 추가하는 것입니다.
의도한 병합 산출물:

```text
data/processed/dev_dataset/role=dev_regular/dev_dataset_2022_2024_regular.parquet
reports/generated/validation/dev_dataset_2022_2024_regular.json
artifacts/models/transition/v0_2022_2024_regular.json
reports/generated/transition/v0_2022_2024_regular.json
```

병합 script는 연도별 입력과 병합 결과를 모두 validation한 뒤 학습으로 넘겨야 합니다.

## Stage 5: 추천 전 진단

Milestone 5 전에 다음 진단을 추가합니다:

- count, base-out state, pitch type, relative zone별 log loss
- outcome class별 rare outcome recall
- home run recall
- expected calibration error
- profile coverage
- early-season shrinkage rate
- 개인화 feature null rate

이 리포트로 결정론적 전이 baseline을 추천 ranking에 사용할 수 있는지, 아니면
Milestone 4.5 smoothing 보강이 필요한지 판단합니다.

## Stage 6: Milestone 5 추천 엔진

다음 feature milestone은 target snapshot에서 모든 pitch type과 13개 상대좌표 zone
후보를 평가해야 합니다. 과거 zone 빈도나 도달 가능성을 이유로 후보를 제거하지
않습니다.

예상 명령 형태:

```bash
uv run baseball-zerobase recommend-pitches \
  --snapshot data/processed/snapshots/role=dev_regular/snapshots_2022_regular_profiled.parquet \
  --model artifacts/models/transition/v0_2022_regular.json \
  --output reports/generated/recommendations/sample_recommendations.json
```

추천 결과에는 ranked candidates, expected transition value, 그리고 투수 구종 소유,
타자 약점, 타자 위협도, 당일 상태를 사용한 간결한 설명이 포함되어야 합니다.
