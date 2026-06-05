# Baseball Zerobase

Baseball Zerobase는 누수 방지 MLB 선발투수 전략 연구 파이프라인을 위한
클린룸 재작성 프로젝트입니다. 이 저장소는 이전 프로젝트의 코드, 모델 가중치,
또는 처리된 데이터를 사용하면 안 됩니다.

Milestone 1-4의 범위는 데이터 기반, 경험적 베이스라인, 결정론적 개인화 feature,
계약 우선 공유 전이모델을 포함합니다: 불변 원천 데이터 수집, 선발투수와 라인업
맥락, 투구 전 스냅샷, as-of 선발 자격, 투수 프로필, 타자 약점 유형, 타자 위협도,
당일 누적 상태, 경험적 행동 및 전이 베이스라인, 이닝 시뮬레이션, 롤링 검증,
합법 전이 확률분포를 포함합니다. 신경망 모델, 추천, API 서빙, 웹 UI는 범위
밖입니다.

## 데이터 파티션

개발 작업은 2022-2025 MLB 정규시즌 데이터만 사용합니다.

잠긴 파티션은 개발 중 절대 사용하지 않습니다:

- `game_type`이 `F`, `D`, `L`, `W`인 2025 포스트시즌 경기
- `2026-03-25`부터 `2026-05-31`까지의 2026 정규시즌 경기

개발 중에는 잠긴 테스트를 실행하거나 잠긴 데이터 경로를 읽으면 안 됩니다. 잠긴
평가는 별도의 최종 검토 단계에서만 사용합니다.

## 설치와 품질 검사

```bash
uv sync
uv run baseball-zerobase version
scripts/check.sh
```

`scripts/check.sh`는 필수 로컬 품질 게이트를 실행합니다:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright src tests
uv run pytest -q
```

## 스모크 테스트

빠른 로컬 검증에는 합성 데이터 파이프라인 스모크 명령을 사용합니다. 이 명령은
네트워크 데이터를 다운로드하지 않으며 자동화 테스트에서 안전하게 사용할 수
있습니다.

```bash
uv run baseball-zerobase pipeline-smoke
```

명령은 전이 음의 로그 가능도와 시뮬레이션 절단률을 포함한 베이스라인 지표
요약을 출력합니다.

## 작은 범위 수집 예시

네트워크 수집은 수동 개발 작업이며 자동화 테스트에 포함하지 않습니다. 먼저 작은
날짜 범위로 실행하세요:

```bash
uv run baseball-zerobase download-statcast \
  --start 2022-04-07 \
  --end 2022-04-10
```

이 명령은 다음과 같은 불변 개발 파티션을 씁니다:

```text
data/raw/statcast/start=2022-04-07_end=2022-04-10.parquet
```

개발 전용 `game_pk` 목록을 로컬 parquet로 추출한 뒤, 대응하는 MLB StatsAPI 경기
피드를 내려받습니다:

```bash
uv run baseball-zerobase download-games \
  --game-pks-parquet data/work/game_pks_2022-04-07_2022-04-10.parquet
```

풀시즌 다운로드는 오래 걸리고 네트워크에 의존하는 작업입니다. 자동화 테스트의
일부가 되어서는 안 됩니다.

## 파이프라인 명령

저장되는 개발 파이프라인 순서는 다음과 같습니다:

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
```

`prepared_pitch.parquet`에는 선발투수와 안정 라인업 맥락이 연결된 개발 정규시즌
행만 들어 있어야 합니다. 모든 경로는 `data/locked` 밖에 두세요.

## 개인화 Feature

Milestone 3 feature builder는 개발 snapshot 위에서 동작하는 결정론적 변환입니다.
투수 프로필은 target 경기를 제외하고 이전 2개 시즌 투구 이력을 사용하며, 시즌 초
축소 추정 flag를 기록합니다. 타자 약점 유형은 chase, whiff, called strike 같은
반응 성향만 사용합니다. 타자 위협도는 출루, 장타, 홈런, 삼진 같은 타석 종료 outcome
proxy를 사용합니다. 당일 누적 상태는 target pitch 이전의 같은 경기 행만 사용합니다.

각 Milestone 3 feature family는 `*_as_of_timestamp` 열을 씁니다. 검증은 feature
timestamp가 비어 있거나 target `pitch_timestamp`보다 엄격히 이전이 아니면 거부합니다.

## 공유 전이모델

Milestone 4는 결정론적 공유 전이모델 v0를 추가합니다. 이 모델은 개발 전용 전이
count를 smoothing하고, target label column을 model feature에서 제외하며, 정규화된
합법 `TransitionAtom` 확률분포를 출력하고, 이닝 시뮬레이터에서 직접 사용할 수
있습니다. 구성요소 평가는 전이 log loss, calibration error, rare outcome recall,
한글 요약을 보고합니다.

## GPU 학습 준비

원격 서버에서 개발 전용 dataset 생성과 전이 baseline 학습을 준비할 때는 GPU 학습
런북을 사용합니다:

- 영어: `docs/gpu-training-runbook.md`
- 한글 검토본: `docs/gpu-training-runbook.ko.md`

런북에는 서버 스모크, 작은 실제 데이터 스모크, 2022 전체 시즌 확장, 2022-2024 확장,
진단, Milestone 5 추천 준비 단계가 포함되어 있습니다.

## 언어 검토

사용자 대상 영어 Markdown 문서에는 한국어 검토용 대응 문서가 있어야 합니다.
`README.md`와 `README.ko.md`는 항상 함께 유지하세요.
