# AGENTS.md — baseball-zerobase

Codex는 작업 시작 전 이 파일을 자동으로 읽는다. 이 파일은 **진입점**이며, 상세는 가리키는 문서를 따른다.
(문서 중복 방지: 결정 전문을 여기 복붙하지 않는다. 항상 아래 문서를 실제로 열어 읽는다.)

## 작업 전 필독 (순서대로)

1. `DESIGN_DECISIONS.md` — 확정 결정 + **금지사항**. 금지사항은 절대 위반 금지.
2. 작업 대상 모듈의 `modules/<m>/AGENTS.md` (있으면) + `modules/<m>/README.md`.
3. 관련 시 `BASELINE.md`(SmartPitch 대비), `docs/glossary.md`(용어).

## 절대 규칙

- **결정·평가는 사람(Song)이 한다. 구현·실행·실험은 Codex가 한다.**
- 설계 결정을 "개선"이라며 임의로 바꾸지 않는다. 변경이 필요하면 코드를 고치기 전에 사람에게 먼저 확인한다.
- `DESIGN_DECISIONS.md`의 금지사항(절대 생산성 지표 금지, 투수→타자 종속 금지, 학습=서빙 입력 일치, 존 상대좌표, raw 보존, 단순분류기 회귀 금지)을 매 작업에서 점검한다.
- 작업 완료 시 해당 모듈의 `EXPERIMENTS.md`에 핵심 사실(설정·결과·이슈)을 한 줄 기록한다.

## 환경

- 머신: Mac Mini M4 (Apple Silicon), CPU 작업. GPU 작업은 Colab(별도).
- 패키지: uv. 프로젝트 경로: `~/projects/baseball-zerobase`. 데이터: Mac 로컬 디스크.
- 작업 디렉터리 구조: 코드는 `src/`, 데이터 `data/`, 결과 `results/`, 문서 `docs/`·`modules/`.

## 코드 규칙

- 설정 상수는 `config.py`에 집중. 하드코딩 금지.
- 커밋 전 `scripts/check.sh` 통과(ruff + 타입체크 + 테스트). pre-commit hook이 자동 실행.
- 기존 문서(`docs/`, `modules/`, 루트 `.md`)를 덮어쓰지 않는다. EXPERIMENTS.md 기록 추가만 허용.

## 서브에이전트 사용

- 독립 작업·실험만 병렬화. 의존성 있는 작업(데이터→클러스터→학습)은 순차.
- 커스텀 에이전트: `explorer`(읽기 탐색), `experiment-runner`(실험 배치), `reviewer`(검수 보조). 정의는 `.codex/agents/`.
- 대량 실험은 `spawn_agents_on_csv`로 후보를 행 단위 병렬(클러스터링 K, NN 구조 등).
