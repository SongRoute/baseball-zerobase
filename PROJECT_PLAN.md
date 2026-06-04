# PROJECT_PLAN

> baseball-zerobase — MLB 투구 추천 서비스
> 이 문서는 **최상위 인덱스**다. 세부 내용은 각 문서를 가리키기만 하고, 여기서 중복 보유하지 않는다.
> 상세 결정은 `DESIGN_DECISIONS.md`, baseline 비교는 `BASELINE.md`, 모듈별 명세는 `modules/*/SPEC.md` 참조.

---

## 한 줄 정의

매 상황(상태)에서 "어떤 구종을 어느 존에 던지는 것이 가장 유리한가"를 데이터 기반으로 제시하고,
시청자가 실제 투수의 선택과 비교("정배/역배")하며 경기를 보게 만드는 서비스.

## 목표 사용자

가볍게 야구를 즐기는 시청자(야구 너드·전문가가 아님).
성공 기준은 예측 정확도가 아니라 **비교의 재미 + 설명의 납득성**.
→ 제품 철학 상세: `docs/product.md`

## 핵심 차별점 (baseline 대비)

SmartPitch(Otremba 2022)를 baseline으로 삼되, 다음에서 갈라진다:
카운트만 → 풀 상태공간 / Value Iteration → 함수근사 / 타석 내 → 타석 너머 추론 / 수동 → 자동 클러스터링.
→ 상세: `BASELINE.md`

---

## 문서 지도

| 문서 | 책임 (단일) |
|---|---|
| `PROJECT_PLAN.md` (이 문서) | 무엇을·왜·순서. 인덱스. |
| `DESIGN_DECISIONS.md` | 확정된 결정 + 이유 + **금지사항**. 에이전트의 헌법. |
| `BASELINE.md` | SmartPitch 요약 + 우리 확장점 + 평가 기여. |
| `docs/product.md` | 제품 정의·사용자·UX 철학 상세. |
| `docs/architecture.md` | 전체 시스템 구조(학습/서빙 경로, 데이터 흐름). |
| `docs/evaluation.md` | 평가 방법론(NN 손실 + 야구상식 정성 + 백테스팅). |
| `docs/glossary.md` | 용어집(상태공간, 13존, RE24, FQI 등). |
| `modules/<m>/README.md` | 모듈 개요·입출력·의존성. |
| `modules/<m>/SPEC.md` | 모듈 상세 명세(작업 직전 작성). |
| `modules/<m>/EXPERIMENTS.md` | 모듈 실험 기록(방법론 후보 비교). |

문서 중복 금지 원칙: 전역 진실은 루트 3개, 상세 설명은 docs/, 모듈 로컬 사실은 modules/.
같은 사실을 두 곳에 쓰지 않는다.

---

## 모듈 목록

| 모듈 | 역할 | 레이어 |
|---|---|---|
| `data` | Statcast 수집·캐시·검증 | 1 |
| `clustering` | 투수/타자 클러스터링 + 위협도 | 1 |
| `transition_nn` | P(결과\|상태,행동) — 행동 필터 | 2 |
| `q_function` | (상태,행동)→기대실점 — FQI 등 | 2 |
| `inference` | 필터→정렬→Top-1/2/위험 | 3 |
| `serving` | 사전계산 캐시·StatsAPI 어댑터 | 3 |
| `output` | 자연어 매핑·정배/역배 UI | 4 |

---

## 작업 순서 (의존성 순)

1. 문서 토대 (PROJECT_PLAN / DESIGN_DECISIONS / BASELINE) ← **현재**
2. 환경 세팅 + 데이터 수집·캐시·검증 (`data`)
3. 구종 군 / 존 그리드 확정
4. 타자·투수 클러스터링 → 검수 (`clustering`)
5. 룩업 테이블 + fallback
6. 추천 결과 데이터 구조 + reward 설계 확정
7. 학습 데이터 재구성
8. transition NN 학습 (baseline 재현, CE ≤ 0.861) (`transition_nn`)
9. Q함수 학습 (FQI 등 실험·선택) (`q_function`)
10. 추론 코어 (`inference`)
11. 평가 (NN 손실 + 야구상식 + 백테스팅)
12. 서빙·캐시·어댑터 (`serving`)
13. 출력 레이어 (`output`)
14. 라이브 리허설 → 2026 재학습 → 배포

의존성: 1→14 대체로 순차. 단 6번(추천 데이터 구조)은 당겨서 먼저 확정하면 1~2 레이어를 그 목표 향해 짤 수 있어 재작업을 줄인다.

## 환경 / 역할

- Mac Mini 로컬(CPU): 세팅·데이터·클러스터링·분석. 패키지 uv, 경로 `~/projects/baseball-zerobase`, 데이터 로컬 디스크.
- Colab 유료(GPU): NN·FQI 학습 단계부터.
- 코드 = GitHub(SongRoute), 데이터·가중치 = 로컬/T7 SSD.
- 역할: 결정·평가 = Song / 구현·실행·실험 = 에이전트 / 설계보조·사양작성·검수기준 = Claude.
- 환경 상세: `docs/architecture.md`, 기술 셋업: `CLAUDE.md`(코드 작업 시 생성).
