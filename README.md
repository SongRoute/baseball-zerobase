# baseball-zerobase

MLB 투구 추천 서비스. 매 상황에서 "어떤 구종을 어느 존에 던지는 것이 유리한가"를
데이터로 제시하고, 시청자가 실제 투수 선택과 비교(정배/역배)하며 보게 만든다.

## 문서 진입 순서
1. `PROJECT_PLAN.md` — 무엇을·왜·순서 (최상위 인덱스)
2. `DESIGN_DECISIONS.md` — 확정 결정 + 금지사항 (에이전트 필독)
3. `BASELINE.md` — SmartPitch 대비 확장점
4. `docs/` — product / architecture / evaluation / glossary
5. `modules/<m>/` — 모듈별 README / SPEC / EXPERIMENTS

## 에이전트 작업 규칙
- 모든 작업 시작 시 `DESIGN_DECISIONS.md`의 금지사항을 읽고 위반하지 않는다.
- 결정·평가는 사람(Song). 구현·실행·실험은 에이전트.
- 문서 중복 금지: 전역 진실은 루트, 상세는 docs/, 모듈 로컬은 modules/.
