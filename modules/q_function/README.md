# 모듈: q_function

> 역할: (상태,행동)→기대실점 — FQI 등 실험·선택
> 레이어: 2 · 의존: transition_nn, clustering
> 전역 결정·금지사항은 루트 `DESIGN_DECISIONS.md` 참조. 이 문서는 모듈 개요만 담는다.

## 입력
transition 튜플(s,a,r,s'), reward(RE24/xwOBA)

## 출력
Q함수 가중치 + 표본수 게이트 테이블

## 문서
- `README.md` (이 문서): 개요·입출력·의존성.
- `SPEC.md`: 상세 명세. **작업 직전에 작성**(미리 추측으로 채우지 않음).
- `EXPERIMENTS.md`: 방법론 후보 비교·실험 기록.

## 상태
- [ ] SPEC 작성
- [ ] 구현
- [ ] 검수 (Song)

## 모듈 로컬 메모
(이 모듈 안에서만 유효한 결정·주의사항. 전역 결정은 여기 쓰지 말고 DESIGN_DECISIONS로.)
