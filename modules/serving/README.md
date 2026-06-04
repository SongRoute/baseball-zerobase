# 모듈: serving

> 역할: 사전계산 캐시 + StatsAPI 어댑터
> 레이어: 3 · 의존: inference, clustering
> 전역 결정·금지사항은 루트 `DESIGN_DECISIONS.md` 참조. 이 문서는 모듈 개요만 담는다.

## 입력
StatsAPI live JSON / 선수별 사전계산 테이블

## 출력
상태벡터 → 캐시 룩업 결과

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
