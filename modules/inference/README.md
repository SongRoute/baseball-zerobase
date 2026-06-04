# 모듈: inference

> 역할: 추론 코어: 필터→정렬→Top-1/2/위험
> 레이어: 3 · 의존: transition_nn, q_function
> 전역 결정·금지사항은 루트 `DESIGN_DECISIONS.md` 참조. 이 문서는 모듈 개요만 담는다.

## 입력
상태공간 1건

## 출력
정렬된 (구종,존,score) 리스트 → Top-1/2/위험

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
