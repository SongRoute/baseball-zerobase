# BASELINE — SmartPitch (Otremba 2022) 대비 우리 확장점

> baseline: Stephen Otremba, "SmartPitch: Applied Machine Learning for Professional Baseball
> Pitching Strategy", MIT MEng Thesis, 2022.
> 이 문서의 책임: baseline이 무엇을 했고, 우리가 어디서·왜 갈라지는가. 발표·평가의 토대.

---

## 1. SmartPitch 요약

투수-타자 대결을 MDP로 모델링하고 Value Iteration으로 최적 투구 정책을 구한다.

| 구성요소 | SmartPitch 방식 |
|---|---|
| 상태공간 | **볼카운트만.** 비종료 12개 + 종료(삼진/볼넷/인플레이) = 총 20개. 주자·아웃·이닝·점수차 미포함. |
| 행동공간 | (구종 × 위치). 구종은 투수별 arsenal, 위치는 존 그리드(예시 9존). |
| 전이확률 | 직접 계산 불가(표본 부족) → **신경망 추정**. 입력 77차원(행동+상태 one-hot+타자 벡터), 출력 4-클래스(strike/ball/foul/in-play) softmax. |
| 보상 | 비인플레이: RE288(Run Expectancy) 상태 간 차이. 인플레이: xwOBA. |
| 최적정책 | **Value Iteration**(Bellman). 결과를 존 히트맵으로 시각화. |
| 구종 식별 | UMAP 2D 임베딩, **사람이 눈으로** 군집 식별(자동 클러스터링은 future work). 타자는 Statcast 사전계산 지표 벡터를 그냥 사용. |
| 평가 | (1) transition NN 손실(CE 0.861, Brier 0.482) (2) 히트맵을 야구 상식으로 정성 해석. 추천 정확도 검증은 **하지 않음**(future work로 명시). |

NN 성능 표(논문 7.1): Baseline 1.356 / LogReg 1.281 / RandomForest 0.968 / GBT 0.930 / **NN 0.861** (Cross Entropy).

---

## 2. 우리가 갈라지는 지점 (확장점)

| 항목 | SmartPitch | baseball-zerobase | 왜 |
|---|---|---|---|
| 상태공간 | 카운트만(12) | 풀 상태(~58만): base-out·이닝·점수차·클러스터 | 상황 의존 전략(병살·승부회피)을 표현하려면 카운트만으론 부족 |
| RL 방법 | Value Iteration(tabular) | 함수근사(FQI 등) | 풀 상태×78행동=4,500만 셀, tabular 불가 |
| 시간지평 | 보상을 "주자 없음·0아웃" 단일 시나리오 고정 | base-out·점수차를 상태에 포함, 타석 너머 고려 | 강타자 승부 회피, 1아웃 주자 병살 유도 등을 잡으려고 |
| 클러스터링 | 투수 구종 = 사람이 눈으로. 타자 = 사전계산 지표 그대로 | 투수·타자 **자동 클러스터링**, 타자는 약점유형+위협도 2축 | 논문이 future work로 남긴 것. 자동화·확장 |
| 평가 | NN 손실 + 정성 해석만 | + **백테스팅 기반 약한 정량 평가** | 논문이 미룬 future work. 우리 기여 지점 |
| 제품 | 투수 코칭용 히트맵 | 가벼운 시청자용 정배/역배 비교 | 타깃·UX가 근본적으로 다름 |

---

## 3. 공통점 (재현하는 부분)

- **transition NN = 거의 동일.** 4-클래스(strike/ball/foul/in-play) 확률분포, CE 손실. baseline의 검증된 레시피를 재현·확장.
  우리 구조에선 이것이 "행동 필터" 역할(D8).
- 보상에 RE 계열 + xwOBA(약한 타구 유도) 채택은 동일(D9).
- UMAP을 클러스터링 차원축소에 사용(단, 우리는 HDBSCAN 자동화 추가).

→ **S1 sprint 목표(CE ≤ 0.861)는 이 transition NN 재현으로 달성.** 거기서부터 우리 설계가 갈라진다.

---

## 4. baseline의 한계 = 우리 기회

논문 본인들이 결론(8.3)·reflections(9장)에서 명시한 future work:
- 자동 클러스터링 → **우리가 함 (clustering 모듈).**
- 대안 확률 추정 알고리즘(앙상블 등) → q_function 실험에서 탐색 가능.
- 일/주 단위 등 더 세밀한 입력 → 우리 개인 보조 피처·선수별 테이블이 일부 대응.
- 엄밀한 정량 평가 + **백테스팅** → 우리 평가의 핵심 기여 (docs/evaluation.md).

평가의 본질적 한계("정답 없음")는 우리도 못 푼다. 다만 우리 제품 컨셉(정배/역배 비교)은
추천의 정확성을 주장하지 않으므로 이 한계를 **제품 설계로 우회**한다 (D1).
