# AGENTS.md — data 모듈

루트 `AGENTS.md`와 `DESIGN_DECISIONS.md`를 먼저 따른다. 아래는 이 모듈 로컬 규칙.

## 이 모듈의 핵심 금지/주의
- raw 데이터는 **그대로** 캐시. 임의 필터링·가공 금지 (DESIGN_DECISIONS 금지사항 5).
- 시즌은 `config.py` 상수로. 검증은 과거 완결 시즌(우선 2024), 최종은 2026.
- pybaseball 대량 수집은 월/2주 단위로 쪼개 즉시 parquet 캐시. 캐시 있으면 재수집 금지.

## 산출물
- `data/raw/statcast_<season>.parquet`
- `results/data_validation_<season>.md` (pitch_type·zone 분포 포함 — 다음 단계 입력, 사람이 검수)

## 검수 포인트 (사람)
- pitch_type 분포(신규 구종코드 포함 여부), zone 값 체계 → 구종 군 매핑·존 그리드 설계의 근거.
