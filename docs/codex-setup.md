# Codex 세팅 가이드

이 프로젝트는 Codex 위임을 위해 매뉴얼(Claude Code 고급 자동화)을 Codex 네이티브로 번안해 적용했다.

## 구성요소

| 파일 | 역할 |
|---|---|
| `AGENTS.md` (루트) | Codex가 작업 전 자동 로드하는 진입점. 문서 가리키기. |
| `modules/<m>/AGENTS.md` | 모듈 작업 시 추가 로드되는 로컬 규칙(가까운 게 우선). |
| `.codex/config.toml` | 서브에이전트 동시성([agents]), 문서 한도 등. |
| `.codex/agents/*.toml` | 커스텀 서브에이전트(explorer/experiment_runner/reviewer). |
| `.codex/hooks.json` | (선택·실험) 작업 종료 알림. config.toml에서 켜야 작동. |
| `scripts/check.sh` | 통합 검증(ruff+타입+테스트). |
| `scripts/git-hooks/pre-commit` | 커밋 전 check.sh 자동 실행. |

## 설치 (최초 1회, Mac 로컬)

```bash
cd ~/projects/baseball-zerobase

# git pre-commit hook 활성화
git config core.hooksPath scripts/git-hooks

# (선택) Codex 네이티브 알림 hook 켜기: .codex/config.toml에서
#   [features]
#   codex_hooks = true
# 주석 해제
```

## 사용

- **일반 작업**: 그냥 `codex` 실행. 루트+모듈 AGENTS.md가 자동 로드돼 금지사항을 따른다.
- **탐색**: "explorer로 data 모듈 구조 파악해줘" → 읽기 전용 탐색.
- **실험 병렬**: "experiment_runner로 클러스터링 K=4~10과 스케일러 2종을 각각 돌려 EXPERIMENTS.md에 비교표로 정리해줘."
- **대량 실험**: `spawn_agents_on_csv`로 후보를 행 단위 병렬(예: 후보별 설정 CSV → 워커마다 한 실험).
- **검수 보조**: "reviewer로 이번 변경이 DESIGN_DECISIONS 금지사항을 어기는지 점검해줘." (최종 판단은 사람)

## 주의

- 모델 슬러그(`.codex/agents/*.toml`의 model)는 계정에서 `/model`로 선택 가능한 실제 이름으로 맞출 것.
- 서브에이전트는 토큰을 더 쓴다. 독립 작업·실험에만 쓰고, 의존성 체인(데이터→클러스터→학습)은 순차.
- 검수·결정은 사람. 토큰 여유가 있어도 산출물 검증까지 자동화하지 않는다.
