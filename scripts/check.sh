#!/usr/bin/env bash
# 통합 검증: 포맷/린트/타입/테스트를 한 방에. 커밋 전 pre-commit hook이 자동 실행.
# uv 환경 기준. 단계별로 실패 시 즉시 중단.
set -euo pipefail

echo "==> ruff (lint + format check)"
uv run ruff check . || { echo "ruff check 실패"; exit 1; }
uv run ruff format --check . || { echo "ruff format 어긋남 (uv run ruff format . 로 수정)"; exit 1; }

# 타입 체크 (mypy 또는 pyright 중 설치된 것 사용)
if uv run python -c "import mypy" 2>/dev/null; then
  echo "==> mypy"
  uv run mypy src/ || { echo "mypy 실패"; exit 1; }
fi

# 테스트 (tests/ 있을 때만)
if [ -d tests ]; then
  echo "==> pytest"
  uv run pytest -q || { echo "pytest 실패"; exit 1; }
else
  echo "==> tests/ 없음, 스킵"
fi

echo "==> check 통과"
