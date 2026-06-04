from datetime import date
from pathlib import Path

import pytest

from baseball_zerobase.data.splits import (
    DatasetRole,
    LockedDataError,
    classify_row,
    guard_dev_path,
    require_dev_role,
)


def test_classifies_locked_partitions() -> None:
    assert classify_row(date(2025, 10, 1), "F") is DatasetRole.LOCKED_POSTSEASON_2025
    assert classify_row(date(2026, 4, 15), "R") is DatasetRole.LOCKED_REGULAR_2026
    assert classify_row(date(2024, 7, 1), "R") is DatasetRole.DEV_REGULAR


def test_dev_guard_rejects_locked_role() -> None:
    with pytest.raises(LockedDataError):
        require_dev_role(DatasetRole.LOCKED_REGULAR_2026)


def test_dev_guard_rejects_locked_path(tmp_path: Path) -> None:
    locked = tmp_path / "data" / "locked"
    with pytest.raises(LockedDataError):
        guard_dev_path(locked / "snapshots.parquet", locked)
