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


@pytest.mark.parametrize("game_type", ["F", "D", "L", "W"])
def test_classifies_2025_postseason_as_locked(game_type: str) -> None:
    assert classify_row(date(2025, 10, 1), game_type) is DatasetRole.LOCKED_POSTSEASON_2025


@pytest.mark.parametrize("game_date", [date(2026, 3, 25), date(2026, 5, 31)])
def test_classifies_2026_locked_regular_boundaries_as_locked(game_date: date) -> None:
    assert classify_row(game_date, "R") is DatasetRole.LOCKED_REGULAR_2026


@pytest.mark.parametrize("year", [2022, 2023, 2024, 2025])
def test_classifies_dev_regular_seasons(year: int) -> None:
    assert classify_row(date(year, 7, 1), "R") is DatasetRole.DEV_REGULAR


@pytest.mark.parametrize(
    "role",
    [
        DatasetRole.LOCKED_POSTSEASON_2025,
        DatasetRole.LOCKED_REGULAR_2026,
        DatasetRole.EXCLUDED,
    ],
)
def test_dev_guard_rejects_non_dev_roles(role: DatasetRole) -> None:
    with pytest.raises(LockedDataError):
        require_dev_role(role)


def test_dev_guard_rejects_locked_path(tmp_path: Path) -> None:
    locked = tmp_path / "data" / "locked"
    with pytest.raises(LockedDataError):
        guard_dev_path(locked / "snapshots.parquet", locked)
