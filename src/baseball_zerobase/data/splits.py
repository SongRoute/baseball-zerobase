from datetime import date
from enum import StrEnum
from pathlib import Path


class DatasetRole(StrEnum):
    DEV_REGULAR = "dev_regular"
    LOCKED_POSTSEASON_2025 = "locked_postseason_2025"
    LOCKED_REGULAR_2026 = "locked_regular_2026"
    EXCLUDED = "excluded"


class LockedDataError(RuntimeError):
    pass


def classify_row(game_date: date, game_type: str) -> DatasetRole:
    if game_date.year == 2025 and game_type in {"F", "D", "L", "W"}:
        return DatasetRole.LOCKED_POSTSEASON_2025
    if date(2026, 3, 25) <= game_date <= date(2026, 5, 31) and game_type == "R":
        return DatasetRole.LOCKED_REGULAR_2026
    if 2022 <= game_date.year <= 2025 and game_type == "R":
        return DatasetRole.DEV_REGULAR
    return DatasetRole.EXCLUDED


def require_dev_role(role: DatasetRole) -> None:
    if role is not DatasetRole.DEV_REGULAR:
        raise LockedDataError(f"development pipeline cannot read dataset role {role}")


def guard_dev_path(path: Path, locked_dir: Path) -> None:
    if path.resolve().is_relative_to(locked_dir.resolve()):
        raise LockedDataError(f"development pipeline cannot read locked path {path}")
