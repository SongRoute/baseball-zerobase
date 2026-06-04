from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import polars as pl

from baseball_zerobase.data.splits import DatasetRole, classify_row


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidationReport:
    row_count: int
    action_row_count: int
    relative_zone_counts: dict[str, int]
    action_counts: dict[str, int]
    outcome_counts: dict[str, int]
    terminal_reason_counts: dict[str, int]
    half_inning_ended_counts: dict[str, int]
    locked_row_count: int
    timestamp_joined_counts: dict[str, int]
    dataset_role_counts: dict[str, int]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_snapshots(frame: pl.DataFrame) -> ValidationReport:
    warnings: list[str] = []
    _reject_timestamp_leakage(frame, warnings)

    dataset_role_counts = _dataset_role_counts(frame, warnings)
    locked_row_count = sum(
        count
        for role, count in dataset_role_counts.items()
        if role
        in {
            DatasetRole.LOCKED_POSTSEASON_2025.value,
            DatasetRole.LOCKED_REGULAR_2026.value,
        }
    )
    action_frame = _action_frame(frame, warnings)

    return ValidationReport(
        row_count=frame.height,
        action_row_count=action_frame.height,
        relative_zone_counts=_value_counts(action_frame, "relative_zone"),
        action_counts=_value_counts(action_frame, "action"),
        outcome_counts=_value_counts(frame, "outcome", warnings=warnings),
        terminal_reason_counts=_value_counts(frame, "terminal_reason", warnings=warnings),
        half_inning_ended_counts=_value_counts(frame, "half_inning_ended", warnings=warnings),
        locked_row_count=locked_row_count,
        timestamp_joined_counts=_value_counts(frame, "timestamp_joined", warnings=warnings),
        dataset_role_counts=dataset_role_counts,
        warnings=tuple(warnings),
    )


def _reject_timestamp_leakage(frame: pl.DataFrame, warnings: list[str]) -> None:
    missing = sorted({"as_of_timestamp", "pitch_timestamp"}.difference(frame.columns))
    if missing:
        warnings.append(f"missing leakage timestamp columns: {missing}")
        return

    leaky_keys: list[str] = []
    for index, row in enumerate(
        frame.select(["as_of_timestamp", "pitch_timestamp"]).iter_rows(named=True)
    ):
        as_of_timestamp = _datetime_or_none(row["as_of_timestamp"])
        pitch_timestamp = _datetime_or_none(row["pitch_timestamp"])
        if as_of_timestamp is None or pitch_timestamp is None:
            continue
        if as_of_timestamp >= pitch_timestamp:
            leaky_keys.append(str(index))

    if leaky_keys:
        sample = ", ".join(leaky_keys[:5])
        raise LeakageError(
            "snapshot rows must have as_of_timestamp strictly before pitch_timestamp "
            f"when both exist; leaky row indexes: {sample}"
        )


def _dataset_role_counts(frame: pl.DataFrame, warnings: list[str]) -> dict[str, int]:
    missing = sorted({"game_date", "game_type"}.difference(frame.columns))
    if missing:
        warnings.append(f"missing dataset role columns: {missing}")
        return {}

    counts: dict[str, int] = {}
    for row in frame.select(["game_date", "game_type"]).iter_rows(named=True):
        game_date = _date_value(row["game_date"])
        game_type = "" if row["game_type"] is None else str(row["game_type"])
        role = classify_row(game_date, game_type).value
        counts[role] = counts.get(role, 0) + 1
    return dict(sorted(counts.items()))


def _action_frame(frame: pl.DataFrame, warnings: list[str]) -> pl.DataFrame:
    missing = sorted({"action", "relative_zone"}.difference(frame.columns))
    if missing:
        warnings.append(f"missing action distribution columns: {missing}")
        return pl.DataFrame({"action": [], "relative_zone": []})
    return frame.filter(pl.col("action").is_not_null() & pl.col("relative_zone").is_not_null())


def _value_counts(
    frame: pl.DataFrame,
    column: str,
    *,
    warnings: list[str] | None = None,
) -> dict[str, int]:
    if column not in frame.columns:
        if warnings is not None:
            warnings.append(f"missing count column: {column}")
        return {}

    counts: dict[str, int] = {}
    for row in frame.select(column).iter_rows(named=True):
        value = row[column]
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _date_value(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _datetime_or_none(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return datetime.fromisoformat(str(value))
