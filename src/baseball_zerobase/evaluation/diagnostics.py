from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class TransitionDiagnosticsReport:
    row_count: int
    pitch_type_distribution: dict[str, int]
    relative_zone_distribution: dict[str, int]
    batter_weakness_archetype_distribution: dict[str, int]
    batter_threat_score_bucket_distribution: dict[str, int]
    pitcher_profile_reliability_weight_bucket_distribution: dict[str, int]
    profile_feature_null_rates: dict[str, float]
    pitcher_pitch_type_owned_true_rate: float
    pitcher_pitch_type_owned_counts: dict[str, int]
    daily_state_count_summary: dict[str, dict[str, float | int | None]]
    label_outcome_distribution: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def transition_diagnostics(frame: pl.DataFrame) -> TransitionDiagnosticsReport:
    return TransitionDiagnosticsReport(
        row_count=frame.height,
        pitch_type_distribution=_value_counts(frame, "pitch_type"),
        relative_zone_distribution=_value_counts(frame, "relative_zone"),
        batter_weakness_archetype_distribution=_value_counts(frame, "batter_weakness_archetype"),
        batter_threat_score_bucket_distribution=_bucket_distribution(frame, "batter_threat_score"),
        pitcher_profile_reliability_weight_bucket_distribution=_bucket_distribution(
            frame, "pitcher_profile_reliability_weight"
        ),
        profile_feature_null_rates=_profile_feature_null_rates(frame),
        pitcher_pitch_type_owned_true_rate=_true_rate(frame, "pitcher_pitch_type_owned"),
        pitcher_pitch_type_owned_counts=_bool_counts(frame, "pitcher_pitch_type_owned"),
        daily_state_count_summary=_daily_state_count_summary(frame),
        label_outcome_distribution=_value_counts(frame, "outcome"),
    )


def _value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts: dict[str, int] = {}
    for value in frame.get_column(column).to_list():
        key = "null" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _bucket_distribution(frame: pl.DataFrame, column: str) -> dict[str, int]:
    buckets = {"null": 0, "low": 0, "medium": 0, "high": 0}
    if column not in frame.columns:
        return buckets
    for value in frame.get_column(column).to_list():
        bucket = _numeric_bucket(value)
        buckets[bucket] += 1
    return buckets


def _numeric_bucket(value: object) -> str:
    if value is None:
        return "null"
    try:
        numeric = float(str(value))
    except ValueError:
        return "null"
    if numeric >= 0.66:
        return "high"
    if numeric >= 0.33:
        return "medium"
    return "low"


def _profile_feature_null_rates(frame: pl.DataFrame) -> dict[str, float]:
    columns = [
        column
        for column in frame.columns
        if column.startswith("pitcher_profile_")
        or column == "pitcher_pitch_type_owned"
        or column == "pitcher_owned_pitch_types"
        or column.startswith("batter_weakness_")
        or column.startswith("batter_threat_")
        or column == "daily_state_as_of_timestamp"
        or column.startswith("pitcher_day_")
        or column.startswith("batter_day_")
    ]
    if frame.height == 0:
        return {column: 0.0 for column in sorted(columns)}
    return {
        column: float(frame.get_column(column).null_count()) / frame.height
        for column in sorted(columns)
    }


def _true_rate(frame: pl.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.height == 0:
        return 0.0
    true_count = sum(1 for value in frame.get_column(column).to_list() if value is True)
    return true_count / frame.height


def _bool_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    counts = {"true": 0, "false": 0, "null": 0}
    if column not in frame.columns:
        return counts
    for value in frame.get_column(column).to_list():
        if value is True:
            counts["true"] += 1
        elif value is False:
            counts["false"] += 1
        else:
            counts["null"] += 1
    return counts


def _daily_state_count_summary(frame: pl.DataFrame) -> dict[str, dict[str, float | int | None]]:
    columns = [
        column
        for column in frame.columns
        if column.startswith("pitcher_day_") or column.startswith("batter_day_")
    ]
    return {column: _numeric_summary(frame, column) for column in sorted(columns)}


def _numeric_summary(frame: pl.DataFrame, column: str) -> dict[str, float | int | None]:
    if frame.height == 0:
        return {"count": 0, "null_count": 0, "min": None, "max": None, "mean": None}
    series = frame.get_column(column)
    non_null = series.drop_nulls()
    if non_null.is_empty():
        return {
            "count": frame.height,
            "null_count": series.null_count(),
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": frame.height,
        "null_count": series.null_count(),
        "min": _number_or_none(non_null.min()),
        "max": _number_or_none(non_null.max()),
        "mean": _number_or_none(non_null.mean()),
    }


def _number_or_none(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return float(value)
