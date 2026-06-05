from __future__ import annotations

from datetime import date, datetime
from typing import Any

import polars as pl


_REQUIRED_COLUMNS = {
    "as_of_timestamp",
    "game_date",
    "game_pk",
    "pitch_timestamp",
    "pitcher_id",
    "pitch_type",
}
_PHYSICAL_COLUMNS = (
    "release_speed",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
)


def add_pitcher_profiles(
    snapshot_frame: pl.DataFrame,
    *,
    min_pitch_type_pitches: int = 100,
    min_pitch_type_usage: float = 0.05,
    current_season_min_pitches: int = 300,
    shrinkage_prior_pitches: int = 50,
) -> pl.DataFrame:
    if min_pitch_type_pitches < 0:
        raise ValueError("min_pitch_type_pitches must be non-negative")
    if not 0 <= min_pitch_type_usage <= 1:
        raise ValueError("min_pitch_type_usage must be between 0 and 1")
    if current_season_min_pitches < 0:
        raise ValueError("current_season_min_pitches must be non-negative")
    if shrinkage_prior_pitches < 0:
        raise ValueError("shrinkage_prior_pitches must be non-negative")
    _require_columns(snapshot_frame, _REQUIRED_COLUMNS, "pitcher profile frame")
    if snapshot_frame.is_empty():
        return snapshot_frame.with_columns(
            pl.lit(None, dtype=pl.Datetime).alias("pitcher_profile_as_of_timestamp"),
            pl.lit(0, dtype=pl.Int64).alias("pitcher_profile_prior_pitch_count"),
            pl.lit(0, dtype=pl.Int64).alias("pitcher_profile_current_season_pitch_count"),
            pl.lit(0, dtype=pl.Int64).alias("pitcher_profile_pitch_type_prior_count"),
            pl.lit(0.0, dtype=pl.Float64).alias("pitcher_profile_pitch_type_usage_rate"),
            pl.lit(False, dtype=pl.Boolean).alias("pitcher_profile_current_season_sufficient"),
            pl.lit(False, dtype=pl.Boolean).alias("pitcher_profile_uses_prior_season_shrinkage"),
            pl.lit(0.0, dtype=pl.Float64).alias("pitcher_profile_reliability_weight"),
            pl.lit([], dtype=pl.List(pl.String)).alias("pitcher_owned_pitch_types"),
            pl.lit(False, dtype=pl.Boolean).alias("pitcher_pitch_type_owned"),
            pl.lit(None, dtype=pl.Float64).alias("pitcher_profile_release_speed_mean"),
        )

    rows = list(snapshot_frame.iter_rows(named=True))
    out_rows: list[dict[str, Any]] = []
    for target in rows:
        target_date = _date_value(target["game_date"])
        as_of = _datetime_value(target["as_of_timestamp"])
        prior = [
            row
            for row in rows
            if _integer_or_none(row.get("pitcher_id")) == _integer_or_none(target.get("pitcher_id"))
            and row.get("game_pk") != target.get("game_pk")
            and _date_within_two_years(_date_value(row["game_date"]), target_date)
            and _datetime_value(row["pitch_timestamp"]) < as_of
        ]
        league_prior = [
            row
            for row in rows
            if _date_within_two_years(_date_value(row["game_date"]), target_date)
            and _datetime_value(row["pitch_timestamp"]) < as_of
        ]
        current_season = [
            row for row in prior if _date_value(row["game_date"]).year == target_date.year
        ]
        pitch_type = _string_or_none(target.get("pitch_type"))
        pitch_prior = [row for row in prior if _string_or_none(row.get("pitch_type")) == pitch_type]
        usage_rate = len(pitch_prior) / len(prior) if prior else 0.0
        owned_pitch_types = _owned_pitch_types(
            prior,
            min_pitch_type_pitches=min_pitch_type_pitches,
            min_pitch_type_usage=min_pitch_type_usage,
        )
        reliability = len(prior) / (len(prior) + shrinkage_prior_pitches)
        out_row = dict(target)
        out_row.update(
            {
                "pitcher_profile_as_of_timestamp": as_of,
                "pitcher_profile_prior_pitch_count": len(prior),
                "pitcher_profile_current_season_pitch_count": len(current_season),
                "pitcher_profile_pitch_type_prior_count": len(pitch_prior),
                "pitcher_profile_pitch_type_usage_rate": usage_rate,
                "pitcher_profile_current_season_sufficient": len(current_season)
                >= current_season_min_pitches,
                "pitcher_profile_uses_prior_season_shrinkage": len(current_season)
                < current_season_min_pitches
                and bool(prior),
                "pitcher_profile_reliability_weight": reliability,
                "pitcher_owned_pitch_types": owned_pitch_types,
                "pitcher_pitch_type_owned": pitch_type in owned_pitch_types,
                "pitcher_profile_release_speed_mean": _shrunk_mean(
                    pitch_prior,
                    league_prior,
                    "release_speed",
                    pitch_type,
                    shrinkage_prior_pitches,
                ),
            }
        )
        for column in _PHYSICAL_COLUMNS:
            output_column = f"pitcher_profile_{column}_mean"
            if output_column not in out_row:
                out_row[output_column] = _shrunk_mean(
                    pitch_prior,
                    league_prior,
                    column,
                    pitch_type,
                    shrinkage_prior_pitches,
                )
        out_rows.append(out_row)

    return pl.DataFrame(out_rows)


def _owned_pitch_types(
    rows: list[dict[str, Any]],
    *,
    min_pitch_type_pitches: int,
    min_pitch_type_usage: float,
) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        pitch_type = _string_or_none(row.get("pitch_type"))
        if pitch_type is not None:
            counts[pitch_type] = counts.get(pitch_type, 0) + 1
    total = len(rows)
    return sorted(
        pitch_type
        for pitch_type, count in counts.items()
        if count >= min_pitch_type_pitches and total and count / total >= min_pitch_type_usage
    )


def _shrunk_mean(
    pitch_rows: list[dict[str, Any]],
    league_rows: list[dict[str, Any]],
    column: str,
    pitch_type: str | None,
    prior: int,
) -> float | None:
    if column not in pitch_rows[0] if pitch_rows else False:
        return None
    pitcher_values = [_float_or_none(row.get(column)) for row in pitch_rows]
    pitcher_values = [value for value in pitcher_values if value is not None]
    league_values = [
        _float_or_none(row.get(column))
        for row in league_rows
        if _string_or_none(row.get("pitch_type")) == pitch_type and column in row
    ]
    league_values = [value for value in league_values if value is not None]
    if not pitcher_values and not league_values:
        return None
    if not league_values:
        return sum(pitcher_values) / len(pitcher_values)
    league_mean = sum(league_values) / len(league_values)
    if not pitcher_values:
        return league_mean
    pitcher_mean = sum(pitcher_values) / len(pitcher_values)
    weight = len(pitcher_values) / (len(pitcher_values) + prior) if prior else 1.0
    return weight * pitcher_mean + (1.0 - weight) * league_mean


def _require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _date_within_two_years(value: date, target: date) -> bool:
    try:
        cutoff = target.replace(year=target.year - 2)
    except ValueError:
        cutoff = date(target.year - 2, 2, 28)
    return cutoff <= value < target


def _date_value(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _datetime_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _integer_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
