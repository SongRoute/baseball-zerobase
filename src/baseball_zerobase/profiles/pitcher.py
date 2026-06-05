from __future__ import annotations

from dataclasses import dataclass, field
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

    rows = [
        _with_profile_keys(row, index)
        for index, row in enumerate(snapshot_frame.iter_rows(named=True))
    ]
    out_rows: list[dict[str, Any] | None] = [None] * len(rows)
    rows_by_date: dict[date, list[dict[str, Any]]] = {}
    targets_by_date: dict[date, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_date.setdefault(row["_game_date"], []).append(row)
        targets_by_date.setdefault(row["_game_date"], []).append(row)

    league_window = _ProfileCounts()
    pitcher_window: dict[int | None, _ProfileCounts] = {}
    active_dates: list[date] = []
    added_dates: set[date] = set()
    next_add_index = 0
    sorted_dates = sorted(rows_by_date)

    for target_date in sorted(targets_by_date):
        cutoff = _two_year_cutoff(target_date)
        while next_add_index < len(sorted_dates) and sorted_dates[next_add_index] < target_date:
            date_to_add = sorted_dates[next_add_index]
            for row in rows_by_date[date_to_add]:
                _add_profile_counts(row, league_window, pitcher_window)
            active_dates.append(date_to_add)
            added_dates.add(date_to_add)
            next_add_index += 1
        while active_dates and active_dates[0] < cutoff:
            date_to_remove = active_dates.pop(0)
            if date_to_remove in added_dates:
                for row in rows_by_date[date_to_remove]:
                    _remove_profile_counts(row, league_window, pitcher_window)
                added_dates.remove(date_to_remove)

        for target in targets_by_date[target_date]:
            as_of = target["_as_of_timestamp"]
            pitcher_id = _integer_or_none(target.get("pitcher_id"))
            prior = pitcher_window.get(pitcher_id, _ProfileCounts())
            league_prior = league_window
            current_season_count = prior.season_counts.get(target_date.year, 0)
            pitch_type = _string_or_none(target.get("pitch_type"))
            pitch_prior_count = prior.pitch_type_counts.get(pitch_type, 0)
            usage_rate = pitch_prior_count / prior.pitch_count if prior.pitch_count else 0.0
            owned_pitch_types = _owned_pitch_types_from_counts(
                prior.pitch_type_counts,
                prior.pitch_count,
                min_pitch_type_pitches=min_pitch_type_pitches,
                min_pitch_type_usage=min_pitch_type_usage,
            )
            reliability = (
                prior.pitch_count / (prior.pitch_count + shrinkage_prior_pitches)
                if prior.pitch_count + shrinkage_prior_pitches
                else 0.0
            )
            out_row = dict(target)
            out_row.pop("_row_index")
            out_row.pop("_game_date")
            out_row.pop("_pitch_timestamp")
            out_row.pop("_as_of_timestamp")
            out_row.update(
                {
                    "pitcher_profile_as_of_timestamp": as_of,
                    "pitcher_profile_prior_pitch_count": prior.pitch_count,
                    "pitcher_profile_current_season_pitch_count": current_season_count,
                    "pitcher_profile_pitch_type_prior_count": pitch_prior_count,
                    "pitcher_profile_pitch_type_usage_rate": usage_rate,
                    "pitcher_profile_current_season_sufficient": current_season_count
                    >= current_season_min_pitches,
                    "pitcher_profile_uses_prior_season_shrinkage": current_season_count
                    < current_season_min_pitches
                    and bool(prior.pitch_count),
                    "pitcher_profile_reliability_weight": reliability,
                    "pitcher_owned_pitch_types": owned_pitch_types,
                    "pitcher_pitch_type_owned": pitch_type in owned_pitch_types,
                    "pitcher_profile_release_speed_mean": _shrunk_aggregate_mean(
                        prior,
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
                    out_row[output_column] = _shrunk_aggregate_mean(
                        prior,
                        league_prior,
                        column,
                        pitch_type,
                        shrinkage_prior_pitches,
                    )
            out_rows[target["_row_index"]] = out_row

    return pl.DataFrame([row for row in out_rows if row is not None])


@dataclass
class _ValueStats:
    total: float = 0.0
    count: int = 0

    def add(self, value: float | None) -> None:
        if value is None:
            return
        self.total += value
        self.count += 1

    def remove(self, value: float | None) -> None:
        if value is None:
            return
        self.total -= value
        self.count -= 1

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


@dataclass
class _ProfileCounts:
    pitch_count: int = 0
    season_counts: dict[int, int] = field(default_factory=dict)
    pitch_type_counts: dict[str | None, int] = field(default_factory=dict)
    physical: dict[str | None, dict[str, _ValueStats]] = field(default_factory=dict)


def _with_profile_keys(row: dict[str, Any], index: int) -> dict[str, Any]:
    out = dict(row)
    out["_row_index"] = index
    out["_game_date"] = _date_value(row["game_date"])
    out["_pitch_timestamp"] = _datetime_value(row["pitch_timestamp"])
    out["_as_of_timestamp"] = _datetime_value(row["as_of_timestamp"])
    return out


def _add_profile_counts(
    row: dict[str, Any],
    league_counts: _ProfileCounts,
    pitcher_counts: dict[int | None, _ProfileCounts],
) -> None:
    pitcher_id = _integer_or_none(row.get("pitcher_id"))
    pitcher = pitcher_counts.setdefault(pitcher_id, _ProfileCounts())
    for counts in (league_counts, pitcher):
        _update_profile_counts(counts, row, delta=1)


def _remove_profile_counts(
    row: dict[str, Any],
    league_counts: _ProfileCounts,
    pitcher_counts: dict[int | None, _ProfileCounts],
) -> None:
    pitcher_id = _integer_or_none(row.get("pitcher_id"))
    pitcher = pitcher_counts.setdefault(pitcher_id, _ProfileCounts())
    for counts in (league_counts, pitcher):
        _update_profile_counts(counts, row, delta=-1)


def _update_profile_counts(counts: _ProfileCounts, row: dict[str, Any], *, delta: int) -> None:
    game_date = row["_game_date"]
    pitch_type = _string_or_none(row.get("pitch_type"))
    counts.pitch_count += delta
    counts.season_counts[game_date.year] = counts.season_counts.get(game_date.year, 0) + delta
    if counts.season_counts[game_date.year] == 0:
        del counts.season_counts[game_date.year]
    counts.pitch_type_counts[pitch_type] = counts.pitch_type_counts.get(pitch_type, 0) + delta
    if counts.pitch_type_counts[pitch_type] == 0:
        del counts.pitch_type_counts[pitch_type]
    for column in _PHYSICAL_COLUMNS:
        value = _float_or_none(row.get(column))
        value_stats = counts.physical.setdefault(pitch_type, {}).setdefault(column, _ValueStats())
        if delta > 0:
            value_stats.add(value)
        else:
            value_stats.remove(value)
        if value_stats.count == 0:
            del counts.physical[pitch_type][column]
    if pitch_type in counts.physical and not counts.physical[pitch_type]:
        del counts.physical[pitch_type]


def _owned_pitch_types_from_counts(
    counts: dict[str | None, int],
    total: int,
    *,
    min_pitch_type_pitches: int,
    min_pitch_type_usage: float,
) -> list[str]:
    return sorted(
        pitch_type
        for pitch_type, count in counts.items()
        if pitch_type is not None
        and count >= min_pitch_type_pitches
        and total
        and count / total >= min_pitch_type_usage
    )


def _shrunk_aggregate_mean(
    pitcher_counts: _ProfileCounts,
    league_counts: _ProfileCounts,
    column: str,
    pitch_type: str | None,
    prior: int,
) -> float | None:
    pitcher_values = pitcher_counts.physical.get(pitch_type, {}).get(column, _ValueStats())
    league_values = league_counts.physical.get(pitch_type, {}).get(column, _ValueStats())
    if not pitcher_values.count and not league_values.count:
        return None
    if not league_values.count:
        return pitcher_values.mean
    league_mean = league_values.mean
    if not pitcher_values.count:
        return league_mean
    pitcher_mean = pitcher_values.mean
    if pitcher_mean is None or league_mean is None:
        return None
    weight = pitcher_values.count / (pitcher_values.count + prior) if prior else 1.0
    return weight * pitcher_mean + (1.0 - weight) * league_mean


def _two_year_cutoff(target: date) -> date:
    try:
        return target.replace(year=target.year - 2)
    except ValueError:
        return date(target.year - 2, 2, 28)


def _require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


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
