from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl


_REQUIRED_COLUMNS = {
    "as_of_timestamp",
    "batter_id",
    "outcome",
    "pitch_timestamp",
    "relative_zone",
}
_CHASE_ZONES = {"chase_high", "chase_low", "chase_inside", "chase_away"}
_SWING_OUTCOMES = {
    "swinging_strike",
    "foul",
    "in_play_out",
    "single",
    "double",
    "triple",
    "home_run",
}
_CONTACT_OUTCOMES = {"foul", "in_play_out", "single", "double", "triple", "home_run"}


def add_batter_archetypes(
    snapshot_frame: pl.DataFrame,
    *,
    min_prior_pitches: int = 50,
    shrinkage_prior_pitches: int = 75,
) -> pl.DataFrame:
    if min_prior_pitches < 0:
        raise ValueError("min_prior_pitches must be non-negative")
    if shrinkage_prior_pitches < 0:
        raise ValueError("shrinkage_prior_pitches must be non-negative")
    _require_columns(snapshot_frame, _REQUIRED_COLUMNS, "batter archetype frame")
    rows = [
        _with_temporal_keys(row, index)
        for index, row in enumerate(snapshot_frame.iter_rows(named=True))
    ]
    output: list[dict[str, Any] | None] = [None] * len(rows)
    prior_rows = sorted(rows, key=lambda row: row["_pitch_timestamp"])
    targets = sorted(rows, key=lambda row: row["_as_of_timestamp"])
    batter_counts: dict[Any, _PitchTendencyCounts] = {}
    league_counts = _PitchTendencyCounts()
    prior_index = 0

    for target in targets:
        as_of = target["_as_of_timestamp"]
        while prior_index < len(prior_rows) and prior_rows[prior_index]["_pitch_timestamp"] < as_of:
            _add_pitch_tendency(prior_rows[prior_index], league_counts, batter_counts)
            prior_index += 1

        prior = batter_counts.get(target.get("batter_id"), _PitchTendencyCounts())
        chase_rate = _shrunk_count_rate(
            prior.chase_swing_count,
            prior.pitch_count,
            league_counts.chase_swing_count,
            league_counts.pitch_count,
            shrinkage_prior_pitches,
        )
        whiff_rate = _shrunk_count_rate(
            prior.whiff_count,
            prior.pitch_count,
            league_counts.whiff_count,
            league_counts.pitch_count,
            shrinkage_prior_pitches,
        )
        called_strike_rate = _shrunk_count_rate(
            prior.called_strike_count,
            prior.pitch_count,
            league_counts.called_strike_count,
            league_counts.pitch_count,
            shrinkage_prior_pitches,
        )
        confidence = _confidence(prior.pitch_count, shrinkage_prior_pitches)
        out_row = dict(target)
        out_row.pop("_row_index")
        out_row.pop("_pitch_timestamp")
        out_row.pop("_as_of_timestamp")
        out_row.update(
            {
                "batter_weakness_as_of_timestamp": as_of,
                "batter_weakness_sample_size": prior.pitch_count,
                "batter_weakness_confidence": confidence,
                "batter_weakness_chase_swing_rate": chase_rate,
                "batter_weakness_whiff_rate": whiff_rate,
                "batter_weakness_called_strike_rate": called_strike_rate,
                "batter_weakness_archetype": _archetype(
                    prior.pitch_count,
                    min_prior_pitches,
                    chase_rate,
                    whiff_rate,
                    called_strike_rate,
                ),
            }
        )
        output[target["_row_index"]] = out_row
    return pl.DataFrame([row for row in output if row is not None])


@dataclass
class _PitchTendencyCounts:
    pitch_count: int = 0
    chase_swing_count: int = 0
    whiff_count: int = 0
    called_strike_count: int = 0


def _with_temporal_keys(row: dict[str, Any], index: int) -> dict[str, Any]:
    out = dict(row)
    out["_row_index"] = index
    out["_pitch_timestamp"] = _datetime_value(row["pitch_timestamp"])
    out["_as_of_timestamp"] = _datetime_value(row["as_of_timestamp"])
    return out


def _add_pitch_tendency(
    row: dict[str, Any],
    league_counts: _PitchTendencyCounts,
    batter_counts: dict[Any, _PitchTendencyCounts],
) -> None:
    batter_id = row.get("batter_id")
    batter_count = batter_counts.setdefault(batter_id, _PitchTendencyCounts())
    for counts in (league_counts, batter_count):
        counts.pitch_count += 1
        if _is_chase_swing(row):
            counts.chase_swing_count += 1
        if _is_whiff(row):
            counts.whiff_count += 1
        if _is_called_strike(row):
            counts.called_strike_count += 1


def _archetype(
    sample_size: int,
    min_prior_pitches: int,
    chase_rate: float,
    whiff_rate: float,
    called_strike_rate: float,
) -> str:
    if sample_size < min_prior_pitches:
        return "neutral_unknown"
    if chase_rate >= whiff_rate and chase_rate >= called_strike_rate and chase_rate > 0:
        return "chase_vulnerable"
    if whiff_rate >= called_strike_rate and whiff_rate > 0:
        return "whiff_vulnerable"
    if called_strike_rate > 0:
        return "called_strike_vulnerable"
    return "balanced"


def _is_chase_swing(row: dict[str, Any]) -> bool:
    return (
        str(row.get("relative_zone")) in _CHASE_ZONES and str(row.get("outcome")) in _SWING_OUTCOMES
    )


def _is_whiff(row: dict[str, Any]) -> bool:
    return str(row.get("outcome")) == "swinging_strike"


def _is_called_strike(row: dict[str, Any]) -> bool:
    return str(row.get("outcome")) == "called_strike"


def _shrunk_rate(
    rows: list[dict[str, Any]],
    league_rows: list[dict[str, Any]],
    predicate: Any,
    prior: int,
) -> float:
    league_rate = _rate(league_rows, predicate)
    if not rows:
        return league_rate
    return (sum(1 for row in rows if predicate(row)) + prior * league_rate) / (len(rows) + prior)


def _rate(rows: list[dict[str, Any]], predicate: Any) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)


def _shrunk_count_rate(
    count: int,
    total: int,
    league_count: int,
    league_total: int,
    prior: int,
) -> float:
    league_rate = league_count / league_total if league_total else 0.0
    if not total:
        return league_rate
    return (count + prior * league_rate) / (total + prior)


def _confidence(sample_size: int, shrinkage_prior: int) -> float:
    denominator = sample_size + shrinkage_prior
    return sample_size / denominator if denominator else 0.0


def _require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _datetime_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
