from __future__ import annotations

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
    rows = list(snapshot_frame.iter_rows(named=True))
    output: list[dict[str, Any]] = []
    for target in rows:
        prior = [
            row
            for row in rows
            if row.get("batter_id") == target.get("batter_id")
            and _datetime_value(row["pitch_timestamp"]) < _datetime_value(target["as_of_timestamp"])
        ]
        league_prior = [
            row
            for row in rows
            if _datetime_value(row["pitch_timestamp"]) < _datetime_value(target["as_of_timestamp"])
        ]
        chase_rate = _shrunk_rate(prior, league_prior, _is_chase_swing, shrinkage_prior_pitches)
        whiff_rate = _shrunk_rate(prior, league_prior, _is_whiff, shrinkage_prior_pitches)
        called_strike_rate = _shrunk_rate(
            prior, league_prior, _is_called_strike, shrinkage_prior_pitches
        )
        confidence = len(prior) / (len(prior) + shrinkage_prior_pitches)
        out_row = dict(target)
        out_row.update(
            {
                "batter_weakness_as_of_timestamp": _datetime_value(target["as_of_timestamp"]),
                "batter_weakness_sample_size": len(prior),
                "batter_weakness_confidence": confidence,
                "batter_weakness_chase_swing_rate": chase_rate,
                "batter_weakness_whiff_rate": whiff_rate,
                "batter_weakness_called_strike_rate": called_strike_rate,
                "batter_weakness_archetype": _archetype(
                    len(prior),
                    min_prior_pitches,
                    chase_rate,
                    whiff_rate,
                    called_strike_rate,
                ),
            }
        )
        output.append(out_row)
    return pl.DataFrame(output)


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


def _require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _datetime_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
