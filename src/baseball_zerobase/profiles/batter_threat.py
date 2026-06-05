from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl


_REQUIRED_COLUMNS = {
    "as_of_timestamp",
    "batter_id",
    "outcome",
    "pitch_timestamp",
    "plate_appearance_ended",
}
_REACH = {"single", "double", "triple", "home_run", "walk", "hit_by_pitch", "reach_other"}
_EXTRA_BASE = {"double", "triple", "home_run"}


def add_batter_threat(
    snapshot_frame: pl.DataFrame,
    *,
    shrinkage_prior_pas: int = 50,
) -> pl.DataFrame:
    if shrinkage_prior_pas < 0:
        raise ValueError("shrinkage_prior_pas must be non-negative")
    _require_columns(snapshot_frame, _REQUIRED_COLUMNS, "batter threat frame")
    rows = list(snapshot_frame.iter_rows(named=True))
    output: list[dict[str, Any]] = []
    for target in rows:
        prior = [
            row
            for row in rows
            if row.get("batter_id") == target.get("batter_id")
            and bool(row.get("plate_appearance_ended"))
            and _datetime_value(row["pitch_timestamp"]) < _datetime_value(target["as_of_timestamp"])
        ]
        league_prior = [
            row
            for row in rows
            if bool(row.get("plate_appearance_ended"))
            and _datetime_value(row["pitch_timestamp"]) < _datetime_value(target["as_of_timestamp"])
        ]
        reach_rate = _shrunk_rate(prior, league_prior, _is_reach, shrinkage_prior_pas)
        extra_base_rate = _shrunk_rate(prior, league_prior, _is_extra_base, shrinkage_prior_pas)
        home_run_rate = _shrunk_rate(prior, league_prior, _is_home_run, shrinkage_prior_pas)
        strikeout_rate = _shrunk_rate(prior, league_prior, _is_strikeout, shrinkage_prior_pas)
        score = min(
            1.0, max(0.0, 0.45 * reach_rate + 0.35 * extra_base_rate + 0.20 * home_run_rate)
        )
        out_row = dict(target)
        out_row.update(
            {
                "batter_threat_as_of_timestamp": _datetime_value(target["as_of_timestamp"]),
                "batter_threat_sample_size": len(prior),
                "batter_threat_confidence": len(prior) / (len(prior) + shrinkage_prior_pas),
                "batter_threat_reach_rate": reach_rate,
                "batter_threat_extra_base_rate": extra_base_rate,
                "batter_threat_home_run_rate": home_run_rate,
                "batter_threat_strikeout_rate": strikeout_rate,
                "batter_threat_score": score,
            }
        )
        output.append(out_row)
    return pl.DataFrame(output)


def _is_reach(row: dict[str, Any]) -> bool:
    return str(row.get("outcome")) in _REACH


def _is_extra_base(row: dict[str, Any]) -> bool:
    return str(row.get("outcome")) in _EXTRA_BASE


def _is_home_run(row: dict[str, Any]) -> bool:
    return str(row.get("outcome")) == "home_run"


def _is_strikeout(row: dict[str, Any]) -> bool:
    return str(row.get("outcome")) == "strikeout"


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
