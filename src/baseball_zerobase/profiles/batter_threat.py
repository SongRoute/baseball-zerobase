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
    rows = [
        _with_temporal_keys(row, index)
        for index, row in enumerate(snapshot_frame.iter_rows(named=True))
    ]
    output: list[dict[str, Any] | None] = [None] * len(rows)
    prior_rows = sorted(rows, key=lambda row: row["_pitch_timestamp"])
    targets = sorted(rows, key=lambda row: row["_as_of_timestamp"])
    batter_counts: dict[Any, _ThreatCounts] = {}
    league_counts = _ThreatCounts()
    prior_index = 0

    for target in targets:
        as_of = target["_as_of_timestamp"]
        while prior_index < len(prior_rows) and prior_rows[prior_index]["_pitch_timestamp"] < as_of:
            if bool(prior_rows[prior_index].get("plate_appearance_ended")):
                _add_threat_counts(prior_rows[prior_index], league_counts, batter_counts)
            prior_index += 1

        prior = batter_counts.get(target.get("batter_id"), _ThreatCounts())
        reach_rate = _shrunk_count_rate(
            prior.reach_count,
            prior.pa_count,
            league_counts.reach_count,
            league_counts.pa_count,
            shrinkage_prior_pas,
        )
        extra_base_rate = _shrunk_count_rate(
            prior.extra_base_count,
            prior.pa_count,
            league_counts.extra_base_count,
            league_counts.pa_count,
            shrinkage_prior_pas,
        )
        home_run_rate = _shrunk_count_rate(
            prior.home_run_count,
            prior.pa_count,
            league_counts.home_run_count,
            league_counts.pa_count,
            shrinkage_prior_pas,
        )
        strikeout_rate = _shrunk_count_rate(
            prior.strikeout_count,
            prior.pa_count,
            league_counts.strikeout_count,
            league_counts.pa_count,
            shrinkage_prior_pas,
        )
        score = min(
            1.0, max(0.0, 0.45 * reach_rate + 0.35 * extra_base_rate + 0.20 * home_run_rate)
        )
        out_row = dict(target)
        out_row.pop("_row_index")
        out_row.pop("_pitch_timestamp")
        out_row.pop("_as_of_timestamp")
        out_row.update(
            {
                "batter_threat_as_of_timestamp": as_of,
                "batter_threat_sample_size": prior.pa_count,
                "batter_threat_confidence": _confidence(prior.pa_count, shrinkage_prior_pas),
                "batter_threat_reach_rate": reach_rate,
                "batter_threat_extra_base_rate": extra_base_rate,
                "batter_threat_home_run_rate": home_run_rate,
                "batter_threat_strikeout_rate": strikeout_rate,
                "batter_threat_score": score,
            }
        )
        output[target["_row_index"]] = out_row
    return pl.DataFrame([row for row in output if row is not None])


@dataclass
class _ThreatCounts:
    pa_count: int = 0
    reach_count: int = 0
    extra_base_count: int = 0
    home_run_count: int = 0
    strikeout_count: int = 0


def _with_temporal_keys(row: dict[str, Any], index: int) -> dict[str, Any]:
    out = dict(row)
    out["_row_index"] = index
    out["_pitch_timestamp"] = _datetime_value(row["pitch_timestamp"])
    out["_as_of_timestamp"] = _datetime_value(row["as_of_timestamp"])
    return out


def _add_threat_counts(
    row: dict[str, Any],
    league_counts: _ThreatCounts,
    batter_counts: dict[Any, _ThreatCounts],
) -> None:
    batter_id = row.get("batter_id")
    batter_count = batter_counts.setdefault(batter_id, _ThreatCounts())
    for counts in (league_counts, batter_count):
        counts.pa_count += 1
        if _is_reach(row):
            counts.reach_count += 1
        if _is_extra_base(row):
            counts.extra_base_count += 1
        if _is_home_run(row):
            counts.home_run_count += 1
        if _is_strikeout(row):
            counts.strikeout_count += 1


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
