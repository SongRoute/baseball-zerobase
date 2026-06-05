from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import polars as pl


_REQUIRED_COLUMNS = {
    "as_of_timestamp",
    "at_bat_number",
    "batter_id",
    "game_pk",
    "outcome",
    "pitch_number",
    "pitch_timestamp",
    "pitcher_id",
    "plate_appearance_ended",
    "runs_scored",
}


def add_daily_state(snapshot_frame: pl.DataFrame) -> pl.DataFrame:
    _require_columns(snapshot_frame, _REQUIRED_COLUMNS, "daily state frame")
    rows = [
        _with_temporal_keys(row, index)
        for index, row in enumerate(snapshot_frame.iter_rows(named=True))
    ]
    output: list[dict[str, Any] | None] = [None] * len(rows)
    prior_rows = sorted(rows, key=lambda row: row["_pitch_timestamp"])
    targets = sorted(rows, key=lambda row: row["_as_of_timestamp"])
    pitcher_counts: dict[tuple[Any, Any], _PitcherDayCounts] = {}
    batter_counts: dict[tuple[Any, Any], _BatterDayCounts] = {}
    matchup_counts: dict[tuple[Any, Any, Any], _BatterDayCounts] = {}
    prior_index = 0

    for target in targets:
        as_of = target["_as_of_timestamp"]
        while prior_index < len(prior_rows) and prior_rows[prior_index]["_pitch_timestamp"] < as_of:
            _add_daily_counts(
                prior_rows[prior_index], pitcher_counts, batter_counts, matchup_counts
            )
            prior_index += 1

        game_pk = target.get("game_pk")
        pitcher_id = target.get("pitcher_id")
        batter_id = target.get("batter_id")
        pitcher = pitcher_counts.get((game_pk, pitcher_id), _PitcherDayCounts())
        batter = batter_counts.get((game_pk, batter_id), _BatterDayCounts())
        matchup = matchup_counts.get((game_pk, pitcher_id, batter_id), _BatterDayCounts())
        out_row = dict(target)
        out_row.pop("_row_index")
        out_row.pop("_pitch_timestamp")
        out_row.pop("_as_of_timestamp")
        out_row.update(
            {
                "daily_state_as_of_timestamp": as_of,
                "pitcher_day_prior_pitch_count": pitcher.pitch_count,
                "pitcher_day_prior_pa_count": pitcher.pa_count,
                "pitcher_day_prior_batter_count": len(pitcher.batter_ids),
                "pitcher_day_prior_runs_allowed": pitcher.runs_allowed,
                "pitcher_day_prior_balls": pitcher.balls,
                "pitcher_day_prior_strikes": pitcher.strikes,
                "pitcher_day_prior_in_play": pitcher.in_play,
                "batter_day_prior_pitch_count": batter.pitch_count,
                "batter_day_prior_pa_count": batter.pa_count,
                "batter_day_prior_seen_pitcher_pitch_count": matchup.pitch_count,
                "batter_day_prior_seen_pitcher_pa_count": matchup.pa_count,
            }
        )
        output[target["_row_index"]] = out_row
    return pl.DataFrame([row for row in output if row is not None])


@dataclass
class _PitcherDayCounts:
    pitch_count: int = 0
    pa_count: int = 0
    batter_ids: set[Any] = field(default_factory=set)
    runs_allowed: int = 0
    balls: int = 0
    strikes: int = 0
    in_play: int = 0


@dataclass
class _BatterDayCounts:
    pitch_count: int = 0
    pa_count: int = 0


def _with_temporal_keys(row: dict[str, Any], index: int) -> dict[str, Any]:
    out = dict(row)
    out["_row_index"] = index
    out["_pitch_timestamp"] = _datetime_value(row["pitch_timestamp"])
    out["_as_of_timestamp"] = _datetime_value(row["as_of_timestamp"])
    return out


def _add_daily_counts(
    row: dict[str, Any],
    pitcher_counts: dict[tuple[Any, Any], _PitcherDayCounts],
    batter_counts: dict[tuple[Any, Any], _BatterDayCounts],
    matchup_counts: dict[tuple[Any, Any, Any], _BatterDayCounts],
) -> None:
    game_pk = row.get("game_pk")
    pitcher_id = row.get("pitcher_id")
    batter_id = row.get("batter_id")
    pitcher = pitcher_counts.setdefault((game_pk, pitcher_id), _PitcherDayCounts())
    batter = batter_counts.setdefault((game_pk, batter_id), _BatterDayCounts())
    matchup = matchup_counts.setdefault((game_pk, pitcher_id, batter_id), _BatterDayCounts())
    ended = bool(row.get("plate_appearance_ended"))
    outcome = str(row.get("outcome"))

    pitcher.pitch_count += 1
    pitcher.pa_count += int(ended)
    pitcher.batter_ids.add(batter_id)
    pitcher.runs_allowed += max(0, int(row.get("runs_scored") or 0))
    pitcher.balls += int(outcome == "ball")
    pitcher.strikes += int(outcome in {"called_strike", "swinging_strike"})
    pitcher.in_play += int(outcome in {"in_play_out", "single", "double", "triple", "home_run"})

    batter.pitch_count += 1
    batter.pa_count += int(ended)
    matchup.pitch_count += 1
    matchup.pa_count += int(ended)


def _require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _datetime_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
