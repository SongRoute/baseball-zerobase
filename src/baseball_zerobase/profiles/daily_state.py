from __future__ import annotations

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
    rows = list(snapshot_frame.iter_rows(named=True))
    output: list[dict[str, Any]] = []
    for target in rows:
        as_of = _datetime_value(target["as_of_timestamp"])
        game_pk = target.get("game_pk")
        pitcher_id = target.get("pitcher_id")
        batter_id = target.get("batter_id")
        prior_game = [
            row
            for row in rows
            if row.get("game_pk") == game_pk and _datetime_value(row["pitch_timestamp"]) < as_of
        ]
        prior_pitcher = [row for row in prior_game if row.get("pitcher_id") == pitcher_id]
        prior_batter = [row for row in prior_game if row.get("batter_id") == batter_id]
        prior_matchup = [row for row in prior_batter if row.get("pitcher_id") == pitcher_id]
        out_row = dict(target)
        out_row.update(
            {
                "daily_state_as_of_timestamp": as_of,
                "pitcher_day_prior_pitch_count": len(prior_pitcher),
                "pitcher_day_prior_pa_count": sum(
                    1 for row in prior_pitcher if bool(row.get("plate_appearance_ended"))
                ),
                "pitcher_day_prior_batter_count": len(
                    {row.get("batter_id") for row in prior_pitcher}
                ),
                "pitcher_day_prior_runs_allowed": sum(
                    max(0, int(row.get("runs_scored") or 0)) for row in prior_pitcher
                ),
                "pitcher_day_prior_balls": sum(
                    1 for row in prior_pitcher if str(row.get("outcome")) == "ball"
                ),
                "pitcher_day_prior_strikes": sum(
                    1
                    for row in prior_pitcher
                    if str(row.get("outcome")) in {"called_strike", "swinging_strike"}
                ),
                "pitcher_day_prior_in_play": sum(
                    1
                    for row in prior_pitcher
                    if str(row.get("outcome"))
                    in {"in_play_out", "single", "double", "triple", "home_run"}
                ),
                "batter_day_prior_pitch_count": len(prior_batter),
                "batter_day_prior_pa_count": sum(
                    1 for row in prior_batter if bool(row.get("plate_appearance_ended"))
                ),
                "batter_day_prior_seen_pitcher_pitch_count": len(prior_matchup),
                "batter_day_prior_seen_pitcher_pa_count": sum(
                    1 for row in prior_matchup if bool(row.get("plate_appearance_ended"))
                ),
            }
        )
        output.append(out_row)
    return pl.DataFrame(output)


def _require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _datetime_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
