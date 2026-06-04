from __future__ import annotations

from datetime import date, datetime
from typing import Any

import polars as pl


ELIGIBILITY_COLUMNS = [
    "prior_two_season_starter_pitches",
    "current_season_prior_pitches",
    "starter_eligible",
]

_REQUIRED_COLUMNS = {
    "game_pk",
    "game_date",
    "pitcher_id",
    "is_official_starter_pitch",
}


def add_starter_eligibility(
    snapshot_frame: pl.DataFrame,
    *,
    min_prior_pitches: int,
) -> pl.DataFrame:
    """Attach as-of starter eligibility without counting the current game."""

    if min_prior_pitches < 0:
        raise ValueError("min_prior_pitches must be non-negative")
    missing = sorted(_REQUIRED_COLUMNS.difference(snapshot_frame.columns))
    if missing:
        raise ValueError(f"starter eligibility frame is missing columns: {missing}")
    if snapshot_frame.is_empty():
        return snapshot_frame.with_columns(
            pl.lit(0, dtype=pl.Int64).alias("prior_two_season_starter_pitches"),
            pl.lit(0, dtype=pl.Int64).alias("current_season_prior_pitches"),
            pl.lit(False, dtype=pl.Boolean).alias("starter_eligible"),
        )

    pitcher_games = (
        snapshot_frame.group_by(["pitcher_id", "game_pk", "game_date"])
        .agg(
            pl.col("is_official_starter_pitch")
            .fill_null(False)
            .cast(pl.Int64)
            .sum()
            .alias("starter_pitches")
        )
        .sort(["pitcher_id", "game_date", "game_pk"])
    )
    metrics = _prior_pitch_metrics(pitcher_games, min_prior_pitches)
    return snapshot_frame.join(metrics, on=["pitcher_id", "game_pk"], how="left").with_columns(
        pl.col("prior_two_season_starter_pitches").fill_null(0).cast(pl.Int64),
        pl.col("current_season_prior_pitches").fill_null(0).cast(pl.Int64),
        pl.col("starter_eligible").fill_null(False).cast(pl.Boolean),
    )


def _prior_pitch_metrics(pitcher_games: pl.DataFrame, min_prior_pitches: int) -> pl.DataFrame:
    rows = pitcher_games.iter_rows(named=True)
    by_pitcher: dict[int | None, list[dict[str, Any]]] = {}
    for row in rows:
        pitcher_id = _integer_or_none(row["pitcher_id"])
        by_pitcher.setdefault(pitcher_id, []).append(row)

    metric_rows: list[dict[str, Any]] = []
    for pitcher_rows in by_pitcher.values():
        ordered_rows = sorted(
            pitcher_rows,
            key=lambda row: (_date_value(row["game_date"]), int(row["game_pk"])),
        )
        prior_rows: list[dict[str, Any]] = []
        for row in ordered_rows:
            game_date = _date_value(row["game_date"])
            cutoff = _two_year_cutoff(game_date)
            prior_two_season = sum(
                int(prior["starter_pitches"])
                for prior in prior_rows
                if cutoff <= _date_value(prior["game_date"]) < game_date
            )
            current_season = sum(
                int(prior["starter_pitches"])
                for prior in prior_rows
                if _date_value(prior["game_date"]).year == game_date.year
                and _date_value(prior["game_date"]) < game_date
            )
            metric_rows.append(
                {
                    "pitcher_id": _integer_or_none(row["pitcher_id"]),
                    "game_pk": int(row["game_pk"]),
                    "prior_two_season_starter_pitches": prior_two_season,
                    "current_season_prior_pitches": current_season,
                    "starter_eligible": prior_two_season >= min_prior_pitches,
                }
            )
            prior_rows.append(row)

    return pl.DataFrame(
        metric_rows,
        schema={
            "pitcher_id": pl.Int64,
            "game_pk": pl.Int64,
            "prior_two_season_starter_pitches": pl.Int64,
            "current_season_prior_pitches": pl.Int64,
            "starter_eligible": pl.Boolean,
        },
    )


def _two_year_cutoff(value: date) -> date:
    try:
        return value.replace(year=value.year - 2)
    except ValueError:
        return date(value.year - 2, 2, 28)


def _date_value(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _integer_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
