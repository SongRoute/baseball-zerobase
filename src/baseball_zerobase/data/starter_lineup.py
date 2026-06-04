from __future__ import annotations

from typing import Any

import polars as pl


def _lineup_slot(row: dict[str, Any]) -> int | None:
    lineup = row["offense_initial_lineup"]
    batter = row["batter"]
    if lineup is None or batter is None:
        return None

    try:
        return list(lineup).index(batter) + 1
    except ValueError:
        return None


def attach_starter_and_lineup_context(
    statcast_frame: pl.DataFrame,
    normalized_game_frame: pl.DataFrame,
) -> pl.DataFrame:
    half_inning = pl.col("inning_topbot").str.to_lowercase()

    joined = statcast_frame.join(normalized_game_frame, on="game_pk", how="left")

    return (
        joined.with_columns(
            expected_starter_id=pl.when(half_inning == "top")
            .then(pl.col("home_starter_id"))
            .when(half_inning == "bottom")
            .then(pl.col("away_starter_id"))
            .otherwise(None),
            offense_initial_lineup=pl.when(half_inning == "top")
            .then(pl.col("away_initial_lineup"))
            .when(half_inning == "bottom")
            .then(pl.col("home_initial_lineup"))
            .otherwise(None),
            offense_initial_lineup_stands=pl.when(half_inning == "top")
            .then(pl.col("away_initial_lineup_stands"))
            .when(half_inning == "bottom")
            .then(pl.col("home_initial_lineup_stands"))
            .otherwise(None),
        )
        .with_columns(
            is_official_starter_pitch=pl.col("pitcher") == pl.col("expected_starter_id"),
            lineup_stable=pl.col("first_substitution_at_bat").is_null()
            | (pl.col("at_bat_number") < pl.col("first_substitution_at_bat")),
            current_lineup_slot=pl.struct(["batter", "offense_initial_lineup"]).map_elements(
                _lineup_slot,
                return_dtype=pl.Int64,
            ),
        )
    )
