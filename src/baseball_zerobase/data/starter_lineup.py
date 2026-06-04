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


def _validate_unique_game_pk(normalized_game_frame: pl.DataFrame) -> None:
    duplicate_game_pks = (
        normalized_game_frame.group_by("game_pk")
        .agg(pl.len().alias("row_count"))
        .filter(pl.col("row_count") > 1)
        .sort("game_pk")
        .get_column("game_pk")
        .to_list()
    )
    if duplicate_game_pks:
        duplicate_list = ", ".join(str(game_pk) for game_pk in duplicate_game_pks)
        raise ValueError(
            "normalized_game_frame must contain unique game_pk values; "
            f"duplicates: {duplicate_list}"
        )


def attach_starter_and_lineup_context(
    statcast_frame: pl.DataFrame,
    normalized_game_frame: pl.DataFrame,
) -> pl.DataFrame:
    _validate_unique_game_pk(normalized_game_frame)

    half_inning = pl.col("inning_topbot").str.to_lowercase()

    joined = statcast_frame.join(normalized_game_frame, on="game_pk", how="left")
    has_lineup_context = (
        pl.col("expected_starter_id").is_not_null()
        & pl.col("offense_initial_lineup").is_not_null()
        & pl.col("offense_initial_lineup_stands").is_not_null()
    )

    return joined.with_columns(
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
    ).with_columns(
        is_official_starter_pitch=(pl.col("pitcher") == pl.col("expected_starter_id")).fill_null(
            False
        ),
        lineup_stable=has_lineup_context
        & (
            pl.col("first_substitution_at_bat").is_null()
            | (pl.col("at_bat_number") < pl.col("first_substitution_at_bat"))
        ),
        current_lineup_slot=pl.struct(["batter", "offense_initial_lineup"]).map_elements(
            _lineup_slot,
            return_dtype=pl.Int64,
        ),
    )
