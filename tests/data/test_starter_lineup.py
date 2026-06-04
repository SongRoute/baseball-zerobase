from __future__ import annotations

import polars as pl
import pytest

from baseball_zerobase.data.starter_lineup import attach_starter_and_lineup_context


def test_keeps_only_official_starter_pitches(statcast_frame, normalized_game_frame) -> None:
    result = attach_starter_and_lineup_context(statcast_frame, normalized_game_frame)
    assert result["is_official_starter_pitch"].to_list() == [True, True, False]


def test_marks_snapshots_unstable_at_first_substitution(statcast_frame, normalized_game_frame) -> None:
    result = attach_starter_and_lineup_context(statcast_frame, normalized_game_frame)
    assert result["lineup_stable"].to_list() == [True, True, False]


def test_preserves_rows_and_attaches_lineup_context(statcast_frame, normalized_game_frame) -> None:
    result = attach_starter_and_lineup_context(statcast_frame, normalized_game_frame)

    assert result.height == statcast_frame.height
    assert result["expected_starter_id"].to_list() == [501, 601, 501]
    assert result["offense_initial_lineup"].to_list() == [
        [201, 202, 203, 204, 205, 206, 207, 208, 209],
        [101, 102, 103, 104, 105, 106, 107, 108, 109],
        [201, 202, 203, 204, 205, 206, 207, 208, 209],
    ]
    assert result["offense_initial_lineup_stands"].to_list() == [
        ["L", "R", "L", "R", "L", "R", "L", "R", "L"],
        ["R", "L", "R", "L", "R", "L", "R", "L", "R"],
        ["L", "R", "L", "R", "L", "R", "L", "R", "L"],
    ]
    assert result["current_lineup_slot"].to_list() == [1, 1, 2]


def test_unmatched_games_preserve_rows_without_stable_lineup_context(
    statcast_frame,
    normalized_game_frame,
) -> None:
    unmatched_statcast_frame = pl.concat(
        [
            statcast_frame,
            statcast_frame.head(1).with_columns(game_pk=pl.lit(99999, dtype=pl.Int64)),
        ]
    )

    result = attach_starter_and_lineup_context(unmatched_statcast_frame, normalized_game_frame)

    assert result.height == unmatched_statcast_frame.height
    unmatched_row = result.filter(pl.col("game_pk") == 99999).row(0, named=True)
    assert unmatched_row["expected_starter_id"] is None
    assert unmatched_row["offense_initial_lineup"] is None
    assert unmatched_row["offense_initial_lineup_stands"] is None
    assert unmatched_row["is_official_starter_pitch"] is False
    assert unmatched_row["lineup_stable"] is False
    assert unmatched_row["current_lineup_slot"] is None


def test_rejects_duplicate_normalized_game_keys(normalized_game_frame, statcast_frame) -> None:
    duplicated_game_frame = pl.concat([normalized_game_frame, normalized_game_frame])

    with pytest.raises(
        ValueError,
        match="normalized_game_frame must contain unique game_pk values; duplicates: 12345",
    ):
        attach_starter_and_lineup_context(statcast_frame, duplicated_game_frame)
