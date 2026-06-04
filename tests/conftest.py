from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest


@pytest.fixture
def statcast_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [12345, 12345, 12345],
            "pitcher": [501, 601, 777],
            "batter": [201, 101, 202],
            "at_bat_number": [1, 2, 3],
            "pitch_number": [1, 1, 1],
            "inning": [1, 1, 2],
            "inning_topbot": ["Top", "Bottom", "Top"],
            "pitch_type": ["FF", "SL", "CH"],
        }
    )


@pytest.fixture
def normalized_game_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [12345],
            "game_date": ["2024-04-01"],
            "game_type": ["R"],
            "home_team_id": [119],
            "away_team_id": [137],
            "home_starter_id": [501],
            "away_starter_id": [601],
            "home_starter_throws": ["R"],
            "away_starter_throws": ["L"],
            "home_initial_lineup": [[101, 102, 103, 104, 105, 106, 107, 108, 109]],
            "away_initial_lineup": [[201, 202, 203, 204, 205, 206, 207, 208, 209]],
            "home_initial_lineup_stands": [["R", "L", "R", "L", "R", "L", "R", "L", "R"]],
            "away_initial_lineup_stands": [["L", "R", "L", "R", "L", "R", "L", "R", "L"]],
            "game_start_timestamp": [datetime(2024, 4, 1, 23, 5)],
            "first_substitution_at_bat": [3],
        }
    )


@pytest.fixture
def prepared_pitch_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [12345, 12345, 12345],
            "at_bat_number": [1, 1, 1],
            "pitch_number": [1, 2, 3],
            "pitcher": [501, 501, 501],
            "batter": [201, 201, 201],
            "pitch_type": ["FF", "FF", "SL"],
            "events": [None, None, "strikeout"],
            "description": ["ball", "called_strike", "swinging_strike"],
        }
    )


@pytest.fixture
def starter_snapshot_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [12345, 12346],
            "game_start_timestamp": [datetime(2024, 4, 1, 23, 5), datetime(2024, 4, 7, 20, 10)],
            "starter_id": [501, 501],
            "pitch_count": [86, 92],
            "eligible_pitch_types": [["FF", "SL"], ["FF", "SL", "CH"]],
        }
    )


@pytest.fixture
def baseline_snapshot_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "snapshot_id": [1, 1, 2],
            "action": ["FF", "SL", "FF"],
            "transition_atom": ["strike", "ball", "in_play_out"],
            "count": [7, 4, 3],
        }
    )


@pytest.fixture
def valid_snapshot_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "snapshot_id": [1, 2],
            "game_start_timestamp": [datetime(2024, 4, 1, 23, 5), datetime(2024, 4, 7, 20, 10)],
            "feature_available_at": [datetime(2024, 4, 1, 23, 4), datetime(2024, 4, 7, 20, 9)],
            "label_available_at": [datetime(2024, 4, 1, 23, 6), datetime(2024, 4, 7, 20, 11)],
        }
    )


@pytest.fixture
def leaky_snapshot_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "snapshot_id": [1],
            "game_start_timestamp": [datetime(2024, 4, 1, 23, 5)],
            "feature_available_at": [datetime(2024, 4, 1, 23, 6)],
            "label_available_at": [datetime(2024, 4, 1, 23, 7)],
        }
    )


@pytest.fixture
def initial_game_state() -> dict[str, object]:
    return {
        "balls": 0,
        "strikes": 0,
        "outs": 0,
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
    }


@pytest.fixture
def fitted_baselines() -> dict[str, object]:
    return {
        "pitch_type_rates": {"FF": 0.6, "SL": 0.4},
        "transition_rates": {"strike": 0.5, "ball": 0.3, "in_play_out": 0.2},
    }
