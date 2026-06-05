from datetime import date, datetime

import polars as pl

from baseball_zerobase.profiles.daily_state import add_daily_state


def test_daily_state_counts_only_prior_same_game_information() -> None:
    frame = pl.DataFrame(
        {
            "game_pk": [1, 1, 1],
            "game_date": [date(2024, 4, 1)] * 3,
            "pitch_timestamp": [
                datetime(2024, 4, 1, 18, 0),
                datetime(2024, 4, 1, 18, 1),
                datetime(2024, 4, 1, 18, 2),
            ],
            "as_of_timestamp": [
                datetime(2024, 4, 1, 17, 59),
                datetime(2024, 4, 1, 18, 0, 10),
                datetime(2024, 4, 1, 18, 1, 10),
            ],
            "at_bat_number": [1, 1, 2],
            "pitch_number": [1, 2, 1],
            "pitcher_id": [501, 501, 501],
            "batter_id": [201, 201, 202],
            "balls": [0, 1, 0],
            "strikes": [0, 0, 0],
            "outcome": ["ball", "home_run", "called_strike"],
            "runs_scored": [0, 4, 0],
            "plate_appearance_ended": [False, True, False],
        }
    )

    featured = add_daily_state(frame)
    first = featured.row(0, named=True)
    third = featured.row(2, named=True)

    assert first["pitcher_day_prior_pitch_count"] == 0
    assert third["pitcher_day_prior_pitch_count"] == 2
    assert third["pitcher_day_prior_runs_allowed"] == 4
    assert third["batter_day_prior_pitch_count"] == 0
    assert third["daily_state_as_of_timestamp"] < third["pitch_timestamp"]
