from datetime import date, datetime

import polars as pl

from baseball_zerobase.profiles.pitcher import add_pitcher_profiles


def test_pitcher_profile_excludes_target_game_and_marks_owned_pitch_type() -> None:
    frame = pl.DataFrame(
        {
            "game_pk": [1, 2, 3, 99],
            "game_date": [
                date(2023, 4, 1),
                date(2023, 4, 8),
                date(2024, 4, 1),
                date(2024, 4, 7),
            ],
            "pitch_timestamp": [
                datetime(2023, 4, 1, 18, 0),
                datetime(2023, 4, 8, 18, 0),
                datetime(2024, 4, 1, 18, 0),
                datetime(2024, 4, 7, 18, 0),
            ],
            "as_of_timestamp": [
                datetime(2023, 4, 1, 17, 59),
                datetime(2023, 4, 8, 17, 59),
                datetime(2024, 4, 1, 17, 59),
                datetime(2024, 4, 7, 17, 59),
            ],
            "pitcher_id": [501, 501, 501, 501],
            "pitch_type": ["FF", "FF", "SL", "FF"],
            "release_speed": [90.0, 92.0, 80.0, 100.0],
        }
    )

    profiled = add_pitcher_profiles(
        frame,
        min_pitch_type_pitches=2,
        min_pitch_type_usage=0.5,
        current_season_min_pitches=2,
        shrinkage_prior_pitches=1,
    )
    target = profiled.row(3, named=True)

    assert target["pitcher_profile_prior_pitch_count"] == 3
    assert target["pitcher_profile_current_season_pitch_count"] == 1
    assert target["pitcher_profile_pitch_type_prior_count"] == 2
    assert target["pitcher_pitch_type_owned"] is True
    assert target["pitcher_profile_uses_prior_season_shrinkage"] is True
    assert target["pitcher_profile_as_of_timestamp"] < target["pitch_timestamp"]
    assert target["pitcher_profile_release_speed_mean"] < 100.0
