from datetime import date, datetime

import polars as pl

from baseball_zerobase.profiles.batter_threat import add_batter_threat


def test_batter_threat_uses_prior_terminal_outcomes_with_shrinkage() -> None:
    frame = pl.DataFrame(
        {
            "game_pk": [1, 2, 3],
            "game_date": [date(2024, 4, 1), date(2024, 4, 2), date(2024, 4, 8)],
            "pitch_timestamp": [
                datetime(2024, 4, 1, 18, 0),
                datetime(2024, 4, 2, 18, 0),
                datetime(2024, 4, 8, 18, 0),
            ],
            "as_of_timestamp": [
                datetime(2024, 4, 1, 17, 59),
                datetime(2024, 4, 2, 17, 59),
                datetime(2024, 4, 8, 17, 59),
            ],
            "batter_id": [201, 201, 201],
            "outcome": ["home_run", "strikeout", "single"],
            "plate_appearance_ended": [True, True, True],
        }
    )

    featured = add_batter_threat(frame, shrinkage_prior_pas=2)
    target = featured.row(2, named=True)

    assert target["batter_threat_sample_size"] == 2
    assert 0.0 < target["batter_threat_score"] < 1.0
    assert target["batter_threat_home_run_rate"] > 0.0
    assert target["batter_threat_as_of_timestamp"] < target["pitch_timestamp"]
