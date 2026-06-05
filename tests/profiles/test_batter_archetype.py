from datetime import date, datetime

import polars as pl

from baseball_zerobase.profiles.batter_archetype import add_batter_archetypes


def test_batter_archetype_uses_prior_response_tendencies_without_threat_columns() -> None:
    frame = pl.DataFrame(
        {
            "game_pk": [1, 1, 2],
            "game_date": [date(2024, 4, 1), date(2024, 4, 1), date(2024, 4, 8)],
            "pitch_timestamp": [
                datetime(2024, 4, 1, 18, 0),
                datetime(2024, 4, 1, 18, 1),
                datetime(2024, 4, 8, 18, 0),
            ],
            "as_of_timestamp": [
                datetime(2024, 4, 1, 17, 59),
                datetime(2024, 4, 1, 18, 0, 10),
                datetime(2024, 4, 8, 17, 59),
            ],
            "batter_id": [201, 201, 201],
            "pitch_type": ["SL", "SL", "FF"],
            "relative_zone": ["chase_low", "chase_away", "middle_middle"],
            "outcome": ["swinging_strike", "swinging_strike", "home_run"],
        }
    )

    featured = add_batter_archetypes(frame, min_prior_pitches=2, shrinkage_prior_pitches=1)
    target = featured.row(2, named=True)

    assert target["batter_weakness_sample_size"] == 2
    assert target["batter_weakness_archetype"] == "chase_vulnerable"
    assert "batter_threat_score" not in featured.columns
