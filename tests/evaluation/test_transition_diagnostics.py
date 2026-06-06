from __future__ import annotations

import polars as pl
import pytest

from baseball_zerobase.evaluation.diagnostics import transition_diagnostics


def test_transition_diagnostics_reports_profile_and_label_coverage() -> None:
    frame = pl.DataFrame(
        {
            "pitch_type": ["FF", "SL", "FF"],
            "relative_zone": ["middle_middle", "chase_low", "middle_middle"],
            "batter_weakness_archetype": [
                "chase_vulnerable",
                "neutral_unknown",
                "chase_vulnerable",
            ],
            "batter_threat_score": [0.1, 0.5, None],
            "pitcher_profile_reliability_weight": [0.9, 0.4, None],
            "pitcher_pitch_type_owned": [True, False, True],
            "pitcher_profile_prior_pitch_count": [100, 50, None],
            "batter_threat_sample_size": [10, 0, None],
            "batter_weakness_sample_size": [20, 1, None],
            "daily_state_as_of_timestamp": [1, 2, 3],
            "pitcher_day_prior_pitch_count": [0, 7, 12],
            "batter_day_prior_seen_pitcher_pitch_count": [0, 2, 4],
            "outcome": ["ball", "home_run", "ball"],
        }
    )

    report = transition_diagnostics(frame)

    assert report.row_count == 3
    assert report.pitch_type_distribution == {"FF": 2, "SL": 1}
    assert report.relative_zone_distribution == {"chase_low": 1, "middle_middle": 2}
    assert report.batter_weakness_archetype_distribution == {
        "chase_vulnerable": 2,
        "neutral_unknown": 1,
    }
    assert report.batter_threat_score_bucket_distribution == {
        "null": 1,
        "low": 1,
        "medium": 1,
        "high": 0,
    }
    assert report.pitcher_profile_reliability_weight_bucket_distribution == {
        "null": 1,
        "low": 0,
        "medium": 1,
        "high": 1,
    }
    assert report.profile_feature_null_rates["pitcher_profile_prior_pitch_count"] == pytest.approx(
        1 / 3
    )
    assert report.pitcher_pitch_type_owned_true_rate == pytest.approx(2 / 3)
    assert report.pitcher_pitch_type_owned_counts == {"true": 2, "false": 1, "null": 0}
    assert report.daily_state_count_summary["pitcher_day_prior_pitch_count"]["max"] == 12
    assert report.label_outcome_distribution == {"ball": 2, "home_run": 1}
