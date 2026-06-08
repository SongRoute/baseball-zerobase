from datetime import date, datetime, timedelta
from typing import cast

import polars as pl
import pytest

from baseball_zerobase.data.contracts import RelativeZone
from baseball_zerobase.inference.recommender import recommend_pitches
from baseball_zerobase.inference.schemas import PitchRecommendation, RecommendationReport
from baseball_zerobase.models.transition import SharedTransitionModelV0


def test_recommendations_score_full_pitch_type_zone_grid_without_zone_filtering() -> None:
    model = _fitted_model()

    report = recommend_pitches(
        model,
        _target_row(),
        pitch_types=["FF", "SL"],
        top_k=None,
    )

    assert report.candidate_count == 26
    assert report.zone_filtering == "disabled"
    assert {item.pitch_type for item in report.recommendations} == {"FF", "SL"}
    assert {item.relative_zone for item in report.recommendations if item.pitch_type == "FF"} == {
        zone.value for zone in RelativeZone
    }


def test_transition_risk_score_prefers_strike_distribution_to_home_run_distribution() -> None:
    model = _fitted_model()

    report = recommend_pitches(model, _target_row(), pitch_types=["FF", "SL"], top_k=None)
    ff_middle = _find(report, "FF", "middle_middle")
    sl_middle = _find(report, "SL", "middle_middle")
    ff_top_atoms = cast(list[dict[str, object]], ff_middle.explanation["top_transition_atoms"])
    sl_home_run_probability = cast(float, sl_middle.explanation["home_run_probability"])
    ff_home_run_probability = cast(float, ff_middle.explanation["home_run_probability"])

    assert ff_middle.ranking_score < sl_middle.ranking_score
    assert ff_top_atoms[0]["outcome"] == "called_strike"
    assert sl_home_run_probability > ff_home_run_probability


def test_candidate_owned_flag_is_recomputed_from_prior_pitch_type_list() -> None:
    model = _fitted_model()
    row = {**_target_row(), "pitch_type": "SL", "pitcher_pitch_type_owned": True}

    report = recommend_pitches(model, row, pitch_types=["FF", "SL"], top_k=None)

    assert _find(report, "FF", "middle_middle").explanation["pitcher_pitch_type_owned"] is True
    assert _find(report, "SL", "middle_middle").explanation["pitcher_pitch_type_owned"] is False


def test_target_labels_and_actual_action_do_not_change_recommendations() -> None:
    model = _fitted_model()
    original = recommend_pitches(model, _target_row(), pitch_types=["FF", "SL"], top_k=None)
    mutated = recommend_pitches(
        model,
        {
            **_target_row(),
            "pitch_type": "CU",
            "relative_zone": "chase_high",
            "action": "CU:chase_high",
            "outcome": "home_run",
            "runs_scored": 4,
            "balls_after": 0,
            "strikes_after": 0,
            "outs_after": 0,
            "runners_after": 0,
            "plate_appearance_ended": True,
            "half_inning_ended": False,
        },
        pitch_types=["FF", "SL"],
        top_k=None,
    )

    assert mutated.to_dict()["recommendations"] == original.to_dict()["recommendations"]


def test_recommendations_reject_feature_timestamps_at_or_after_target_pitch() -> None:
    row = {**_target_row(), "batter_profile_as_of_timestamp": _target_row()["pitch_timestamp"]}

    with pytest.raises(ValueError, match="must be before pitch_timestamp"):
        recommend_pitches(_fitted_model(), row, pitch_types=["FF"], top_k=None)


def test_recommendations_reject_base_as_of_timestamp_at_target_pitch() -> None:
    row = {**_target_row(), "as_of_timestamp": _target_row()["pitch_timestamp"]}

    with pytest.raises(ValueError, match="as_of_timestamp must be before pitch_timestamp"):
        recommend_pitches(_fitted_model(), row, pitch_types=["FF"], top_k=None)


@pytest.mark.parametrize("column", ["zone", "plate_x", "plate_z", "release_speed", "pfx_x"])
def test_recommendations_reject_raw_current_pitch_measurements(column: str) -> None:
    row = {**_target_row(), column: 1}

    with pytest.raises(ValueError, match="serving input cannot include current-pitch measurement"):
        recommend_pitches(_fitted_model(), row, pitch_types=["FF"], top_k=None)


def _fitted_model() -> SharedTransitionModelV0:
    return SharedTransitionModelV0(min_support=1, prior_weight=0.0).fit(
        _transition_training_frame(),
        training_manifest_hash="synthetic:m5",
    )


def _transition_training_frame() -> pl.DataFrame:
    start = datetime(2024, 4, 1, 18, 0)
    return pl.DataFrame(
        {
            "game_pk": [1, 2],
            "game_date": [date(2024, 4, 1), date(2024, 4, 2)],
            "game_type": ["R", "R"],
            "pitch_timestamp": [start, start + timedelta(days=1)],
            "as_of_timestamp": [
                start - timedelta(seconds=1),
                start + timedelta(days=1, seconds=-1),
            ],
            "pitch_type": ["FF", "SL"],
            "relative_zone": ["middle_middle", "middle_middle"],
            "action": ["FF:middle_middle", "SL:middle_middle"],
            "balls": [0, 0],
            "strikes": [0, 0],
            "outs": [0, 0],
            "runners": [0, 0],
            "stand": ["R", "R"],
            "p_throws": ["R", "R"],
            "outcome": ["called_strike", "home_run"],
            "balls_after": [0, 0],
            "strikes_after": [1, 0],
            "outs_after": [0, 0],
            "runners_after": [0, 0],
            "runs_scored": [0, 1],
            "plate_appearance_ended": [False, True],
            "half_inning_ended": [False, False],
            "terminal_reason": [None, None],
            "pitcher_owned_pitch_types": [["FF"], ["SL"]],
            "pitcher_pitch_type_owned": [True, True],
            "batter_weakness_archetype": ["chase_vulnerable", "chase_vulnerable"],
            "batter_threat_score": [0.25, 0.25],
        }
    )


def _target_row() -> dict[str, object]:
    pitch_timestamp = datetime(2024, 4, 3, 18, 0)
    return {
        "game_pk": 3,
        "game_date": date(2024, 4, 3),
        "game_type": "R",
        "pitch_timestamp": pitch_timestamp,
        "as_of_timestamp": pitch_timestamp - timedelta(seconds=1),
        "pitch_type": "FF",
        "relative_zone": "middle_middle",
        "action": "FF:middle_middle",
        "balls": 0,
        "strikes": 0,
        "outs": 0,
        "runners": 0,
        "stand": "R",
        "p_throws": "R",
        "outcome": "ball",
        "balls_after": 1,
        "strikes_after": 0,
        "outs_after": 0,
        "runners_after": 0,
        "runs_scored": 0,
        "plate_appearance_ended": False,
        "half_inning_ended": False,
        "terminal_reason": None,
        "pitcher_owned_pitch_types": ["FF"],
        "pitcher_pitch_type_owned": False,
        "batter_weakness_archetype": "chase_vulnerable",
        "batter_threat_score": 0.25,
    }


def _find(
    report: RecommendationReport,
    pitch_type: str,
    relative_zone: str,
) -> PitchRecommendation:
    for recommendation in report.recommendations:
        if (
            recommendation.pitch_type == pitch_type
            and recommendation.relative_zone == relative_zone
        ):
            return recommendation
    raise AssertionError(f"missing recommendation {pitch_type}:{relative_zone}")
