import pytest

from baseball_zerobase.models.transition_context import transition_context_from_row


def test_context_excludes_target_label_columns() -> None:
    context = transition_context_from_row(
        {
            "pitch_type": "FF",
            "relative_zone": "middle_middle",
            "balls": 1,
            "strikes": 2,
            "outs": 1,
            "runners": 3,
            "stand": "R",
            "p_throws": "L",
            "batter_threat_score": 0.42,
            "outcome": "home_run",
            "balls_after": 0,
            "strikes_after": 0,
            "outs_after": 1,
            "runners_after": 0,
            "runs_scored": 1,
            "plate_appearance_ended": True,
            "half_inning_ended": False,
        }
    )

    assert context.pitch_type == "FF"
    assert context.relative_zone == "middle_middle"
    assert context.balls == 1
    assert context.features["batter_threat_score"] == 0.42
    assert "outcome" not in context.features
    assert "runs_scored" not in context.features
    assert "balls_after" not in context.features


def test_context_excludes_all_target_labels_and_raw_zone() -> None:
    context = transition_context_from_row(
        {
            "pitch_type": "FF",
            "relative_zone": "middle_middle",
            "zone": 5,
            "balls": 0,
            "strikes": 0,
            "outs": 0,
            "runners": 0,
            "outcome": "ball",
            "balls_after": 1,
            "strikes_after": 0,
            "outs_after": 0,
            "runners_after": 0,
            "runs_scored": 0,
            "plate_appearance_ended": False,
            "half_inning_ended": False,
            "terminal_reason": None,
            "transition_atom": "label",
        }
    )

    for column in {
        "outcome",
        "balls_after",
        "strikes_after",
        "outs_after",
        "runners_after",
        "runs_scored",
        "plate_appearance_ended",
        "half_inning_ended",
        "terminal_reason",
        "transition_atom",
        "zone",
    }:
        assert column not in context.features


def test_context_rejects_feature_timestamp_not_before_pitch() -> None:
    with pytest.raises(ValueError, match="feature timestamp"):
        transition_context_from_row(
            {
                "pitch_type": "FF",
                "relative_zone": "middle_middle",
                "balls": 0,
                "strikes": 0,
                "outs": 0,
                "runners": 0,
                "pitch_timestamp": "2024-04-01T18:00:00",
                "pitcher_profile_as_of_timestamp": "2024-04-01T18:00:00",
            }
        )
