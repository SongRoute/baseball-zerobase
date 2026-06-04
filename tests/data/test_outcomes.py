import pytest
from pydantic import ValidationError

from baseball_zerobase.data.contracts import OutcomeLabel, TransitionAtom
from baseball_zerobase.data.outcomes import map_outcome


def test_terminal_event_overrides_pitch_description() -> None:
    assert map_outcome(description="hit_into_play", event="home_run") is OutcomeLabel.HOME_RUN


def test_nonterminal_pitch_description_is_preserved() -> None:
    assert map_outcome(description="foul", event=None) is OutcomeLabel.FOUL


def test_hit_by_pitch_event_uses_full_statcast_value() -> None:
    outcome = map_outcome(description=None, event="hit_by_pitch")
    assert outcome is OutcomeLabel.HBP
    assert outcome.value == "hit_by_pitch"


@pytest.mark.parametrize(
    ("description", "event", "expected"),
    [
        ("intent_ball", "intent_walk", OutcomeLabel.WALK),
        ("swinging_strike", "strikeout_double_play", OutcomeLabel.STRIKEOUT),
        ("hit_into_play", "sac_fly_double_play", OutcomeLabel.IN_PLAY_OUT),
    ],
)
def test_terminal_event_aliases_override_pitch_description(
    description: str,
    event: str,
    expected: OutcomeLabel,
) -> None:
    assert map_outcome(description=description, event=event) is expected


def test_transition_atom_accepts_possible_state() -> None:
    atom = TransitionAtom(
        outcome=OutcomeLabel.BALL,
        balls_after=3,
        strikes_after=2,
        outs_after=3,
        runners_after=(True, False, True),
        runs_scored=0,
        plate_appearance_ended=False,
        half_inning_ended=False,
        terminal_reason=None,
    )
    assert atom.runners_after == (True, False, True)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("balls_after", -1),
        ("balls_after", 4),
        ("strikes_after", -1),
        ("strikes_after", 3),
        ("outs_after", -1),
        ("outs_after", 4),
        ("runs_scored", -1),
    ],
)
def test_transition_atom_rejects_impossible_state_counts(field_name: str, value: int) -> None:
    payload = {
        "outcome": OutcomeLabel.BALL,
        "balls_after": 0,
        "strikes_after": 0,
        "outs_after": 0,
        "runners_after": (False, False, False),
        "runs_scored": 0,
        "plate_appearance_ended": False,
        "half_inning_ended": False,
        "terminal_reason": None,
    }
    payload[field_name] = value

    with pytest.raises(ValidationError):
        TransitionAtom(**payload)


def test_outcome_label_values_match_contract() -> None:
    assert {label.value for label in OutcomeLabel} == {
        "ball",
        "called_strike",
        "swinging_strike",
        "foul",
        "in_play_out",
        "single",
        "double",
        "triple",
        "home_run",
        "walk",
        "strikeout",
        "hit_by_pitch",
        "reach_other",
        "other",
    }
