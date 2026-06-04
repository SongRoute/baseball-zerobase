from baseball_zerobase.data.contracts import OutcomeLabel
from baseball_zerobase.data.outcomes import map_outcome


def test_terminal_event_overrides_pitch_description() -> None:
    assert map_outcome(description="hit_into_play", event="home_run") is OutcomeLabel.HOME_RUN


def test_nonterminal_pitch_description_is_preserved() -> None:
    assert map_outcome(description="foul", event=None) is OutcomeLabel.FOUL


def test_hit_by_pitch_event_uses_full_statcast_value() -> None:
    outcome = map_outcome(description=None, event="hit_by_pitch")
    assert outcome is OutcomeLabel.HBP
    assert outcome.value == "hit_by_pitch"


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
