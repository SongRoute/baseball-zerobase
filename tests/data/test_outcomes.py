from baseball_zerobase.data.contracts import OutcomeLabel
from baseball_zerobase.data.outcomes import map_outcome


def test_terminal_event_overrides_pitch_description() -> None:
    assert map_outcome(description="hit_into_play", event="home_run") is OutcomeLabel.HOME_RUN


def test_nonterminal_pitch_description_is_preserved() -> None:
    assert map_outcome(description="foul", event=None) is OutcomeLabel.FOUL
