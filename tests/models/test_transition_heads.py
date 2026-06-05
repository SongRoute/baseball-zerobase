from baseball_zerobase.data.contracts import OutcomeLabel, TransitionAtom
from baseball_zerobase.models.transition_context import transition_context_from_row
from baseball_zerobase.models.transition_heads import head_labels_from_row, is_legal_transition


def test_head_labels_map_swinging_strike_to_swing_no_contact() -> None:
    labels = head_labels_from_row({"outcome": OutcomeLabel.SWINGING_STRIKE.value})

    assert labels.swing == "swing"
    assert labels.contact == "no_contact"
    assert labels.plate_appearance == "continues"


def test_legal_transition_mask_rejects_decreasing_outs() -> None:
    context = transition_context_from_row(
        {
            "pitch_type": "FF",
            "relative_zone": "middle_middle",
            "balls": 1,
            "strikes": 1,
            "outs": 2,
            "runners": 0,
        }
    )
    atom = TransitionAtom(
        outcome=OutcomeLabel.BALL,
        balls_after=2,
        strikes_after=1,
        outs_after=1,
        runners_after=(False, False, False),
        runs_scored=0,
        plate_appearance_ended=False,
        half_inning_ended=False,
        terminal_reason=None,
    )

    assert not is_legal_transition(context, atom)


def test_legal_mask_rejects_terminal_pa_without_count_reset() -> None:
    context = transition_context_from_row(
        {
            "pitch_type": "FF",
            "relative_zone": "middle_middle",
            "balls": 3,
            "strikes": 2,
            "outs": 0,
            "runners": 0,
        }
    )
    atom = TransitionAtom(
        outcome=OutcomeLabel.WALK,
        balls_after=3,
        strikes_after=2,
        outs_after=0,
        runners_after=(True, False, False),
        runs_scored=0,
        plate_appearance_ended=True,
        half_inning_ended=False,
        terminal_reason=None,
    )

    assert not is_legal_transition(context, atom)
