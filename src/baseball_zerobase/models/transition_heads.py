from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from baseball_zerobase.data.contracts import OutcomeLabel, TransitionAtom
from baseball_zerobase.models.transition_context import TransitionContext


_SWING_OUTCOMES = {
    OutcomeLabel.SWINGING_STRIKE.value,
    OutcomeLabel.FOUL.value,
    OutcomeLabel.IN_PLAY_OUT.value,
    OutcomeLabel.SINGLE.value,
    OutcomeLabel.DOUBLE.value,
    OutcomeLabel.TRIPLE.value,
    OutcomeLabel.HOME_RUN.value,
    OutcomeLabel.REACH_OTHER.value,
}
_CONTACT_OUTCOMES = {
    OutcomeLabel.FOUL.value,
    OutcomeLabel.IN_PLAY_OUT.value,
    OutcomeLabel.SINGLE.value,
    OutcomeLabel.DOUBLE.value,
    OutcomeLabel.TRIPLE.value,
    OutcomeLabel.HOME_RUN.value,
    OutcomeLabel.REACH_OTHER.value,
}
_BATTED_BALL_OUTCOMES = {
    OutcomeLabel.IN_PLAY_OUT.value,
    OutcomeLabel.SINGLE.value,
    OutcomeLabel.DOUBLE.value,
    OutcomeLabel.TRIPLE.value,
    OutcomeLabel.HOME_RUN.value,
    OutcomeLabel.REACH_OTHER.value,
}


@dataclass(frozen=True, slots=True)
class TransitionHeadLabels:
    swing: str
    contact: str | None
    called_result: str | None
    contact_result: str | None
    batted_ball: str | None
    plate_appearance: str


def head_labels_from_row(row: Mapping[str, object]) -> TransitionHeadLabels:
    outcome = str(row["outcome"])
    swing = "swing" if outcome in _SWING_OUTCOMES else "take"
    contact = None
    called_result = None
    contact_result = None
    batted_ball = None

    if swing == "swing":
        contact = "contact" if outcome in _CONTACT_OUTCOMES else "no_contact"
        if contact == "contact":
            contact_result = "in_play" if outcome in _BATTED_BALL_OUTCOMES else "foul"
            if contact_result == "in_play":
                batted_ball = outcome
    elif outcome in {
        OutcomeLabel.BALL.value,
        OutcomeLabel.CALLED_STRIKE.value,
        OutcomeLabel.HBP.value,
    }:
        called_result = outcome

    plate_appearance = "continues"
    if outcome == OutcomeLabel.WALK.value:
        plate_appearance = "walk"
    elif outcome == OutcomeLabel.STRIKEOUT.value:
        plate_appearance = "strikeout"
    elif outcome == OutcomeLabel.HBP.value:
        plate_appearance = "hbp"
    elif outcome == OutcomeLabel.REACH_OTHER.value:
        plate_appearance = "reach_other"
    elif outcome in _BATTED_BALL_OUTCOMES:
        plate_appearance = "terminal_in_play"

    return TransitionHeadLabels(
        swing=swing,
        contact=contact,
        called_result=called_result,
        contact_result=contact_result,
        batted_ball=batted_ball,
        plate_appearance=plate_appearance,
    )


def is_legal_transition(context: TransitionContext, atom: TransitionAtom) -> bool:
    if atom.outs_after < context.outs or atom.outs_after > 3:
        return False
    if not 0 <= atom.balls_after <= 3 or not 0 <= atom.strikes_after <= 2:
        return False
    if len(atom.runners_after) != 3 or any(
        not isinstance(value, bool) for value in atom.runners_after
    ):
        return False
    if atom.runs_scored < 0:
        return False
    if atom.plate_appearance_ended:
        if atom.balls_after != 0 or atom.strikes_after != 0:
            return False
    else:
        if atom.balls_after < context.balls or atom.strikes_after < context.strikes:
            return False
    if atom.half_inning_ended and atom.terminal_reason not in {"three_outs", "game_end"}:
        return False
    if not atom.half_inning_ended and atom.terminal_reason is not None:
        return False
    return True
