from __future__ import annotations

from typing import Any

import pytest

from baseball_zerobase.data.contracts import OutcomeLabel, TransitionAtom
from baseball_zerobase.simulation.inning import InningSimulator
from baseball_zerobase.simulation.state import GameState


def test_same_seed_produces_same_run_distribution() -> None:
    simulator = InningSimulator(
        SequencedBehaviorModel([("FF", "middle_middle"), ("SL", "low_away")]),
        RandomRunTransitionModel(),
        max_pitches=20,
    )
    state = _initial_state()

    first = simulator.simulate_many(state, trials=100, seed=42)
    second = simulator.simulate_many(state, trials=100, seed=42)

    assert first.runs == second.runs
    assert first.pitch_counts == second.pitch_counts
    assert first.zero_run_probability == second.zero_run_probability
    assert first.two_plus_run_probability == second.two_plus_run_probability


def test_simulation_terminates_without_truncation_on_deterministic_baseline() -> None:
    simulator = InningSimulator(
        SequencedBehaviorModel([("FF", "middle_middle")]),
        SequenceTransitionModel([_out_atom(1), _out_atom(2), _out_atom(3, half_inning_ended=True)]),
        max_pitches=10,
    )

    result = simulator.simulate_many(_initial_state(), trials=20, seed=7)

    assert result.runs == (0,) * 20
    assert result.pitch_counts == (3,) * 20
    assert result.truncated_trials == 0
    assert result.zero_run_probability == 1.0
    assert result.two_plus_run_probability == 0.0


def test_fixed_first_action_is_used_for_first_pitch_only() -> None:
    behavior = SequencedBehaviorModel([("FF", "middle_middle")])
    transitions = RecordingTransitionModel(
        [_called_strike_atom(), _out_atom(1), _out_atom(2), _out_atom(3, half_inning_ended=True)]
    )
    simulator = InningSimulator(behavior, transitions, max_pitches=10)

    simulator.simulate_many(_initial_state(), trials=1, seed=9, fixed_first_action=("SL", "low_away"))

    assert transitions.actions == [
        ("SL", "low_away"),
        ("FF", "middle_middle"),
        ("FF", "middle_middle"),
        ("FF", "middle_middle"),
    ]


def test_invalid_transition_that_decreases_outs_is_rejected() -> None:
    simulator = InningSimulator(
        SequencedBehaviorModel([("FF", "middle_middle")]),
        SequenceTransitionModel([_out_atom(0)]),
        max_pitches=10,
    )

    with pytest.raises(ValueError, match="decrease outs"):
        simulator.simulate_many(_initial_state(outs=1), trials=1, seed=1)


class SequencedBehaviorModel:
    def __init__(self, actions: list[tuple[str, str]]) -> None:
        self._actions = actions
        self._index = 0

    def sample(
        self,
        rng: Any,
        *,
        balls: int,
        strikes: int,
        stand: str,
        p_throws: str,
    ) -> tuple[str, str]:
        del rng, balls, strikes, stand, p_throws
        action = self._actions[self._index % len(self._actions)]
        self._index += 1
        return action


class SequenceTransitionModel:
    def __init__(self, atoms: list[TransitionAtom]) -> None:
        self._atoms = atoms
        self._index = 0

    def sample(
        self,
        rng: object,
        *,
        pitch_type: str,
        relative_zone: str,
        balls: int,
        strikes: int,
        outs: int,
        runners: int,
        stand: str,
        p_throws: str,
    ) -> TransitionAtom:
        del rng, pitch_type, relative_zone, balls, strikes, outs, runners, stand, p_throws
        atom = self._atoms[self._index % len(self._atoms)]
        self._index += 1
        return atom


class RecordingTransitionModel(SequenceTransitionModel):
    def __init__(self, atoms: list[TransitionAtom]) -> None:
        super().__init__(atoms)
        self.actions: list[tuple[str, str]] = []

    def sample(
        self,
        rng: object,
        *,
        pitch_type: str,
        relative_zone: str,
        balls: int,
        strikes: int,
        outs: int,
        runners: int,
        stand: str,
        p_throws: str,
    ) -> TransitionAtom:
        self.actions.append((pitch_type, relative_zone))
        return super().sample(
            rng,
            pitch_type=pitch_type,
            relative_zone=relative_zone,
            balls=balls,
            strikes=strikes,
            outs=outs,
            runners=runners,
            stand=stand,
            p_throws=p_throws,
        )


class RandomRunTransitionModel:
    def sample(
        self,
        rng: Any,
        *,
        pitch_type: str,
        relative_zone: str,
        balls: int,
        strikes: int,
        outs: int,
        runners: int,
        stand: str,
        p_throws: str,
    ) -> TransitionAtom:
        del pitch_type, relative_zone, balls, strikes, runners, stand, p_throws
        runs = int(rng.choice(3, p=[0.5, 0.25, 0.25]))
        return _out_atom(outs + 1, runs_scored=runs, half_inning_ended=outs + 1 >= 3)


def _initial_state(*, outs: int = 0) -> GameState:
    return GameState(
        balls=0,
        strikes=0,
        outs=outs,
        runners=0,
        inning=1,
        score_diff=0,
        batting_order_index=0,
        lineup_ids=(10, 20, 30),
        lineup_stands=("R", "L", "R"),
        stand="R",
        p_throws="R",
    )


def _called_strike_atom() -> TransitionAtom:
    return TransitionAtom(
        outcome=OutcomeLabel.CALLED_STRIKE,
        balls_after=0,
        strikes_after=1,
        outs_after=0,
        runners_after=(False, False, False),
        runs_scored=0,
        plate_appearance_ended=False,
        half_inning_ended=False,
        terminal_reason=None,
    )


def _out_atom(
    outs_after: int,
    *,
    runs_scored: int = 0,
    half_inning_ended: bool = False,
) -> TransitionAtom:
    return TransitionAtom(
        outcome=OutcomeLabel.IN_PLAY_OUT,
        balls_after=0,
        strikes_after=0,
        outs_after=outs_after,
        runners_after=(False, False, False),
        runs_scored=runs_scored,
        plate_appearance_ended=True,
        half_inning_ended=half_inning_ended,
        terminal_reason="three_outs" if half_inning_ended else None,
    )
