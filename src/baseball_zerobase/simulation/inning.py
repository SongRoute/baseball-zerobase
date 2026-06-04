from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from baseball_zerobase.data.contracts import RelativeZone, TransitionAtom
from baseball_zerobase.simulation.state import GameState

ActionKey = tuple[str, str]


class _ChoiceRng(Protocol):
    def choice(self, a: int, *, p: list[float]) -> object: ...


class _BehaviorModel(Protocol):
    def sample(
        self,
        rng: _ChoiceRng,
        *,
        balls: int,
        strikes: int,
        stand: str,
        p_throws: str,
    ) -> ActionKey: ...


class _TransitionModel(Protocol):
    def sample(
        self,
        rng: _ChoiceRng,
        *,
        pitch_type: str,
        relative_zone: str | RelativeZone,
        balls: int,
        strikes: int,
        outs: int,
        runners: int,
        stand: str,
        p_throws: str,
    ) -> TransitionAtom: ...


@dataclass(frozen=True, slots=True)
class InningSimulationResult:
    runs: tuple[int, ...]
    pitch_counts: tuple[int, ...]
    zero_run_probability: float
    two_plus_run_probability: float
    truncated_trials: int


@dataclass(frozen=True, slots=True)
class _TrialResult:
    runs: int
    pitch_count: int
    truncated: bool


@dataclass(slots=True)
class InningSimulator:
    behavior_model: _BehaviorModel
    transition_model: _TransitionModel
    max_pitches: int = 100

    def __post_init__(self) -> None:
        if self.max_pitches < 1:
            raise ValueError("max_pitches must be positive")

    def simulate_many(
        self,
        initial_state: GameState,
        *,
        trials: int,
        seed: int,
        fixed_first_action: ActionKey | None = None,
    ) -> InningSimulationResult:
        if trials < 1:
            raise ValueError("trials must be positive")

        rng = np.random.default_rng(seed)
        trial_results = tuple(
            self.simulate_one(
                initial_state,
                rng=rng,
                fixed_first_action=fixed_first_action,
            )
            for _ in range(trials)
        )
        runs = tuple(result.runs for result in trial_results)
        pitch_counts = tuple(result.pitch_count for result in trial_results)
        return InningSimulationResult(
            runs=runs,
            pitch_counts=pitch_counts,
            zero_run_probability=_mean(runs_array == 0)
            if (runs_array := np.asarray(runs, dtype=np.int_)).size
            else 0.0,
            two_plus_run_probability=_mean(runs_array >= 2) if runs_array.size else 0.0,
            truncated_trials=sum(result.truncated for result in trial_results),
        )

    def simulate_one(
        self,
        initial_state: GameState,
        *,
        rng: _ChoiceRng,
        fixed_first_action: ActionKey | None = None,
    ) -> _TrialResult:
        state = initial_state
        runs = 0
        pitch_count = 0
        terminated = state.outs >= 3

        while not terminated and pitch_count < self.max_pitches:
            action = (
                fixed_first_action
                if pitch_count == 0 and fixed_first_action is not None
                else self.behavior_model.sample(
                    rng,
                    balls=state.balls,
                    strikes=state.strikes,
                    stand=state.stand,
                    p_throws=state.p_throws,
                )
            )
            pitch_type, relative_zone = action
            atom = self.transition_model.sample(
                rng,
                pitch_type=pitch_type,
                relative_zone=relative_zone,
                balls=state.balls,
                strikes=state.strikes,
                outs=state.outs,
                runners=state.runners,
                stand=state.stand,
                p_throws=state.p_throws,
            )
            _validate_transition(state, atom)

            runs += atom.runs_scored
            state = _apply_transition(state, atom)
            pitch_count += 1
            terminated = atom.half_inning_ended or state.outs >= 3

        return _TrialResult(
            runs=runs,
            pitch_count=pitch_count,
            truncated=not terminated,
        )


def _apply_transition(state: GameState, atom: TransitionAtom) -> GameState:
    next_state = replace(
        state,
        balls=atom.balls_after,
        strikes=atom.strikes_after,
        outs=atom.outs_after,
        runners=_runners_to_mask(atom.runners_after),
        score_diff=state.score_diff + atom.runs_scored,
    )
    if atom.plate_appearance_ended:
        return next_state.advance_batting_order()
    return next_state


def _validate_transition(state: GameState, atom: TransitionAtom) -> None:
    if atom.outs_after < state.outs:
        raise ValueError("transition cannot decrease outs")
    if not 0 <= atom.outs_after <= 3:
        raise ValueError("transition outs_after must be between 0 and 3")
    if not 0 <= atom.balls_after <= 3:
        raise ValueError("transition balls_after must be between 0 and 3")
    if not 0 <= atom.strikes_after <= 2:
        raise ValueError("transition strikes_after must be between 0 and 2")

    _runners_to_mask(atom.runners_after)

    if atom.plate_appearance_ended:
        if atom.balls_after != 0 or atom.strikes_after != 0:
            raise ValueError("plate appearance transitions must reset the count")
        return

    if atom.balls_after < state.balls:
        raise ValueError("non-terminal pitch transition cannot decrease balls")
    if atom.strikes_after < state.strikes:
        raise ValueError("non-terminal pitch transition cannot decrease strikes")


def _runners_to_mask(runners: tuple[bool, bool, bool]) -> int:
    if len(runners) != 3:
        raise ValueError("runners_after must contain first, second, and third base")
    mask = 0
    for index, occupied in enumerate(runners):
        if not isinstance(occupied, bool):
            raise ValueError("runners_after entries must be bool values")
        if occupied:
            mask |= 1 << index
    return mask


def _mean(values: NDArray[np.bool_]) -> float:
    return float(np.mean(values))
