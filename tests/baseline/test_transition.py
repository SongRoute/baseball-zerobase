from __future__ import annotations

import json
import math
from typing import TypedDict

import numpy as np
import polars as pl
import pytest

from baseball_zerobase.baseline.transition import EmpiricalTransitionModel
from baseball_zerobase.data.contracts import OutcomeLabel, TransitionAtom


class _TransitionContext(TypedDict):
    pitch_type: str
    relative_zone: str
    balls: int
    strikes: int
    outs: int
    runners: int
    stand: str
    p_throws: str


def test_transition_distribution_sums_to_one() -> None:
    model = EmpiricalTransitionModel(min_support=2).fit(
        _transition_frame(),
        training_manifest_hash="sha256:transitions",
    )

    distribution = model.predict_distribution(
        pitch_type="FF",
        relative_zone="middle_middle",
        balls=0,
        strikes=0,
        outs=0,
        runners=0,
        stand="R",
        p_throws="R",
    )

    assert np.isclose(sum(distribution.values()), 1.0)
    assert model.last_backoff_level == "action_balls_strikes_outs_runners_stand_p_throws"


@pytest.mark.parametrize(
    ("context", "expected_level", "expected_support"),
    [
        (
            {
                "pitch_type": "FF",
                "relative_zone": "middle_middle",
                "balls": 0,
                "strikes": 0,
                "outs": 0,
                "runners": 0,
                "stand": "R",
                "p_throws": "R",
            },
            "action_balls_strikes_outs_runners_stand_p_throws",
            3,
        ),
        (
            {
                "pitch_type": "FF",
                "relative_zone": "middle_middle",
                "balls": 1,
                "strikes": 1,
                "outs": 0,
                "runners": 0,
                "stand": "R",
                "p_throws": "R",
            },
            "action_balls_strikes_stand_p_throws",
            2,
        ),
        (
            {
                "pitch_type": "SL",
                "relative_zone": "low_away",
                "balls": 2,
                "strikes": 1,
                "outs": 0,
                "runners": 0,
                "stand": "R",
                "p_throws": "R",
            },
            "action_balls_strikes",
            2,
        ),
        (
            {
                "pitch_type": "CH",
                "relative_zone": "chase_low",
                "balls": 0,
                "strikes": 0,
                "outs": 0,
                "runners": 0,
                "stand": "R",
                "p_throws": "R",
            },
            "action",
            2,
        ),
        (
            {
                "pitch_type": "CU",
                "relative_zone": "middle_middle",
                "balls": 0,
                "strikes": 0,
                "outs": 0,
                "runners": 0,
                "stand": "R",
                "p_throws": "R",
            },
            "global",
            11,
        ),
    ],
)
def test_transition_model_selects_first_supported_backoff_level(
    context: _TransitionContext,
    expected_level: str,
    expected_support: int,
) -> None:
    model = EmpiricalTransitionModel(min_support=2).fit(
        _transition_frame(),
        training_manifest_hash="sha256:transitions",
    )

    assert model.support(**context) == expected_support
    assert model.last_backoff_level == expected_level


def test_transition_probabilities_are_unsmoothed_observed_frequencies() -> None:
    model = EmpiricalTransitionModel(min_support=2).fit(
        _transition_frame(),
        training_manifest_hash="sha256:transitions",
    )

    distribution = model.predict_distribution(
        pitch_type="FF",
        relative_zone="middle_middle",
        balls=0,
        strikes=0,
        outs=0,
        runners=0,
        stand="R",
        p_throws="R",
    )

    assert distribution[_called_strike_atom()] == 2 / 3
    assert distribution[_ball_atom()] == 1 / 3


def test_sampled_transition_preserves_atom_invariants() -> None:
    model = EmpiricalTransitionModel(min_support=2).fit(
        _transition_frame(),
        training_manifest_hash="sha256:transitions",
    )

    atom = model.sample(
        np.random.default_rng(42),
        pitch_type="FF",
        relative_zone="middle_middle",
        balls=0,
        strikes=0,
        outs=0,
        runners=0,
        stand="R",
        p_throws="R",
    )

    assert isinstance(atom, TransitionAtom)
    assert 0 <= atom.balls_after <= 3
    assert 0 <= atom.strikes_after <= 2
    assert 0 <= atom.outs_after <= 3
    assert len(atom.runners_after) == 3
    assert all(isinstance(occupied, bool) for occupied in atom.runners_after)


def test_log_probability_uses_epsilon_floor_for_unseen_atom() -> None:
    model = EmpiricalTransitionModel(min_support=2, epsilon=1e-6).fit(
        _transition_frame(),
        training_manifest_hash="sha256:transitions",
    )

    log_probability = model.log_probability(
        _single_atom(),
        {
            "pitch_type": "FF",
            "relative_zone": "middle_middle",
            "balls": 0,
            "strikes": 0,
            "outs": 0,
            "runners": 0,
            "stand": "R",
            "p_throws": "R",
        },
    )

    assert math.isclose(log_probability, math.log(1e-6))


def test_transition_model_serializes_counts_actions_settings_and_manifest_hash() -> None:
    model = EmpiricalTransitionModel(min_support=2, epsilon=1e-6).fit(
        _transition_frame(),
        training_manifest_hash="sha256:transitions",
    )

    payload = json.loads(model.to_json())
    restored = EmpiricalTransitionModel.from_json(model.to_json())
    context = {
        "pitch_type": "FF",
        "relative_zone": "middle_middle",
        "balls": 0,
        "strikes": 0,
        "outs": 0,
        "runners": 0,
        "stand": "R",
        "p_throws": "R",
    }

    assert payload["training_manifest_hash"] == "sha256:transitions"
    assert payload["settings"]["min_support"] == 2
    assert payload["settings"]["epsilon"] == 1e-6
    assert ["FF", "middle_middle"] in payload["actions"]
    assert payload["counts"]["global"]
    assert payload["counts"]["action_balls_strikes_outs_runners_stand_p_throws"]
    assert restored.training_manifest_hash == "sha256:transitions"
    assert restored.predict_distribution(**context) == model.predict_distribution(**context)


def test_transition_model_rejects_invalid_rows_instead_of_fitting_impossible_atoms() -> None:
    model = EmpiricalTransitionModel(min_support=1).fit(
        pl.concat(
            [
                _transition_frame().head(1),
                _transition_frame().head(1).with_columns(pl.lit(4).alias("balls_after")),
            ],
            how="vertical_relaxed",
        ),
        training_manifest_hash="sha256:transitions",
    )

    assert model.support(
        pitch_type="FF",
        relative_zone="middle_middle",
        balls=0,
        strikes=0,
        outs=0,
        runners=0,
        stand="R",
        p_throws="R",
    ) == 1


def test_transition_model_rejects_raw_statcast_zone_fallback() -> None:
    raw_zone_frame = _transition_frame().head(1).with_columns(pl.lit(5).alias("zone"))

    with pytest.raises(ValueError, match="raw Statcast zone is not allowed"):
        EmpiricalTransitionModel().fit(
            raw_zone_frame,
            training_manifest_hash="sha256:transitions",
        )


def _transition_frame() -> pl.DataFrame:
    rows = [
        _row("FF", "middle_middle", 0, 0, 0, 0, "R", "R", _called_strike_atom()),
        _row("FF", "middle_middle", 0, 0, 0, 0, "R", "R", _called_strike_atom()),
        _row("FF", "middle_middle", 0, 0, 0, 0, "R", "R", _ball_atom()),
        _row("FF", "middle_middle", 1, 1, 0, 0, "R", "R", _called_strike_atom()),
        _row(
            "FF",
            "middle_middle",
            1,
            1,
            1,
            1,
            "R",
            "R",
            _ball_atom(outs_after=1, runners_after=(True, False, False)),
        ),
        _row("SL", "low_away", 2, 1, 0, 0, "R", "R", _called_strike_atom()),
        _row("SL", "low_away", 2, 1, 1, 0, "L", "L", _ball_atom(outs_after=1)),
        _row("CH", "chase_low", 3, 2, 0, 0, "R", "R", _called_strike_atom()),
        _row("CH", "chase_low", 1, 0, 0, 0, "L", "L", _ball_atom()),
        _row("SI", "middle_away", 0, 1, 2, 3, "L", "R", _single_atom()),
        _row("SI", "middle_away", 0, 1, 2, 3, "L", "R", _single_atom()),
    ]
    return pl.DataFrame(rows)


def _row(
    pitch_type: str,
    relative_zone: str,
    balls: int,
    strikes: int,
    outs: int,
    runners: int,
    stand: str,
    p_throws: str,
    atom: TransitionAtom,
) -> dict[str, object]:
    atom_payload = atom.model_dump(mode="json")
    return {
        "pitch_type": pitch_type,
        "relative_zone": relative_zone,
        "action": f"{pitch_type}:{relative_zone}",
        "balls": balls,
        "strikes": strikes,
        "outs": outs,
        "runners": runners,
        "stand": stand,
        "p_throws": p_throws,
        **atom_payload,
    }


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


def _ball_atom(
    *,
    outs_after: int = 0,
    runners_after: tuple[bool, bool, bool] = (False, False, False),
) -> TransitionAtom:
    return TransitionAtom(
        outcome=OutcomeLabel.BALL,
        balls_after=1,
        strikes_after=0,
        outs_after=outs_after,
        runners_after=runners_after,
        runs_scored=0,
        plate_appearance_ended=False,
        half_inning_ended=False,
        terminal_reason=None,
    )


def _single_atom() -> TransitionAtom:
    return TransitionAtom(
        outcome=OutcomeLabel.SINGLE,
        balls_after=0,
        strikes_after=0,
        outs_after=2,
        runners_after=(True, False, False),
        runs_scored=1,
        plate_appearance_ended=True,
        half_inning_ended=False,
        terminal_reason=None,
    )
