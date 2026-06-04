from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from baseball_zerobase.baseline.behavior import EmpiricalBehaviorModel


def test_behavior_probabilities_sum_to_one() -> None:
    model = EmpiricalBehaviorModel(min_support=2).fit(
        _behavior_frame(),
        training_manifest_hash="sha256:dev",
    )

    probs = model.predict_proba(balls=0, strikes=0, stand="R", p_throws="R")

    assert np.isclose(sum(probs.values()), 1.0)


def test_behavior_model_uses_exact_smoothed_context_probabilities() -> None:
    model = EmpiricalBehaviorModel(min_support=2, alpha=0.5).fit(
        _behavior_frame(),
        training_manifest_hash="sha256:dev",
    )

    probs = model.predict_proba(balls=0, strikes=0, stand="R", p_throws="R")

    assert model.last_backoff_level == "balls_strikes_stand_p_throws"
    assert np.isclose(probs[("FF", "middle_middle")], 3.5 / 5.0)
    assert np.isclose(probs[("SL", "low_away")], 1.5 / 5.0)


def test_behavior_model_backs_off_when_context_is_sparse() -> None:
    model = EmpiricalBehaviorModel(min_support=100).fit(
        _behavior_frame(),
        training_manifest_hash="sha256:dev",
    )

    probs = model.predict_proba(balls=3, strikes=2, stand="L", p_throws="L")

    assert probs
    assert model.last_backoff_level == "global"


def test_behavior_model_uses_count_backoff_when_handed_context_is_sparse() -> None:
    model = EmpiricalBehaviorModel(min_support=5, alpha=0.5).fit(
        _behavior_frame(),
        training_manifest_hash="sha256:dev",
    )

    probs = model.predict_proba(balls=0, strikes=0, stand="R", p_throws="R")

    assert model.last_backoff_level == "balls_strikes"
    assert np.isclose(probs[("FF", "middle_middle")], 5.5 / 7.0)
    assert np.isclose(probs[("SL", "low_away")], 1.5 / 7.0)


def test_behavior_model_rejects_raw_statcast_zone_fallback() -> None:
    raw_zone_frame = pl.DataFrame(
        {
            "pitch_type": ["FF"],
            "zone": [5],
            "balls": [0],
            "strikes": [0],
            "stand": ["R"],
            "p_throws": ["R"],
            "count": [1],
        }
    )

    with pytest.raises(ValueError, match="raw Statcast zone is not allowed"):
        EmpiricalBehaviorModel().fit(raw_zone_frame, training_manifest_hash="sha256:dev")


def test_behavior_model_serializes_counts_settings_and_manifest_hash() -> None:
    model = EmpiricalBehaviorModel(min_support=2, alpha=0.5).fit(
        _behavior_frame(),
        training_manifest_hash="sha256:dev",
    )

    payload = json.loads(model.to_json())
    restored = EmpiricalBehaviorModel.from_json(model.to_json())

    assert payload["training_manifest_hash"] == "sha256:dev"
    assert payload["settings"]["min_support"] == 2
    assert payload["actions"] == [["FF", "middle_middle"], ["SL", "low_away"]]
    assert payload["counts"]["global"]
    assert restored.training_manifest_hash == "sha256:dev"
    assert restored.predict_proba(
        balls=0, strikes=0, stand="R", p_throws="R"
    ) == model.predict_proba(
        balls=0,
        strikes=0,
        stand="R",
        p_throws="R",
    )


def _behavior_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "pitch_type": ["FF", "SL", "FF", "FF"],
            "relative_zone": ["middle_middle", "low_away", "middle_middle", "middle_middle"],
            "balls": [0, 0, 1, 0],
            "strikes": [0, 0, 0, 0],
            "stand": ["R", "R", "L", "L"],
            "p_throws": ["R", "R", "R", "L"],
            "count": [3, 1, 2, 2],
        }
    )
