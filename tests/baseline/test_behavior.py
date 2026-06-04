from __future__ import annotations

import numpy as np

from baseball_zerobase.baseline.behavior import EmpiricalBehaviorModel


def test_behavior_probabilities_sum_to_one(baseline_snapshot_frame) -> None:
    model = EmpiricalBehaviorModel(min_support=2).fit(baseline_snapshot_frame)

    probs = model.predict_proba(balls=0, strikes=0, stand="R", p_throws="R")

    assert np.isclose(sum(probs.values()), 1.0)


def test_behavior_model_backs_off_when_context_is_sparse(baseline_snapshot_frame) -> None:
    model = EmpiricalBehaviorModel(min_support=100).fit(baseline_snapshot_frame)

    probs = model.predict_proba(balls=3, strikes=2, stand="L", p_throws="L")

    assert probs
    assert model.last_backoff_level == "global"
