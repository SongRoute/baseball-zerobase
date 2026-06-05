import pytest

from baseball_zerobase.models.transition import SharedTransitionModelV0
from baseball_zerobase.models.transition_context import transition_context_from_row
from baseball_zerobase.models.transition_heads import is_legal_transition

from tests.models.transition_fixtures import transition_training_frame


def test_transition_model_predicts_normalized_legal_distribution() -> None:
    model = SharedTransitionModelV0(min_support=1, prior_weight=1.0)
    model.fit(transition_training_frame(), training_manifest_hash="synthetic:m4")
    context = transition_context_from_row(
        {
            "pitch_type": "FF",
            "relative_zone": "middle_middle",
            "balls": 0,
            "strikes": 0,
            "outs": 0,
            "runners": 0,
            "stand": "R",
            "p_throws": "R",
            "pitcher_pitch_type_owned": True,
            "batter_weakness_archetype": "chase_vulnerable",
            "batter_threat_score": 0.75,
        }
    )

    distribution = model.predict_distribution(context)

    assert sum(distribution.values()) == pytest.approx(1.0)
    assert all(probability >= 0 for probability in distribution.values())
    assert all(is_legal_transition(context, atom) for atom in distribution)
    assert model.support(context) >= 1


def test_rare_outcome_gets_smoothed_nonzero_probability() -> None:
    model = SharedTransitionModelV0(min_support=1, prior_weight=1.0)
    model.fit(transition_training_frame(), training_manifest_hash="synthetic:m4")
    context = transition_context_from_row(
        {
            "pitch_type": "FF",
            "relative_zone": "middle_middle",
            "balls": 0,
            "strikes": 0,
            "outs": 0,
            "runners": 0,
            "pitcher_pitch_type_owned": False,
            "batter_weakness_archetype": "neutral_unknown",
            "batter_threat_score": 0.1,
        }
    )

    distribution = model.predict_distribution(context)

    assert any(
        atom.outcome.value == "home_run" and probability > 0
        for atom, probability in distribution.items()
    )
