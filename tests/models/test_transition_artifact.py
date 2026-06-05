from pathlib import Path

from baseball_zerobase.models.transition import SharedTransitionModelV0
from baseball_zerobase.models.transition_artifact import (
    read_transition_artifact,
    write_transition_artifact,
)
from baseball_zerobase.models.transition_context import transition_context_from_row

from tests.models.transition_fixtures import transition_training_frame


def test_transition_artifact_round_trip_preserves_predictions(tmp_path: Path) -> None:
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
        }
    )
    path = tmp_path / "transition_model.json"

    write_transition_artifact(model, path)
    loaded = read_transition_artifact(path)

    assert loaded.predict_distribution(context) == model.predict_distribution(context)
