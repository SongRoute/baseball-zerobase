from baseball_zerobase.evaluation.transition import evaluate_transition_model
from baseball_zerobase.models.transition import SharedTransitionModelV0

from tests.models.transition_fixtures import transition_training_frame


def test_evaluate_transition_model_reports_korean_summary() -> None:
    frame = transition_training_frame()
    model = SharedTransitionModelV0(min_support=1, prior_weight=1.0).fit(
        frame,
        training_manifest_hash="synthetic:m4",
    )

    report = evaluate_transition_model(model, frame)

    payload = report.to_dict()
    assert report.row_count > 0
    assert "korean_summary" in payload
    assert "log_loss" in payload
