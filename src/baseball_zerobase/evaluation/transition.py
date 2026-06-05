from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from baseball_zerobase.data.contracts import OutcomeLabel
from baseball_zerobase.models.calibration import (
    expected_calibration_error,
    log_loss,
    rare_outcome_recall,
)
from baseball_zerobase.models.transition import SharedTransitionModelV0, _atom_from_row
from baseball_zerobase.models.transition_context import transition_context_from_row


_RARE_OUTCOMES = {
    OutcomeLabel.HOME_RUN.value,
    OutcomeLabel.DOUBLE.value,
    OutcomeLabel.TRIPLE.value,
    OutcomeLabel.WALK.value,
    OutcomeLabel.STRIKEOUT.value,
    OutcomeLabel.HBP.value,
}


@dataclass(frozen=True, slots=True)
class TransitionEvaluationReport:
    row_count: int
    log_loss: float
    home_run_recall: float
    rare_outcome_recall: float
    expected_calibration_error: float
    support_min: int
    support_max: int
    korean_summary: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_transition_model(
    model: SharedTransitionModelV0,
    frame: pl.DataFrame,
) -> TransitionEvaluationReport:
    actual_probabilities: list[float] = []
    home_run_probabilities: list[float] = []
    home_run_outcomes: list[bool] = []
    rare_probabilities: list[float] = []
    rare_outcomes: list[bool] = []
    supports: list[int] = []

    for row in frame.iter_rows(named=True):
        context = transition_context_from_row(row)
        atom = _atom_from_row(row)
        distribution = model.predict_distribution(context)
        actual_probabilities.append(distribution.get(atom, 0.0))
        home_run_probability = sum(
            probability
            for candidate, probability in distribution.items()
            if candidate.outcome == OutcomeLabel.HOME_RUN
        )
        rare_probability = sum(
            probability
            for candidate, probability in distribution.items()
            if candidate.outcome.value in _RARE_OUTCOMES
        )
        home_run_probabilities.append(home_run_probability)
        home_run_outcomes.append(atom.outcome == OutcomeLabel.HOME_RUN)
        rare_probabilities.append(rare_probability)
        rare_outcomes.append(atom.outcome.value in _RARE_OUTCOMES)
        supports.append(model.support(context))

    report_log_loss = log_loss(actual_probabilities, [True] * len(actual_probabilities))
    report_ece = expected_calibration_error(
        actual_probabilities, [True] * len(actual_probabilities)
    )
    rare_recall = rare_outcome_recall(rare_probabilities, rare_outcomes)
    hr_recall = rare_outcome_recall(home_run_probabilities, home_run_outcomes)
    row_count = len(actual_probabilities)
    return TransitionEvaluationReport(
        row_count=row_count,
        log_loss=report_log_loss,
        home_run_recall=hr_recall,
        rare_outcome_recall=rare_recall,
        expected_calibration_error=report_ece,
        support_min=min(supports) if supports else 0,
        support_max=max(supports) if supports else 0,
        korean_summary=f"전이모델 평가 행 수 {row_count}개, 로그손실 {report_log_loss:.4f}.",
    )
