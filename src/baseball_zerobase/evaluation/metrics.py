from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import log
from typing import Any, TypeVar

from scipy.stats import wasserstein_distance

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class InningDistributionMetrics:
    predicted_mean_runs: float
    actual_mean_runs: float
    mean_run_error: float
    wasserstein_distance: float
    zero_run_probability_error: float
    multi_run_probability_error: float


@dataclass(frozen=True, slots=True)
class ActionRankingMetrics:
    top1_accuracy: float
    top3_accuracy: float
    negative_log_likelihood: float
    evaluated_count: int


def inning_distribution_metrics(
    *,
    predicted: Sequence[int | float],
    actual: Sequence[int | float],
) -> InningDistributionMetrics:
    predicted_values = [float(value) for value in predicted]
    actual_values = [float(value) for value in actual]
    predicted_mean = _mean(predicted_values)
    actual_mean = _mean(actual_values)
    distance = (
        float(wasserstein_distance(predicted_values, actual_values))
        if predicted_values and actual_values
        else 0.0
    )
    return InningDistributionMetrics(
        predicted_mean_runs=predicted_mean,
        actual_mean_runs=actual_mean,
        mean_run_error=abs(predicted_mean - actual_mean),
        wasserstein_distance=distance,
        zero_run_probability_error=abs(
            _probability(predicted_values, lambda value: value == 0)
            - _probability(actual_values, lambda value: value == 0)
        ),
        multi_run_probability_error=abs(
            _probability(predicted_values, lambda value: value >= 2)
            - _probability(actual_values, lambda value: value >= 2)
        ),
    )


def action_ranking_metrics(
    *,
    predicted: Sequence[Mapping[Any, float]],
    actual: Sequence[object],
    epsilon: float = 1e-12,
) -> ActionRankingMetrics:
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    top1_hits = 0
    top3_hits = 0
    nll = 0.0
    evaluated = 0
    for probabilities, actual_value in zip(predicted, actual, strict=True):
        normalized = _normalize_distribution(probabilities)
        if not normalized:
            continue
        actual_label = _label(actual_value)
        ranked = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
        top_labels = [label for label, _ in ranked]
        top1_hits += int(top_labels[:1] == [actual_label])
        top3_hits += int(actual_label in top_labels[:3])
        nll -= log(max(normalized.get(actual_label, 0.0), epsilon))
        evaluated += 1

    return ActionRankingMetrics(
        top1_accuracy=top1_hits / evaluated if evaluated else 0.0,
        top3_accuracy=top3_hits / evaluated if evaluated else 0.0,
        negative_log_likelihood=nll / evaluated if evaluated else 0.0,
        evaluated_count=evaluated,
    )


def transition_atom_negative_log_likelihood(
    *,
    predicted: Sequence[Mapping[Any, float]],
    actual: Sequence[object],
    epsilon: float = 1e-12,
) -> float:
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    losses: list[float] = []
    for probabilities, actual_value in zip(predicted, actual, strict=True):
        normalized = _normalize_distribution(probabilities)
        if normalized:
            losses.append(-log(max(normalized.get(_label(actual_value), 0.0), epsilon)))
    return _mean(losses)


def outcome_brier_score(
    *,
    predicted: Sequence[Mapping[Any, float]],
    actual: Sequence[object],
    labels: Iterable[object] | None = None,
) -> float:
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")

    explicit_labels = None if labels is None else {_label(label) for label in labels}
    scores: list[float] = []
    for probabilities, actual_value in zip(predicted, actual, strict=True):
        normalized = _normalize_distribution(probabilities)
        actual_label = _label(actual_value)
        observed_labels = set(normalized)
        label_set = observed_labels | {actual_label}
        if explicit_labels is not None:
            label_set |= explicit_labels
        scores.append(
            sum(
                (normalized.get(label, 0.0) - (1.0 if label == actual_label else 0.0)) ** 2
                for label in label_set
            )
        )
    return _mean(scores)


def _normalize_distribution(probabilities: Mapping[Any, float]) -> dict[str, float]:
    return {
        _label(label): float(probability)
        for label, probability in probabilities.items()
        if float(probability) >= 0
    }


def _label(value: object) -> str:
    if isinstance(value, tuple | list):
        return ":".join(_label(part) for part in value)
    model_dump = getattr(value, "model_dump_json", None)
    if callable(model_dump):
        return str(model_dump())
    return str(value)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _probability(values: Sequence[float], predicate: object) -> float:
    checker = predicate
    if not callable(checker):
        raise TypeError("predicate must be callable")
    return sum(1 for value in values if checker(value)) / len(values) if values else 0.0
