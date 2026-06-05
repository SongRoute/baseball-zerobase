from __future__ import annotations

from math import log
from collections.abc import Sequence


def log_loss(
    probabilities: Sequence[float], outcomes: Sequence[bool], *, epsilon: float = 1e-12
) -> float:
    _validate_lengths(probabilities, outcomes)
    if not probabilities:
        return 0.0
    total = 0.0
    for probability, outcome in zip(probabilities, outcomes, strict=True):
        clipped = min(1.0 - epsilon, max(epsilon, float(probability)))
        total += -log(clipped if outcome else 1.0 - clipped)
    return total / len(probabilities)


def brier_score(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float:
    _validate_lengths(probabilities, outcomes)
    if not probabilities:
        return 0.0
    return sum(
        (float(probability) - float(outcome)) ** 2
        for probability, outcome in zip(probabilities, outcomes, strict=True)
    ) / len(probabilities)


def expected_calibration_error(
    probabilities: Sequence[float],
    outcomes: Sequence[bool],
    *,
    bins: int = 10,
) -> float:
    _validate_lengths(probabilities, outcomes)
    if bins < 1:
        raise ValueError("bins must be positive")
    if not probabilities:
        return 0.0
    total = len(probabilities)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indexes = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= float(probability) < upper
            or (bin_index == bins - 1 and float(probability) == 1.0)
        ]
        if not indexes:
            continue
        confidence = sum(float(probabilities[index]) for index in indexes) / len(indexes)
        accuracy = sum(float(outcomes[index]) for index in indexes) / len(indexes)
        error += len(indexes) / total * abs(accuracy - confidence)
    return error


def rare_outcome_recall(
    predicted_probabilities: Sequence[float], outcomes: Sequence[bool], *, threshold: float = 0.01
) -> float:
    _validate_lengths(predicted_probabilities, outcomes)
    positives = [index for index, outcome in enumerate(outcomes) if outcome]
    if not positives:
        return 0.0
    detected = sum(1 for index in positives if float(predicted_probabilities[index]) >= threshold)
    return detected / len(positives)


def _validate_lengths(probabilities: Sequence[float], outcomes: Sequence[bool]) -> None:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
