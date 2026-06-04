import math

from baseball_zerobase.evaluation.metrics import (
    action_ranking_metrics,
    inning_distribution_metrics,
    outcome_brier_score,
)


def test_inning_metrics_compare_zero_and_multi_run_rates() -> None:
    metrics = inning_distribution_metrics(predicted=[0, 0, 1, 2], actual=[0, 1, 1, 3])

    assert metrics.zero_run_probability_error == 0.25
    assert metrics.multi_run_probability_error == 0.0


def test_action_ranking_metrics_include_top_k_and_nll() -> None:
    metrics = action_ranking_metrics(
        predicted=[
            {"FF:middle_middle": 0.7, "SL:low_away": 0.2, "CH:chase_low": 0.1},
            {"FF:middle_middle": 0.5, "SL:low_away": 0.3, "CH:chase_low": 0.2},
        ],
        actual=["FF:middle_middle", "CH:chase_low"],
    )

    assert metrics.top1_accuracy == 0.5
    assert metrics.top3_accuracy == 1.0
    assert math.isclose(metrics.negative_log_likelihood, -(math.log(0.7) + math.log(0.2)) / 2)


def test_outcome_brier_score_for_multiclass_probabilities() -> None:
    score = outcome_brier_score(
        predicted=[
            {"strikeout": 0.75, "walk": 0.25},
            {"strikeout": 0.25, "walk": 0.75},
        ],
        actual=["strikeout", "strikeout"],
    )

    assert math.isclose(score, ((0.25**2 + 0.25**2) + (0.75**2 + 0.75**2)) / 2)
