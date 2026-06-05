import pytest

from baseball_zerobase.models.calibration import (
    brier_score,
    expected_calibration_error,
    log_loss,
)


def test_expected_calibration_error_groups_probabilities() -> None:
    value = expected_calibration_error(
        probabilities=[0.1, 0.8],
        outcomes=[False, True],
        bins=2,
    )

    assert value >= 0


def test_log_loss_and_brier_score_are_finite() -> None:
    assert log_loss([0.9, 0.2], [True, False]) == pytest.approx(0.164252, rel=1e-5)
    assert brier_score([0.9, 0.2], [True, False]) == pytest.approx(0.025)
