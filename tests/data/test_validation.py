from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from baseball_zerobase.cli import app
from baseball_zerobase.data.contracts import OutcomeLabel
from baseball_zerobase.data.validation import LeakageError, audit_snapshots


@pytest.fixture
def valid_snapshot_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [12345, 12345, 12345],
            "at_bat_number": [1, 1, 2],
            "pitch_number": [1, 2, 1],
            "game_date": ["2024-04-01", "2024-04-01", "2024-04-01"],
            "game_type": ["R", "R", "R"],
            "pitcher_id": [501, 501, 501],
            "expected_starter_id": [501, 501, 501],
            "is_official_starter_pitch": [True, True, True],
            "lineup_stable": [True, True, True],
            "starter_eligible": [False, True, True],
            "pitch_timestamp": [
                datetime(2024, 4, 1, 23, 5, 1),
                datetime(2024, 4, 1, 23, 5, 12),
                datetime(2024, 4, 1, 23, 5, 30),
            ],
            "as_of_timestamp": [
                datetime(2024, 4, 1, 23, 5),
                datetime(2024, 4, 1, 23, 5, 1),
                datetime(2024, 4, 1, 23, 5, 12),
            ],
            "timestamp_joined": [True, True, False],
            "pitch_type": ["FF", "SL", "CH"],
            "relative_zone": ["middle_middle", "chase_high", "low_away"],
            "action": ["FF:middle_middle", "SL:chase_high", "CH:low_away"],
            "outcome": [
                OutcomeLabel.BALL.value,
                OutcomeLabel.CALLED_STRIKE.value,
                OutcomeLabel.STRIKEOUT.value,
            ],
            "balls": [0, 1, 2],
            "strikes": [0, 1, 2],
            "outs": [0, 0, 2],
            "runners": [0, 1, 7],
            "balls_after": [1, 2, 0],
            "strikes_after": [0, 2, 0],
            "outs_after": [0, 0, 3],
            "runners_after": [1, 3, 0],
            "runs_scored": [0, 0, 1],
            "plate_appearance_ended": [False, True, True],
            "terminal_reason": [None, None, "three_outs"],
            "half_inning_ended": [False, False, True],
        }
    )


@pytest.fixture
def leaky_snapshot_frame(valid_snapshot_frame: pl.DataFrame) -> pl.DataFrame:
    return valid_snapshot_frame.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.col("pitch_timestamp"))
        .otherwise(pl.col("as_of_timestamp"))
        .alias("as_of_timestamp")
    )


def test_audit_rejects_future_as_of_timestamp(leaky_snapshot_frame: pl.DataFrame) -> None:
    with pytest.raises(LeakageError):
        audit_snapshots(leaky_snapshot_frame)


def test_audit_rejects_future_profile_feature_timestamp(
    valid_snapshot_frame: pl.DataFrame,
) -> None:
    leaky = valid_snapshot_frame.with_columns(
        pl.col("pitch_timestamp").alias("pitcher_profile_as_of_timestamp")
    )

    with pytest.raises(LeakageError, match="feature timestamp"):
        audit_snapshots(leaky)


def test_audit_rejects_duplicate_pitch_keys(valid_snapshot_frame: pl.DataFrame) -> None:
    duplicate = valid_snapshot_frame.with_columns(
        _replace_row("at_bat_number", 1, 1),
        _replace_row("pitch_number", 1, 1),
    )

    with pytest.raises(LeakageError, match="duplicate"):
        audit_snapshots(duplicate)


@pytest.mark.parametrize(
    ("label", "mutate", "message"),
    [
        (
            "locked row",
            lambda frame: frame.with_columns(pl.lit("2026-04-01").alias("game_date")),
            "locked",
        ),
        (
            "null as of timestamp",
            lambda frame: frame.with_columns(_replace_row("as_of_timestamp", 0, None)),
            "timestamp",
        ),
        (
            "null pitch timestamp",
            lambda frame: frame.with_columns(_replace_row("pitch_timestamp", 0, None)),
            "timestamp",
        ),
        (
            "invalid balls",
            lambda frame: frame.with_columns(_replace_row("balls", 0, 4)),
            "balls",
        ),
        (
            "invalid strikes",
            lambda frame: frame.with_columns(_replace_row("strikes", 0, 3)),
            "strikes",
        ),
        (
            "impossible pre-pitch three outs",
            lambda frame: frame.with_columns(_replace_row("outs", 0, 3)),
            "outs",
        ),
        (
            "invalid outs",
            lambda frame: frame.with_columns(_replace_row("outs", 0, 4)),
            "outs",
        ),
        (
            "invalid runners",
            lambda frame: frame.with_columns(_replace_row("runners", 0, 8)),
            "runners",
        ),
        (
            "invalid runners after",
            lambda frame: frame.with_columns(_replace_row("runners_after", 0, -1)),
            "runners_after",
        ),
        (
            "invalid relative zone",
            lambda frame: frame.with_columns(_replace_row("relative_zone", 0, "in_zone")),
            "relative_zone",
        ),
        (
            "starter flag false",
            lambda frame: frame.with_columns(_replace_row("is_official_starter_pitch", 0, False)),
            "official starter",
        ),
        (
            "starter mismatch",
            lambda frame: frame.with_columns(_replace_row("expected_starter_id", 0, 999)),
            "official starter",
        ),
        (
            "unstable lineup",
            lambda frame: frame.with_columns(_replace_row("lineup_stable", 0, False)),
            "lineup_stable",
        ),
        (
            "negative runs scored",
            lambda frame: frame.with_columns(_replace_row("runs_scored", 0, -1)),
            "runs_scored",
        ),
        (
            "decreasing outs",
            lambda frame: frame.with_columns(_replace_row("outs_after", 2, 1)),
            "outs_after",
        ),
        (
            "missing terminal reason",
            lambda frame: frame.with_columns(_replace_row("terminal_reason", 2, None)),
            "terminal_reason",
        ),
        (
            "invalid terminal reason",
            lambda frame: frame.with_columns(_replace_row("terminal_reason", 2, "rain_delay")),
            "terminal_reason",
        ),
        (
            "nonterminal terminal reason",
            lambda frame: frame.with_columns(_replace_row("terminal_reason", 0, "three_outs")),
            "terminal_reason",
        ),
        (
            "unknown outcome",
            lambda frame: frame.with_columns(_replace_row("outcome", 0, "mystery")),
            "outcome",
        ),
        (
            "unsupported outcome",
            lambda frame: frame.with_columns(_replace_row("outcome", 0, OutcomeLabel.OTHER.value)),
            "outcome",
        ),
        (
            "unknown action",
            lambda frame: frame.with_columns(_replace_row("action", 0, "mystery")),
            "action",
        ),
    ],
)
def test_audit_rejects_required_fail_closed_conditions(
    valid_snapshot_frame: pl.DataFrame,
    label: str,
    mutate: Callable[[pl.DataFrame], pl.DataFrame],
    message: str,
) -> None:
    with pytest.raises(LeakageError, match=message):
        audit_snapshots(mutate(valid_snapshot_frame))


def test_audit_reports_action_and_terminal_distributions(
    valid_snapshot_frame: pl.DataFrame,
) -> None:
    report = audit_snapshots(valid_snapshot_frame)
    assert report.row_count > 0
    assert report.game_count == 1
    assert sum(report.relative_zone_counts.values()) == report.action_row_count
    assert report.locked_row_count == 0
    assert report.outcome_counts == {"ball": 1, "called_strike": 1, "strikeout": 1}
    assert report.pitch_type_counts == {"CH": 1, "FF": 1, "SL": 1}
    assert report.starter_eligible_counts == {"False": 1, "True": 2}
    assert report.transition_outcome_counts == {"ball": 1, "called_strike": 1, "strikeout": 1}
    assert report.timestamp_join_rate == pytest.approx(2 / 3)
    assert report.unknown_action_rate == 0
    assert report.unknown_outcome_rate == 0
    assert report.included_counts["dev_regular"] == 3
    assert report.included_counts["non_null_action"] == 3
    assert report.excluded_counts["locked"] == 0
    assert report.terminal_reason_counts == {"three_outs": 1}
    assert report.half_inning_ended_counts == {"False": 2, "True": 1}
    assert report.timestamp_joined_counts == {"False": 1, "True": 2}


def test_validate_dataset_command_reads_input_and_writes_report(
    tmp_path: Path,
    valid_snapshot_frame: pl.DataFrame,
) -> None:
    snapshots_path = tmp_path / "data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet"
    snapshots_path.parent.mkdir(parents=True)
    valid_snapshot_frame.write_parquet(snapshots_path)
    report_path = tmp_path / "reports/generated/validation.json"

    result = CliRunner().invoke(
        app,
        [
            "validate-dataset",
            "--input",
            str(snapshots_path),
            "--report",
            str(report_path),
            "--config",
            str(tmp_path / "configs/base.yaml"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["row_count"] == 3
    assert payload["locked_row_count"] == 0
    assert str(report_path) in result.stdout


def test_validate_dataset_command_rejects_locked_report_path(
    tmp_path: Path,
    valid_snapshot_frame: pl.DataFrame,
) -> None:
    snapshots_path = tmp_path / "data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet"
    snapshots_path.parent.mkdir(parents=True)
    valid_snapshot_frame.write_parquet(snapshots_path)

    result = CliRunner().invoke(
        app,
        [
            "validate-dataset",
            "--input",
            str(snapshots_path),
            "--report",
            str(tmp_path / "data/locked/reports/validation.json"),
            "--config",
            str(tmp_path / "configs/base.yaml"),
        ],
    )

    assert result.exit_code != 0
    assert "locked path" in result.output


def _replace_row(column: str, index: int, value: object) -> pl.Expr:
    return (
        pl.when(pl.arange(0, pl.len()) == index)
        .then(pl.lit(value))
        .otherwise(pl.col(column))
        .alias(column)
    )
