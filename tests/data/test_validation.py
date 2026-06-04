from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from baseball_zerobase.cli import app
from baseball_zerobase.data.validation import LeakageError, audit_snapshots


@pytest.fixture
def valid_snapshot_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [12345, 12345, 12345],
            "game_date": ["2024-04-01", "2024-04-01", "2024-04-01"],
            "game_type": ["R", "R", "R"],
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
            "action": ["FF", "SL", None],
            "relative_zone": ["in_zone", "chase", "waste"],
            "outcome": ["ball", "strike", "strikeout"],
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


def test_audit_reports_action_and_terminal_distributions(
    valid_snapshot_frame: pl.DataFrame,
) -> None:
    report = audit_snapshots(valid_snapshot_frame)
    assert report.row_count > 0
    assert sum(report.relative_zone_counts.values()) == report.action_row_count
    assert report.locked_row_count == 0
    assert report.outcome_counts == {"ball": 1, "strike": 1, "strikeout": 1}
    assert report.terminal_reason_counts == {"three_outs": 1}
    assert report.half_inning_ended_counts == {"False": 2, "True": 1}
    assert report.timestamp_joined_counts == {"False": 1, "True": 2}


def test_validate_dataset_command_reads_snapshots_parquet(
    tmp_path: Path,
    valid_snapshot_frame: pl.DataFrame,
) -> None:
    snapshots_path = tmp_path / "snapshots.parquet"
    valid_snapshot_frame.write_parquet(snapshots_path)

    result = CliRunner().invoke(
        app,
        ["validate-dataset", "--snapshots-parquet", str(snapshots_path)],
    )

    assert result.exit_code == 0
    assert '"row_count": 3' in result.stdout
    assert '"locked_row_count": 0' in result.stdout
