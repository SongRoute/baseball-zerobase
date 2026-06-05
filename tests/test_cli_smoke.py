from typer.testing import CliRunner
from datetime import date, datetime, timedelta

import polars as pl

from baseball_zerobase.cli import app


def test_cli_help_lists_pipeline_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.stdout


def test_cli_version_command_prints_project_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_profile_cli_commands_write_augmented_dev_parquets(tmp_path) -> None:
    input_path = tmp_path / "data/processed/snapshots/role=dev_regular/snapshots.parquet"
    input_path.parent.mkdir(parents=True)
    _profile_cli_frame().write_parquet(input_path)

    pitcher_output = tmp_path / "data/processed/profiles/role=dev_regular/pitcher.parquet"
    batter_output = tmp_path / "data/processed/profiles/role=dev_regular/batter.parquet"
    daily_output = tmp_path / "data/processed/profiles/role=dev_regular/daily.parquet"
    config = tmp_path / "configs/base.yaml"

    pitcher_result = CliRunner().invoke(
        app,
        [
            "build-pitcher-profiles",
            "--input",
            str(input_path),
            "--output-parquet",
            str(pitcher_output),
            "--min-pitch-type-pitches",
            "1",
            "--config",
            str(config),
        ],
    )
    assert pitcher_result.exit_code == 0, pitcher_result.stdout
    assert "pitcher profiles" in pitcher_result.stdout
    assert "pitcher_profile_prior_pitch_count" in pl.read_parquet(pitcher_output).columns

    batter_result = CliRunner().invoke(
        app,
        [
            "build-batter-profiles",
            "--input",
            str(input_path),
            "--output-parquet",
            str(batter_output),
            "--min-prior-pitches",
            "1",
            "--config",
            str(config),
        ],
    )
    assert batter_result.exit_code == 0, batter_result.stdout
    batter_columns = pl.read_parquet(batter_output).columns
    assert "batter_weakness_archetype" in batter_columns
    assert "batter_threat_score" in batter_columns

    daily_result = CliRunner().invoke(
        app,
        [
            "build-daily-state",
            "--input",
            str(input_path),
            "--output-parquet",
            str(daily_output),
            "--config",
            str(config),
        ],
    )
    assert daily_result.exit_code == 0, daily_result.stdout
    assert "pitcher_day_prior_pitch_count" in pl.read_parquet(daily_output).columns


def test_transition_model_cli_fit_and_evaluate(tmp_path) -> None:
    dataset = tmp_path / "data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet"
    dataset.parent.mkdir(parents=True)
    _transition_cli_frame().write_parquet(dataset)
    model_path = tmp_path / "artifacts/models/transition/v0.json"
    report_path = tmp_path / "reports/generated/transition/v0.json"
    config = tmp_path / "configs/base.yaml"

    fit_result = CliRunner().invoke(
        app,
        [
            "fit-transition-model",
            "--dataset",
            str(dataset),
            "--output",
            str(model_path),
            "--config",
            str(config),
        ],
    )
    assert fit_result.exit_code == 0, fit_result.stdout
    assert model_path.exists()

    eval_result = CliRunner().invoke(
        app,
        [
            "evaluate-transition-model",
            "--dataset",
            str(dataset),
            "--model",
            str(model_path),
            "--report",
            str(report_path),
            "--config",
            str(config),
        ],
    )
    assert eval_result.exit_code == 0, eval_result.stdout
    assert "korean_summary" in report_path.read_text(encoding="utf-8")


def _profile_cli_frame() -> pl.DataFrame:
    start = datetime(2024, 4, 1, 18, 0)
    return pl.DataFrame(
        {
            "game_pk": [1, 2],
            "game_date": [date(2024, 4, 1), date(2024, 4, 8)],
            "game_type": ["R", "R"],
            "pitch_timestamp": [start, start + timedelta(days=7)],
            "as_of_timestamp": [
                start - timedelta(seconds=1),
                start + timedelta(days=7, seconds=-1),
            ],
            "at_bat_number": [1, 1],
            "pitch_number": [1, 1],
            "pitcher_id": [501, 501],
            "batter_id": [201, 201],
            "pitch_type": ["FF", "FF"],
            "relative_zone": ["middle_middle", "middle_middle"],
            "outcome": ["home_run", "strikeout"],
            "plate_appearance_ended": [True, True],
            "runs_scored": [1, 0],
            "balls": [0, 0],
            "strikes": [0, 0],
        }
    )


def _transition_cli_frame() -> pl.DataFrame:
    start = datetime(2024, 4, 1, 18, 0)
    return pl.DataFrame(
        {
            "game_pk": [1, 2],
            "game_date": [date(2024, 4, 1), date(2024, 4, 8)],
            "game_type": ["R", "R"],
            "pitch_timestamp": [start, start + timedelta(days=7)],
            "as_of_timestamp": [
                start - timedelta(seconds=1),
                start + timedelta(days=7, seconds=-1),
            ],
            "pitch_type": ["FF", "FF"],
            "relative_zone": ["middle_middle", "middle_middle"],
            "balls": [0, 0],
            "strikes": [0, 0],
            "outs": [0, 0],
            "runners": [0, 0],
            "stand": ["R", "R"],
            "p_throws": ["R", "R"],
            "outcome": ["ball", "called_strike"],
            "balls_after": [1, 0],
            "strikes_after": [0, 1],
            "outs_after": [0, 0],
            "runners_after": [0, 0],
            "runs_scored": [0, 0],
            "plate_appearance_ended": [False, False],
            "half_inning_ended": [False, False],
            "terminal_reason": [None, None],
        }
    )
