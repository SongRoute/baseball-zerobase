from datetime import date, datetime, timedelta

import polars as pl
from typer.testing import CliRunner

from baseball_zerobase.cli import app
from baseball_zerobase.data.eligibility import add_starter_eligibility
from baseball_zerobase.data.snapshots import build_snapshots
from baseball_zerobase.data.starter_lineup import attach_starter_and_lineup_context
from baseball_zerobase.evaluation.rolling import evaluate_fold


def test_fixture_pipeline_builds_and_evaluates_baseline() -> None:
    prepared = attach_starter_and_lineup_context(
        _fixture_statcast_frame(),
        _fixture_normalized_game_frame(),
    )
    snapshots = build_snapshots(prepared)
    eligible_snapshots = add_starter_eligibility(snapshots, min_prior_pitches=1)

    report = evaluate_fold(
        eligible_snapshots,
        train_years=(2022,),
        validation_year=2023,
        trials=20,
    )

    assert report.transition_negative_log_likelihood >= 0
    assert report.simulation_truncation_rate == 0


def test_pipeline_smoke_cli_prints_baseline_metric_summary() -> None:
    result = CliRunner().invoke(app, ["pipeline-smoke"])

    assert result.exit_code == 0
    assert "Baseline metric summary" in result.stdout
    assert "transition_negative_log_likelihood" in result.stdout
    assert "simulation_truncation_rate: 0.000" in result.stdout


def _fixture_statcast_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for year in (2022, 2023):
        game_pk = year * 1000 + 1
        game_start = datetime(year, 4, 1, 18, 0)
        for index in range(3):
            pitch_timestamp = game_start + timedelta(minutes=5, seconds=index * 20)
            rows.append(
                {
                    "game_pk": game_pk,
                    "at_bat_number": index + 1,
                    "pitch_number": 1,
                    "inning": 1,
                    "inning_topbot": "Top",
                    "pitcher": 501,
                    "batter": 201 + index,
                    "stand": "R",
                    "p_throws": "R",
                    "balls": 0,
                    "strikes": 0,
                    "outs_when_up": index,
                    "on_1b": None,
                    "on_2b": None,
                    "on_3b": None,
                    "bat_score": 0,
                    "fld_score": 0,
                    "post_bat_score": 0,
                    "pitch_type": "FF",
                    "relative_zone": "middle_middle",
                    "description": "hit_into_play",
                    "events": "field_out",
                    "pitch_timestamp": pitch_timestamp,
                    "completed_event_timestamp": pitch_timestamp + timedelta(seconds=5),
                }
            )
    return pl.DataFrame(rows)


def _fixture_normalized_game_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for year in (2022, 2023):
        rows.append(
            {
                "game_pk": year * 1000 + 1,
                "game_date": date(year, 4, 1),
                "game_type": "R",
                "home_team_id": 119,
                "away_team_id": 137,
                "home_starter_id": 501,
                "away_starter_id": 601,
                "home_starter_throws": "R",
                "away_starter_throws": "L",
                "home_initial_lineup": [101, 102, 103, 104, 105, 106, 107, 108, 109],
                "away_initial_lineup": [201, 202, 203, 204, 205, 206, 207, 208, 209],
                "home_initial_lineup_stands": ["R", "L", "R", "L", "R", "L", "R", "L", "R"],
                "away_initial_lineup_stands": ["R", "R", "R", "L", "R", "L", "R", "L", "R"],
                "game_start_timestamp": datetime(year, 4, 1, 18, 0),
                "first_substitution_at_bat": None,
            }
        )
    return pl.DataFrame(rows)
