from datetime import date
import json
from pathlib import Path

import polars as pl
import pytest

from baseball_zerobase.data.contracts import OutcomeLabel
from baseball_zerobase.data.eligibility import add_starter_eligibility
from baseball_zerobase.data.manifest import sha256_file
from baseball_zerobase.data.snapshots import build_development_dataset, write_development_dataset


@pytest.fixture
def starter_snapshot_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [1, 1, 1, 2],
            "game_date": [
                date(2024, 4, 1),
                date(2024, 4, 1),
                date(2024, 4, 1),
                date(2024, 4, 7),
            ],
            "pitcher_id": [501, 501, 501, 501],
            "is_official_starter_pitch": [True, True, True, True],
        }
    )


def test_eligibility_counts_only_prior_games(starter_snapshot_frame) -> None:
    result = add_starter_eligibility(starter_snapshot_frame, min_prior_pitches=3)
    first_game = result.filter(result["game_pk"] == 1)
    second_game = result.filter(result["game_pk"] == 2)
    assert first_game["starter_eligible"].unique().to_list() == [False]
    assert second_game["prior_two_season_starter_pitches"].min() == 3
    assert second_game["starter_eligible"].unique().to_list() == [True]


def test_eligibility_uses_two_year_window_and_current_season_prior_games() -> None:
    frame = _starter_pitch_rows(
        [
            (1, date(2022, 3, 31), 10),
            (2, date(2023, 4, 1), 2),
            (3, date(2024, 3, 30), 3),
            (4, date(2024, 4, 1), 1),
        ]
    )

    result = add_starter_eligibility(frame, min_prior_pitches=5)
    target_game = result.filter(result["game_pk"] == 4)

    assert target_game["prior_two_season_starter_pitches"].unique().to_list() == [5]
    assert target_game["current_season_prior_pitches"].unique().to_list() == [3]
    assert target_game["starter_eligible"].unique().to_list() == [True]


def test_development_dataset_filters_rows_and_records_counts() -> None:
    snapshots = _development_filter_frame()

    dataset = build_development_dataset(snapshots)

    assert dataset.frame["game_pk"].to_list() == [1]
    assert dataset.filter_counts == {
        "input_rows": 10,
        "dev_regular_rows": 9,
        "official_starter_pitch_rows": 8,
        "stable_lineup_rows": 7,
        "starter_eligible_rows": 6,
        "non_null_action_rows": 5,
        "timestamp_joined_rows": 4,
        "supported_strategic_event_rows": 1,
    }


def test_development_dataset_manifest_records_input_checksums_and_filter_counts(
    tmp_path: Path,
) -> None:
    dataset = build_development_dataset(_development_filter_frame())
    input_path = tmp_path / "snapshots.parquet"
    input_path.write_bytes(b"snapshot input")
    output_path = tmp_path / "dev_dataset.parquet"

    manifest = write_development_dataset(
        dataset,
        output_path,
        source="test-dev-dataset",
        request={"role": "dev_regular"},
        input_paths={"snapshots": input_path},
    )

    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    assert payload["request"]["input_checksums"] == {"snapshots": sha256_file(input_path)}
    assert payload["request"]["filter_counts"] == dataset.filter_counts


def _starter_pitch_rows(games: list[tuple[int, date, int]]) -> pl.DataFrame:
    rows = [
        {
            "game_pk": game_pk,
            "game_date": game_date,
            "pitcher_id": 501,
            "is_official_starter_pitch": True,
        }
        for game_pk, game_date, pitch_count in games
        for _ in range(pitch_count)
    ]
    return pl.DataFrame(rows)


def _development_filter_frame() -> pl.DataFrame:
    labels = [
        "kept",
        "non_dev",
        "not_official",
        "unstable",
        "ineligible",
        "missing_action",
        "missing_timestamp",
        "unsupported_outcome",
        "automatic_call",
        "intentional_walk",
    ]
    frame = pl.DataFrame(
        {
            "game_pk": list(range(1, len(labels) + 1)),
            "game_date": [date(2024, 4, 1)] * len(labels),
            "game_type": ["R"] * len(labels),
            "is_official_starter_pitch": [True] * len(labels),
            "lineup_stable": [True] * len(labels),
            "starter_eligible": [True] * len(labels),
            "pitch_type": ["FF"] * len(labels),
            "relative_zone": ["middle_middle"] * len(labels),
            "action": ["FF:middle_middle"] * len(labels),
            "timestamp_joined": [True] * len(labels),
            "outcome": [OutcomeLabel.BALL.value] * len(labels),
            "description": ["ball"] * len(labels),
            "events": [None] * len(labels),
        }
    )
    return frame.with_columns(
        pl.when(pl.col("game_pk") == 2).then(pl.lit("S")).otherwise(pl.col("game_type")).alias("game_type"),
        pl.when(pl.col("game_pk") == 3)
        .then(pl.lit(False))
        .otherwise(pl.col("is_official_starter_pitch"))
        .alias("is_official_starter_pitch"),
        pl.when(pl.col("game_pk") == 4)
        .then(pl.lit(False))
        .otherwise(pl.col("lineup_stable"))
        .alias("lineup_stable"),
        pl.when(pl.col("game_pk") == 5)
        .then(pl.lit(False))
        .otherwise(pl.col("starter_eligible"))
        .alias("starter_eligible"),
        pl.when(pl.col("game_pk") == 6).then(None).otherwise(pl.col("action")).alias("action"),
        pl.when(pl.col("game_pk") == 7)
        .then(pl.lit(False))
        .otherwise(pl.col("timestamp_joined"))
        .alias("timestamp_joined"),
        pl.when(pl.col("game_pk") == 8)
        .then(pl.lit(OutcomeLabel.OTHER.value))
        .otherwise(pl.col("outcome"))
        .alias("outcome"),
        pl.when(pl.col("game_pk") == 9)
        .then(pl.lit("automatic_ball"))
        .otherwise(pl.col("description"))
        .alias("description"),
        pl.when(pl.col("game_pk") == 10)
        .then(pl.lit("intent_walk"))
        .otherwise(pl.col("events"))
        .alias("events"),
    )
