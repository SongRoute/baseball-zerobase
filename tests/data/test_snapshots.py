from datetime import date, datetime
import json
from pathlib import Path

import polars as pl
import pytest

from baseball_zerobase.data.manifest import ManifestConflictError, sha256_file
from baseball_zerobase.data.snapshots import (
    build_development_dataset,
    build_snapshots,
    write_snapshot_dataset,
)


def test_snapshot_uses_only_pre_pitch_state(prepared_pitch_frame) -> None:
    snapshots = build_snapshots(prepared_pitch_frame)
    first = snapshots.row(0, named=True)
    assert first["balls"] == 0
    assert first["strikes"] == 0
    assert first["runs_scored"] == 0
    assert first["as_of_timestamp"] < first["pitch_timestamp"]


def test_transition_atom_uses_next_observed_state(prepared_pitch_frame) -> None:
    snapshots = build_snapshots(prepared_pitch_frame)
    first = snapshots.row(0, named=True)
    assert first["balls_after"] == 0
    assert first["strikes_after"] == 1
    assert first["plate_appearance_ended"] is False


def test_final_third_out_without_next_row_is_three_out_terminal() -> None:
    snapshots = build_snapshots(
        pl.DataFrame(
            {
                "game_pk": [12345],
                "at_bat_number": [9],
                "pitch_number": [1],
                "inning": [9],
                "inning_topbot": ["Top"],
                "outs_when_up": [2],
                "events": ["strikeout"],
                "description": ["swinging_strike"],
                "pitch_timestamp": [datetime(2024, 4, 1, 23, 5, 10)],
                "completed_event_timestamp": [datetime(2024, 4, 1, 23, 5, 12)],
                "game_start_timestamp": [datetime(2024, 4, 1, 23, 0)],
            }
        )
    )

    final = snapshots.row(0, named=True)
    assert final["outs_after"] == 3
    assert final["half_inning_ended"] is True
    assert final["terminal_reason"] == "three_outs"


def test_missing_normalized_pitch_event_is_preserved_as_unjoined() -> None:
    pitch_frame = pl.DataFrame(
        {
            "game_pk": [12345],
            "at_bat_number": [1],
            "pitch_number": [1],
            "pitch_timestamp": [datetime(2024, 4, 1, 23, 5, 10)],
            "completed_event_timestamp": [datetime(2024, 4, 1, 23, 5, 12)],
            "game_start_timestamp": [datetime(2024, 4, 1, 23, 0)],
        }
    )
    events_frame = pl.DataFrame(
        {
            "game_pk": [12345],
            "at_bat_number": [1],
            "pitch_number": [2],
            "pitch_timestamp": [datetime(2024, 4, 1, 23, 5, 30)],
            "completed_event_timestamp": [datetime(2024, 4, 1, 23, 5, 32)],
        }
    )

    snapshots = build_snapshots(pitch_frame, events_frame)

    assert snapshots.row(0, named=True)["timestamp_joined"] is False


def test_snapshot_dataset_conflict_leaves_existing_data_and_manifest(
    tmp_path: Path,
) -> None:
    first_snapshots = pl.DataFrame({"game_pk": [1], "pitch_number": [1]})
    second_snapshots = pl.DataFrame({"game_pk": [99], "pitch_number": [1]})
    output_path = tmp_path / "snapshots.parquet"

    first_manifest = write_snapshot_dataset(
        first_snapshots,
        output_path,
        source="test-snapshots",
        request={"game_pk": 1},
    )
    original_bytes = output_path.read_bytes()
    original_manifest_payload = json.loads(first_manifest.path.read_text(encoding="utf-8"))

    with pytest.raises(ManifestConflictError):
        write_snapshot_dataset(
            second_snapshots,
            output_path,
            source="test-snapshots",
            request={"game_pk": 99},
        )

    assert output_path.read_bytes() == original_bytes
    assert json.loads(first_manifest.path.read_text(encoding="utf-8")) == original_manifest_payload
    assert original_manifest_payload["sha256"] == sha256_file(output_path)


def test_snapshot_dataset_manifest_conflict_without_existing_data_rolls_back(
    tmp_path: Path,
) -> None:
    snapshots = pl.DataFrame({"game_pk": [1], "pitch_number": [1]})
    output_path = tmp_path / "snapshots.parquet"

    first_manifest = write_snapshot_dataset(
        snapshots,
        output_path,
        source="test-snapshots",
        request={"game_pk": 1},
    )
    output_path.unlink()
    original_manifest_payload = json.loads(first_manifest.path.read_text(encoding="utf-8"))

    with pytest.raises(ManifestConflictError):
        write_snapshot_dataset(
            snapshots,
            output_path,
            source="test-snapshots",
            request={"game_pk": 99},
        )

    assert not output_path.exists()
    assert json.loads(first_manifest.path.read_text(encoding="utf-8")) == original_manifest_payload


def test_joined_pitch_is_unavailable_when_as_of_timestamp_uses_unjoined_previous_pitch() -> None:
    pitch_frame = pl.DataFrame(
        {
            "game_pk": [12345, 12345],
            "game_date": [date(2024, 4, 1), date(2024, 4, 1)],
            "game_type": ["R", "R"],
            "at_bat_number": [1, 1],
            "pitch_number": [1, 2],
            "inning": [1, 1],
            "inning_topbot": ["Top", "Top"],
            "pitch_timestamp": [
                datetime(2024, 4, 1, 23, 5, 10),
                datetime(2024, 4, 1, 23, 5, 30),
            ],
            "completed_event_timestamp": [
                datetime(2024, 4, 1, 23, 5, 12),
                datetime(2024, 4, 1, 23, 5, 32),
            ],
            "game_start_timestamp": [
                datetime(2024, 4, 1, 23, 0),
                datetime(2024, 4, 1, 23, 0),
            ],
            "is_official_starter_pitch": [True, True],
            "lineup_stable": [True, True],
            "pitch_type": ["FF", "FF"],
            "relative_zone": ["middle_middle", "middle_middle"],
            "description": ["ball", "called_strike"],
            "events": [None, None],
        }
    )
    events_frame = pl.DataFrame(
        {
            "game_pk": [12345],
            "at_bat_number": [1],
            "pitch_number": [2],
            "pitch_timestamp": [datetime(2024, 4, 1, 23, 5, 30)],
            "completed_event_timestamp": [datetime(2024, 4, 1, 23, 5, 32)],
        }
    )

    snapshots = build_snapshots(pitch_frame, events_frame)
    second = snapshots.row(1, named=True)

    assert second["as_of_timestamp"] == datetime(2024, 4, 1, 23, 5, 12)
    assert second["timestamp_joined"] is False
    eligible = snapshots.with_columns(pl.lit(True).alias("starter_eligible"))
    dataset = build_development_dataset(eligible)
    assert dataset.frame.is_empty()
    assert dataset.filter_counts["non_null_action_rows"] == 2
    assert dataset.filter_counts["timestamp_joined_rows"] == 0


def test_unjoined_timestamp_provenance_propagates_beyond_one_pitch() -> None:
    pitch_frame = pl.DataFrame(
        {
            "game_pk": [12345, 12345, 12345],
            "game_date": [date(2024, 4, 1), date(2024, 4, 1), date(2024, 4, 1)],
            "game_type": ["R", "R", "R"],
            "at_bat_number": [1, 1, 1],
            "pitch_number": [1, 2, 3],
            "inning": [1, 1, 1],
            "inning_topbot": ["Top", "Top", "Top"],
            "pitch_timestamp": [
                datetime(2024, 4, 1, 23, 5, 10),
                datetime(2024, 4, 1, 23, 5, 30),
                datetime(2024, 4, 1, 23, 5, 50),
            ],
            "completed_event_timestamp": [
                datetime(2024, 4, 1, 23, 5, 12),
                datetime(2024, 4, 1, 23, 5, 32),
                datetime(2024, 4, 1, 23, 5, 52),
            ],
            "game_start_timestamp": [
                datetime(2024, 4, 1, 23, 0),
                datetime(2024, 4, 1, 23, 0),
                datetime(2024, 4, 1, 23, 0),
            ],
            "is_official_starter_pitch": [True, True, True],
            "lineup_stable": [True, True, True],
            "starter_eligible": [True, True, True],
            "pitch_type": ["FF", "FF", "FF"],
            "relative_zone": ["middle_middle", "middle_middle", "middle_middle"],
            "description": ["ball", "called_strike", "foul"],
            "events": [None, None, None],
        }
    )
    events_frame = pl.DataFrame(
        {
            "game_pk": [12345, 12345],
            "at_bat_number": [1, 1],
            "pitch_number": [2, 3],
            "pitch_timestamp": [
                datetime(2024, 4, 1, 23, 5, 30),
                datetime(2024, 4, 1, 23, 5, 50),
            ],
            "completed_event_timestamp": [
                datetime(2024, 4, 1, 23, 5, 32),
                datetime(2024, 4, 1, 23, 5, 52),
            ],
        }
    )

    snapshots = build_snapshots(pitch_frame, events_frame)

    assert snapshots["timestamp_joined"].to_list() == [False, False, False]
    eligible = snapshots.with_columns(pl.lit(True).alias("starter_eligible"))
    assert build_development_dataset(eligible).frame.is_empty()
