from datetime import datetime

import polars as pl

from baseball_zerobase.data.snapshots import build_snapshots


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
