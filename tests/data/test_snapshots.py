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
