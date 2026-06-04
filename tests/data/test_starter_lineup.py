from __future__ import annotations

from baseball_zerobase.data.starter_lineup import attach_starter_and_lineup_context


def test_keeps_only_official_starter_pitches(statcast_frame, normalized_game_frame) -> None:
    result = attach_starter_and_lineup_context(statcast_frame, normalized_game_frame)
    assert result["is_official_starter_pitch"].to_list() == [True, True, False]


def test_marks_snapshots_unstable_at_first_substitution(statcast_frame, normalized_game_frame) -> None:
    result = attach_starter_and_lineup_context(statcast_frame, normalized_game_frame)
    assert result["lineup_stable"].to_list() == [True, True, False]
