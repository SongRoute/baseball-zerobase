from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

import baseball_zerobase.profiles.batter_archetype as batter_archetype
import baseball_zerobase.profiles.batter_threat as batter_threat
import baseball_zerobase.profiles.daily_state as daily_state
import baseball_zerobase.profiles.pitcher as pitcher


def test_pitcher_profile_does_not_rescan_full_history_for_each_target(monkeypatch) -> None:
    frame = _pitcher_profile_frame(row_count=160)
    calls = _count_datetime_calls(monkeypatch, pitcher)

    featured = pitcher.add_pitcher_profiles(
        frame,
        min_pitch_type_pitches=1,
        current_season_min_pitches=0,
    )

    assert featured.height == frame.height
    assert calls["count"] <= frame.height * 8


def test_batter_archetype_does_not_rescan_full_history_for_each_target(monkeypatch) -> None:
    frame = _batter_pitch_frame(row_count=160)
    calls = _count_datetime_calls(monkeypatch, batter_archetype)

    featured = batter_archetype.add_batter_archetypes(frame, min_prior_pitches=0)

    assert featured.height == frame.height
    assert calls["count"] <= frame.height * 8


def test_batter_threat_does_not_rescan_full_history_for_each_target(monkeypatch) -> None:
    frame = _batter_pitch_frame(row_count=160).with_columns(
        pl.lit(True).alias("plate_appearance_ended")
    )
    calls = _count_datetime_calls(monkeypatch, batter_threat)

    featured = batter_threat.add_batter_threat(frame)

    assert featured.height == frame.height
    assert calls["count"] <= frame.height * 8


def test_daily_state_does_not_rescan_full_game_for_each_target(monkeypatch) -> None:
    frame = _daily_state_frame(row_count=160)
    calls = _count_datetime_calls(monkeypatch, daily_state)

    featured = daily_state.add_daily_state(frame)

    assert featured.height == frame.height
    assert calls["count"] <= frame.height * 8


def _count_datetime_calls(monkeypatch, module: Any) -> dict[str, int]:
    original = module._datetime_value
    calls = {"count": 0}

    def wrapped(value: Any) -> datetime:
        calls["count"] += 1
        return original(value)

    monkeypatch.setattr(module, "_datetime_value", wrapped)
    return calls


def _batter_pitch_frame(*, row_count: int) -> pl.DataFrame:
    start = datetime(2024, 4, 1, 18, 0)
    outcomes = ["ball", "called_strike", "swinging_strike", "single"]
    zones = ["heart", "middle_middle", "chase_low", "chase_away"]
    return pl.DataFrame(
        {
            "game_pk": [1 + index // 40 for index in range(row_count)],
            "game_date": [
                date(2024, 4, 1) + timedelta(days=index // 40) for index in range(row_count)
            ],
            "pitch_timestamp": [start + timedelta(minutes=index) for index in range(row_count)],
            "as_of_timestamp": [
                start + timedelta(minutes=index, seconds=-1) for index in range(row_count)
            ],
            "batter_id": [200 + index % 8 for index in range(row_count)],
            "pitch_type": ["FF" if index % 2 else "SL" for index in range(row_count)],
            "relative_zone": [zones[index % len(zones)] for index in range(row_count)],
            "outcome": [outcomes[index % len(outcomes)] for index in range(row_count)],
        }
    )


def _daily_state_frame(*, row_count: int) -> pl.DataFrame:
    start = datetime(2024, 4, 1, 18, 0)
    outcomes = ["ball", "called_strike", "swinging_strike", "single"]
    return pl.DataFrame(
        {
            "game_pk": [1 + index // 80 for index in range(row_count)],
            "game_date": [
                date(2024, 4, 1) + timedelta(days=index // 80) for index in range(row_count)
            ],
            "pitch_timestamp": [start + timedelta(minutes=index) for index in range(row_count)],
            "as_of_timestamp": [
                start + timedelta(minutes=index, seconds=-1) for index in range(row_count)
            ],
            "at_bat_number": [1 + index // 4 for index in range(row_count)],
            "pitch_number": [1 + index % 4 for index in range(row_count)],
            "pitcher_id": [500 + index // 80 for index in range(row_count)],
            "batter_id": [200 + index % 9 for index in range(row_count)],
            "outcome": [outcomes[index % len(outcomes)] for index in range(row_count)],
            "runs_scored": [1 if index % 23 == 0 else 0 for index in range(row_count)],
            "plate_appearance_ended": [index % 4 == 3 for index in range(row_count)],
        }
    )


def _pitcher_profile_frame(*, row_count: int) -> pl.DataFrame:
    start = datetime(2024, 4, 1, 18, 0)
    pitch_types = ["FF", "SL", "CH", "CU"]
    return pl.DataFrame(
        {
            "game_pk": [1 + index // 20 for index in range(row_count)],
            "game_date": [
                date(2024, 4, 1) + timedelta(days=index // 20) for index in range(row_count)
            ],
            "pitch_timestamp": [start + timedelta(minutes=index) for index in range(row_count)],
            "as_of_timestamp": [
                start + timedelta(minutes=index, seconds=-1) for index in range(row_count)
            ],
            "pitcher_id": [500 + index % 5 for index in range(row_count)],
            "pitch_type": [pitch_types[index % len(pitch_types)] for index in range(row_count)],
            "release_speed": [90.0 + index % 7 for index in range(row_count)],
            "pfx_x": [float(index % 3) for index in range(row_count)],
            "pfx_z": [float(index % 5) for index in range(row_count)],
            "release_pos_x": [1.0 + index % 2 for index in range(row_count)],
            "release_pos_z": [5.0 + index % 2 for index in range(row_count)],
            "release_extension": [6.0 + index % 2 for index in range(row_count)],
        }
    )
