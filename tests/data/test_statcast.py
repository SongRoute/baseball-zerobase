import json
from datetime import date

import pandas as pd
import pytest

from baseball_zerobase.data.manifest import ManifestConflictError, sha256_file
from baseball_zerobase.data.statcast import MissingStatcastColumnsError, download_statcast_range


REQUIRED_SOURCE_COLUMNS = {
    "game_pk",
    "game_date",
    "game_type",
    "pitcher",
    "batter",
    "at_bat_number",
    "pitch_number",
    "inning",
    "inning_topbot",
    "home_team",
    "away_team",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "pitch_type",
    "zone",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "description",
    "events",
    "type",
    "bat_score",
    "fld_score",
    "post_bat_score",
    "post_fld_score",
}


def statcast_frame(**overrides: object) -> pd.DataFrame:
    row = {
        "game_pk": 1,
        "game_date": "2024-04-01",
        "game_type": "R",
        "pitcher": 111,
        "batter": 222,
        "at_bat_number": 1,
        "pitch_number": 1,
        "inning": 1,
        "inning_topbot": "Top",
        "home_team": "LAD",
        "away_team": "SF",
        "stand": "R",
        "p_throws": "L",
        "balls": 0,
        "strikes": 0,
        "outs_when_up": 0,
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "pitch_type": "FF",
        "zone": 5,
        "plate_x": 0.1,
        "plate_z": 2.5,
        "sz_top": 3.4,
        "sz_bot": 1.5,
        "description": "called_strike",
        "events": None,
        "type": "S",
        "bat_score": 0,
        "fld_score": 0,
        "post_bat_score": 0,
        "post_fld_score": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_download_statcast_writes_partition_and_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "baseball_zerobase.data.statcast.pybaseball_statcast",
        lambda **kwargs: statcast_frame(),
    )

    result = download_statcast_range(date(2024, 4, 1), date(2024, 4, 1), tmp_path)

    assert result.data_path == tmp_path / "data/raw/statcast/start=2024-04-01_end=2024-04-01.parquet"
    assert result.data_path.exists()
    assert result.manifest_path.exists()


def test_download_statcast_rejects_corrupted_existing_partition_with_matching_manifest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "baseball_zerobase.data.statcast.pybaseball_statcast",
        lambda **kwargs: statcast_frame(),
    )
    result = download_statcast_range(date(2024, 4, 1), date(2024, 4, 1), tmp_path)
    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest_checksum = manifest_payload["sha256"]

    result.data_path.write_bytes(b"corrupted partition bytes")
    assert sha256_file(result.data_path) != manifest_checksum

    with pytest.raises(ManifestConflictError, match="partition already exists"):
        download_statcast_range(date(2024, 4, 1), date(2024, 4, 1), tmp_path)

    assert result.data_path.read_bytes() == b"corrupted partition bytes"
    assert json.loads(result.manifest_path.read_text(encoding="utf-8"))["sha256"] == manifest_checksum


def test_download_statcast_calls_pybaseball_with_iso_dates(tmp_path, monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_statcast(**kwargs: str) -> pd.DataFrame:
        captured.update(kwargs)
        return statcast_frame()

    monkeypatch.setattr("baseball_zerobase.data.statcast.pybaseball_statcast", fake_statcast)

    download_statcast_range(date(2024, 4, 1), date(2024, 4, 2), tmp_path)

    assert captured == {"start_dt": "2024-04-01", "end_dt": "2024-04-02"}


def test_download_statcast_fails_before_writing_when_required_column_missing(
    tmp_path, monkeypatch
) -> None:
    frame = statcast_frame().drop(columns=["pitch_type"])
    monkeypatch.setattr(
        "baseball_zerobase.data.statcast.pybaseball_statcast",
        lambda **kwargs: frame,
    )

    with pytest.raises(MissingStatcastColumnsError, match="pitch_type"):
        download_statcast_range(date(2024, 4, 1), date(2024, 4, 1), tmp_path)

    assert list(tmp_path.rglob("*.parquet")) == []
    assert list(tmp_path.rglob("*.manifest.json")) == []


def test_download_statcast_routes_locked_role_to_locked_raw(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "baseball_zerobase.data.statcast.pybaseball_statcast",
        lambda **kwargs: statcast_frame(game_date="2026-04-01"),
    )

    result = download_statcast_range(date(2026, 4, 1), date(2026, 4, 1), tmp_path)

    assert result.data_path == tmp_path / (
        "data/locked/raw/statcast/start=2026-04-01_end=2026-04-01.parquet"
    )
    assert result.data_path.exists()


def test_download_statcast_rejects_mixed_dataset_roles_before_writing(tmp_path, monkeypatch) -> None:
    frame = pd.concat(
        [
            statcast_frame(game_pk=1, game_date="2024-04-01"),
            statcast_frame(game_pk=2, game_date="2026-04-01"),
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(
        "baseball_zerobase.data.statcast.pybaseball_statcast",
        lambda **kwargs: frame,
    )

    with pytest.raises(ValueError, match="mixed dataset roles"):
        download_statcast_range(date(2024, 4, 1), date(2026, 4, 1), tmp_path)

    assert list(tmp_path.rglob("*.parquet")) == []
    assert list(tmp_path.rglob("*.manifest.json")) == []
