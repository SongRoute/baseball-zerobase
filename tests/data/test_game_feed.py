import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import baseball_zerobase.cli as cli
import baseball_zerobase.data.game_feed as game_feed_module
from baseball_zerobase.data.game_feed import normalize_game_feed, normalize_pitch_events


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parents[1] / "fixtures"


def load_minimal_feed(fixture_dir: Path) -> dict[str, object]:
    return json.loads((fixture_dir / "game_feed_minimal.json").read_text(encoding="utf-8"))


def test_normalize_game_feed_extracts_starters_and_initial_lineups(fixture_dir) -> None:
    feed = load_minimal_feed(fixture_dir)
    game = normalize_game_feed(feed)
    assert game.home_starter_id == 501
    assert game.away_starter_id == 601
    assert len(game.home_initial_lineup) == 9
    assert len(game.away_initial_lineup) == 9


def test_normalize_game_feed_sorts_lineups_and_resolves_switch_hitter_stands(
    fixture_dir: Path,
) -> None:
    feed = load_minimal_feed(fixture_dir)
    game = normalize_game_feed(feed)

    assert game.home_initial_lineup == (101, 102, 103, 104, 105, 106, 107, 108, 109)
    assert game.away_initial_lineup == (201, 202, 203, 204, 205, 206, 207, 208, 209)
    assert game.home_initial_lineup_stands[2] == "R"
    assert game.away_initial_lineup_stands[2] == "L"
    assert game.first_substitution_at_bat == 3


def test_normalize_pitch_events_extracts_pitch_timestamps(fixture_dir: Path) -> None:
    feed = load_minimal_feed(fixture_dir)
    events = normalize_pitch_events(feed)

    assert [(event.at_bat_number, event.pitch_number) for event in events] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (3, 1),
    ]
    assert events[0].game_pk == 12345
    assert events[0].pitch_timestamp.isoformat() == "2024-04-01T23:05:10+00:00"
    assert events[0].completed_event_timestamp.isoformat() == "2024-04-01T23:05:14+00:00"


def test_download_game_feed_writes_raw_manifest_and_normalized_outputs(
    tmp_path: Path,
    fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feed = load_minimal_feed(fixture_dir)
    monkeypatch.setattr(game_feed_module, "fetch_game_feed", lambda game_pk: feed)

    result = game_feed_module.download_game_feed(12345, tmp_path)

    assert result.raw_path == tmp_path / "data/raw/game_feeds/12345.json"
    assert result.raw_path.exists()
    assert result.raw_manifest_path.exists()
    assert result.normalized_game_path == tmp_path / "data/normalized/games/game_pk=12345/game.parquet"
    assert result.normalized_game_path.exists()
    assert result.normalized_pitch_events_path.exists()
    assert result.pitch_event_count == 4


def test_cli_download_games_uses_game_pk_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_path = tmp_path / "games.parquet"
    parquet_path.write_bytes(b"placeholder")
    captured: dict[str, Path] = {}

    def fake_download_games_from_parquet(game_pks_parquet: Path, project_root: Path) -> list[object]:
        captured["game_pks_parquet"] = game_pks_parquet
        captured["project_root"] = project_root
        return [SimpleNamespace(pitch_event_count=2), SimpleNamespace(pitch_event_count=3)]

    monkeypatch.setattr(cli, "download_games_from_parquet", fake_download_games_from_parquet)

    result = CliRunner().invoke(
        cli.app,
        [
            "download-games",
            "--game-pks-parquet",
            str(parquet_path),
            "--config",
            str(tmp_path / "configs/base.yaml"),
        ],
    )

    assert result.exit_code == 0
    assert captured["game_pks_parquet"] == parquet_path
    assert captured["project_root"] == tmp_path
    assert "downloaded 2 game feeds" in result.stdout.lower()
