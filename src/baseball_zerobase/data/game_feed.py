from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from baseball_zerobase.data.manifest import (
    ManifestConflictError,
    manifest_path_for,
    sha256_file,
    write_manifest,
)

GAME_FEED_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


class NormalizedGame(BaseModel):
    game_pk: int
    game_date: date
    game_type: str
    home_team_id: int
    away_team_id: int
    home_starter_id: int
    away_starter_id: int
    home_starter_throws: str
    away_starter_throws: str
    home_initial_lineup: tuple[int, ...]
    away_initial_lineup: tuple[int, ...]
    home_initial_lineup_stands: tuple[str, ...]
    away_initial_lineup_stands: tuple[str, ...]
    game_start_timestamp: datetime
    first_substitution_at_bat: int | None


class NormalizedPitchEvent(BaseModel):
    game_pk: int
    at_bat_number: int
    pitch_number: int
    pitch_timestamp: datetime
    completed_event_timestamp: datetime


@dataclass(frozen=True)
class GameFeedDownloadResult:
    raw_path: Path
    raw_manifest_path: Path
    normalized_game_path: Path
    normalized_pitch_events_path: Path
    pitch_event_count: int


JsonObject = dict[str, Any]


def _mapping(value: Any, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"expected object at {context}")
    return value


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"expected list at {context}")
    return value


def _required_int(value: Any, context: str) -> int:
    if value is None:
        raise ValueError(f"missing integer at {context}")
    return int(value)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _game_data(feed: JsonObject) -> JsonObject:
    return _mapping(feed.get("gameData"), "gameData")


def _live_data(feed: JsonObject) -> JsonObject:
    return _mapping(feed.get("liveData"), "liveData")


def _boxscore_team(feed: JsonObject, side: str) -> JsonObject:
    live_data = _live_data(feed)
    boxscore = _mapping(live_data.get("boxscore"), "liveData.boxscore")
    teams = _mapping(boxscore.get("teams"), "liveData.boxscore.teams")
    return _mapping(teams.get(side), f"liveData.boxscore.teams.{side}")


def _game_team(feed: JsonObject, side: str) -> JsonObject:
    game_data = _game_data(feed)
    teams = _mapping(game_data.get("teams"), "gameData.teams")
    return _mapping(teams.get(side), f"gameData.teams.{side}")


def _player(feed: JsonObject, player_id: int) -> JsonObject:
    game_data = _game_data(feed)
    players = _mapping(game_data.get("players"), "gameData.players")
    player = players.get(f"ID{player_id}")
    if player is None:
        raise ValueError(f"missing player metadata for ID{player_id}")
    return _mapping(player, f"gameData.players.ID{player_id}")


def _official_starter_id(feed: JsonObject, side: str) -> int:
    team = _boxscore_team(feed, side)
    pitchers = _sequence(team.get("pitchers"), f"liveData.boxscore.teams.{side}.pitchers")
    if not pitchers:
        raise ValueError(f"missing official starter for {side}")
    return int(pitchers[0])


def _team_id(feed: JsonObject, side: str) -> int:
    game_team = _game_team(feed, side)
    if "id" in game_team:
        return int(game_team["id"])
    boxscore_team = _boxscore_team(feed, side)
    team = _mapping(boxscore_team.get("team"), f"boxscore team {side}")
    return _required_int(team.get("id"), f"boxscore team {side}.id")


def _pitch_hand(feed: JsonObject, player_id: int) -> str:
    player = _player(feed, player_id)
    return str(_mapping(player.get("pitchHand"), f"player {player_id} pitchHand").get("code"))


def _bat_side(feed: JsonObject, player_id: int) -> str:
    player = _player(feed, player_id)
    return str(_mapping(player.get("batSide"), f"player {player_id} batSide").get("code"))


def _opposite_bat_side(pitcher_throws: str) -> str:
    if pitcher_throws == "R":
        return "L"
    if pitcher_throws == "L":
        return "R"
    raise ValueError(f"cannot resolve switch-hitter side against pitcher throw {pitcher_throws!r}")


def _resolved_bat_side(feed: JsonObject, player_id: int, opposing_starter_throws: str) -> str:
    bat_side = _bat_side(feed, player_id)
    if bat_side == "S":
        return _opposite_bat_side(opposing_starter_throws)
    return bat_side


def _person_id(player_entry: JsonObject) -> int:
    person = player_entry.get("person")
    if isinstance(person, dict) and "id" in person:
        return int(person["id"])
    if "id" in player_entry:
        return int(player_entry["id"])
    raise ValueError(f"missing person id in boxscore player entry: {player_entry}")


def _initial_lineup(feed: JsonObject, side: str) -> tuple[int, ...]:
    team = _boxscore_team(feed, side)
    players = _mapping(team.get("players"), f"liveData.boxscore.teams.{side}.players")
    starters: list[tuple[int, int]] = []
    for raw_player in players.values():
        player = _mapping(raw_player, f"liveData.boxscore.teams.{side}.players[]")
        batting_order = player.get("battingOrder")
        if batting_order is None:
            continue
        order = int(str(batting_order))
        if order % 100 == 0:
            starters.append((order, _person_id(player)))

    lineup = tuple(player_id for _, player_id in sorted(starters))
    if len(lineup) != 9:
        raise ValueError(f"expected 9 initial {side} lineup players, found {len(lineup)}")
    return lineup


def _lineup_stands(
    feed: JsonObject,
    lineup: tuple[int, ...],
    opposing_starter_throws: str,
) -> tuple[str, ...]:
    return tuple(_resolved_bat_side(feed, player_id, opposing_starter_throws) for player_id in lineup)


def _all_plays(feed: JsonObject) -> list[Any]:
    live_data = _live_data(feed)
    plays = _mapping(live_data.get("plays"), "liveData.plays")
    return _sequence(plays.get("allPlays"), "liveData.plays.allPlays")


def _at_bat_number(play: JsonObject) -> int:
    about = play.get("about")
    if isinstance(about, dict) and "atBatIndex" in about:
        return int(about["atBatIndex"]) + 1
    if "atBatIndex" in play:
        return int(play["atBatIndex"]) + 1
    raise ValueError("missing atBatIndex for play")


def _batting_side(play: JsonObject) -> str | None:
    about = play.get("about")
    if not isinstance(about, dict):
        return None
    half_inning = str(about.get("halfInning", "")).lower()
    if half_inning == "top":
        return "away"
    if half_inning == "bottom":
        return "home"
    return None


def _batter_id(play: JsonObject) -> int | None:
    matchup = play.get("matchup")
    if not isinstance(matchup, dict):
        return None
    batter = matchup.get("batter")
    if not isinstance(batter, dict) or "id" not in batter:
        return None
    return int(batter["id"])


def _is_substitution_event(event: JsonObject) -> bool:
    details = event.get("details")
    candidates: list[Any] = []
    if isinstance(details, dict):
        candidates.extend([details.get("event"), details.get("eventType"), details.get("description")])
    candidates.extend([event.get("event"), event.get("eventType")])
    explicit_event_types = {"defensive_switch"}
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = str(candidate).strip().lower().replace("-", "_").replace(" ", "_")
        if "substitution" in normalized or normalized in explicit_event_types:
            return True
    return False


def _first_substitution_at_bat(
    feed: JsonObject,
    home_initial_lineup: tuple[int, ...],
    away_initial_lineup: tuple[int, ...],
) -> int | None:
    initial_lineups = {"home": set(home_initial_lineup), "away": set(away_initial_lineup)}
    boundaries: list[int] = []

    for raw_play in _all_plays(feed):
        play = _mapping(raw_play, "liveData.plays.allPlays[]")
        at_bat_number = _at_bat_number(play)
        batting_side = _batting_side(play)
        batter_id = _batter_id(play)
        if batting_side is not None and batter_id is not None:
            if batter_id not in initial_lineups[batting_side]:
                boundaries.append(at_bat_number)

        for raw_event in _sequence(play.get("playEvents", []), "play.playEvents"):
            event = _mapping(raw_event, "play.playEvents[]")
            if _is_substitution_event(event):
                boundaries.append(at_bat_number)

    return min(boundaries) if boundaries else None


def normalize_game_feed(feed: JsonObject) -> NormalizedGame:
    game_data = _game_data(feed)
    game_info = _mapping(game_data.get("game"), "gameData.game")
    datetime_info = _mapping(game_data.get("datetime"), "gameData.datetime")
    game_pk = _required_int(feed.get("gamePk", game_info.get("pk")), "gamePk")
    game_start_timestamp = _parse_datetime(datetime_info.get("dateTime"))

    home_starter_id = _official_starter_id(feed, "home")
    away_starter_id = _official_starter_id(feed, "away")
    home_starter_throws = _pitch_hand(feed, home_starter_id)
    away_starter_throws = _pitch_hand(feed, away_starter_id)
    home_initial_lineup = _initial_lineup(feed, "home")
    away_initial_lineup = _initial_lineup(feed, "away")

    return NormalizedGame(
        game_pk=game_pk,
        game_date=date.fromisoformat(str(game_data.get("officialDate", game_start_timestamp.date()))),
        game_type=str(game_info.get("type")),
        home_team_id=_team_id(feed, "home"),
        away_team_id=_team_id(feed, "away"),
        home_starter_id=home_starter_id,
        away_starter_id=away_starter_id,
        home_starter_throws=home_starter_throws,
        away_starter_throws=away_starter_throws,
        home_initial_lineup=home_initial_lineup,
        away_initial_lineup=away_initial_lineup,
        home_initial_lineup_stands=_lineup_stands(feed, home_initial_lineup, away_starter_throws),
        away_initial_lineup_stands=_lineup_stands(feed, away_initial_lineup, home_starter_throws),
        game_start_timestamp=game_start_timestamp,
        first_substitution_at_bat=_first_substitution_at_bat(
            feed,
            home_initial_lineup,
            away_initial_lineup,
        ),
    )


def normalize_pitch_events(feed: JsonObject) -> tuple[NormalizedPitchEvent, ...]:
    game_data = _game_data(feed)
    game_info = _mapping(game_data.get("game"), "gameData.game")
    game_pk = _required_int(feed.get("gamePk", game_info.get("pk")), "gamePk")
    events: list[NormalizedPitchEvent] = []

    for raw_play in _all_plays(feed):
        play = _mapping(raw_play, "liveData.plays.allPlays[]")
        at_bat_number = _at_bat_number(play)
        about = play.get("about")
        play_end_time = about.get("endTime") if isinstance(about, dict) else None
        for raw_event in _sequence(play.get("playEvents", []), "play.playEvents"):
            event = _mapping(raw_event, "play.playEvents[]")
            if event.get("isPitch") is not True:
                continue
            pitch_time = event.get("startTime")
            completed_time = event.get("endTime") or play_end_time or pitch_time
            if pitch_time is None or completed_time is None:
                raise ValueError("pitch event is missing startTime/endTime")
            events.append(
                NormalizedPitchEvent(
                    game_pk=game_pk,
                    at_bat_number=at_bat_number,
                    pitch_number=_required_int(event.get("pitchNumber"), "pitchNumber"),
                    pitch_timestamp=_parse_datetime(pitch_time),
                    completed_event_timestamp=_parse_datetime(completed_time),
                )
            )

    return tuple(events)


def _checked_download_game_pk(requested_game_pk: int, normalized_game: NormalizedGame) -> int:
    if normalized_game.game_pk != requested_game_pk:
        raise ValueError(
            "downloaded game feed game_pk mismatch: "
            f"requested {requested_game_pk}, received {normalized_game.game_pk}"
        )
    return normalized_game.game_pk


@retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    wait=wait_exponential(multiplier=0.5, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _fetch_game_feed_with_client(game_pk: int, client: httpx.Client) -> JsonObject:
    response = client.get(GAME_FEED_URL_TEMPLATE.format(game_pk=game_pk))
    response.raise_for_status()
    payload = response.json()
    return _mapping(payload, "game feed response")


def fetch_game_feed(game_pk: int, client: httpx.Client | None = None) -> JsonObject:
    if client is not None:
        return _fetch_game_feed_with_client(game_pk, client)

    with httpx.Client(timeout=30) as managed_client:
        return _fetch_game_feed_with_client(game_pk, managed_client)


def _read_existing_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("sha256"), str):
        raise ManifestConflictError(f"existing manifest is invalid: {path}")
    return loaded["sha256"]


def _install_immutable_file(temp_path: Path, data_path: Path, checksum: str, label: str) -> None:
    manifest_path = manifest_path_for(data_path)
    if data_path.exists():
        actual_checksum = sha256_file(data_path)
        existing_manifest_checksum = _read_existing_sha256(manifest_path)
        if actual_checksum == checksum and (
            existing_manifest_checksum is None or existing_manifest_checksum == checksum
        ):
            temp_path.unlink()
            return
        temp_path.unlink()
        if actual_checksum != checksum:
            raise ManifestConflictError(
                f"{label} already exists with a different checksum: {data_path}"
            )
        raise ManifestConflictError(
            f"{label} manifest already exists with a different checksum: {manifest_path}"
        )

    existing_manifest_checksum = _read_existing_sha256(manifest_path)
    if existing_manifest_checksum is not None and existing_manifest_checksum != checksum:
        temp_path.unlink()
        raise ManifestConflictError(
            f"{label} manifest already exists with a different checksum: {manifest_path}"
        )

    try:
        os.link(temp_path, data_path)
    except FileExistsError:
        actual_checksum = sha256_file(data_path)
        existing_manifest_checksum = _read_existing_sha256(manifest_path)
        temp_path.unlink()
        if actual_checksum == checksum and (
            existing_manifest_checksum is None or existing_manifest_checksum == checksum
        ):
            return
        if actual_checksum != checksum:
            raise ManifestConflictError(
                f"{label} already exists with a different checksum: {data_path}"
            )
        raise ManifestConflictError(
            f"{label} manifest already exists with a different checksum: {manifest_path}"
        )
    temp_path.unlink()


def _write_raw_game_feed(feed: JsonObject, game_pk: int, project_root: Path) -> tuple[Path, Path]:
    data_path = (project_root / "data/raw/game_feeds" / f"{game_pk}.json").resolve()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{data_path.name}.",
        suffix=".tmp",
        dir=data_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        temp_path.write_text(
            json.dumps(feed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum = sha256_file(temp_path)
        _install_immutable_file(temp_path, data_path, checksum, "game feed")
        manifest = write_manifest(
            data_path,
            source="statsapi.mlb.game_feed",
            request={
                "game_pk": game_pk,
                "url": GAME_FEED_URL_TEMPLATE.format(game_pk=game_pk),
            },
            row_count=None,
            schema_names=None,
            sha256=checksum,
        )
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return data_path, manifest.path


def _write_parquet_immutable(
    frame: pl.DataFrame,
    data_path: Path,
    *,
    source: str,
    request: dict[str, Any],
    label: str,
) -> None:
    data_path = data_path.resolve()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{data_path.name}.",
        suffix=".tmp",
        dir=data_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        frame.write_parquet(temp_path)
        checksum = sha256_file(temp_path)
        _install_immutable_file(temp_path, data_path, checksum, label)
        write_manifest(
            data_path,
            source=source,
            request=request,
            row_count=frame.height,
            schema_names=frame.columns,
            sha256=checksum,
        )
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _write_normalized_game(
    game: NormalizedGame,
    data_path: Path,
    *,
    raw_sha256: str,
) -> None:
    _write_parquet_immutable(
        pl.DataFrame([game.model_dump(mode="json")]),
        data_path,
        source="mlb-statsapi-normalized-game",
        request={"game_pk": game.game_pk, "raw_sha256": raw_sha256},
        label="normalized game",
    )


def _write_normalized_pitch_events(
    pitch_events: tuple[NormalizedPitchEvent, ...],
    data_path: Path,
    *,
    game_pk: int,
    raw_sha256: str,
) -> None:
    rows = [event.model_dump(mode="json") for event in pitch_events]
    _write_parquet_immutable(
        pl.DataFrame(rows),
        data_path,
        source="mlb-statsapi-normalized-pitch-events",
        request={"game_pk": game_pk, "raw_sha256": raw_sha256},
        label="normalized pitch events",
    )


def download_game_feed(
    game_pk: int,
    project_root: Path,
    *,
    client: httpx.Client | None = None,
) -> GameFeedDownloadResult:
    feed = fetch_game_feed(game_pk, client) if client is not None else fetch_game_feed(game_pk)
    normalized_game = normalize_game_feed(feed)
    checked_game_pk = _checked_download_game_pk(game_pk, normalized_game)
    pitch_events = normalize_pitch_events(feed)

    raw_path, raw_manifest_path = _write_raw_game_feed(feed, checked_game_pk, project_root)
    raw_sha256 = sha256_file(raw_path)
    normalized_dir = (project_root / "data/normalized/games" / f"game_pk={checked_game_pk}").resolve()
    normalized_game_path = normalized_dir / "game.parquet"
    normalized_pitch_events_path = normalized_dir / "pitch_events.parquet"
    _write_normalized_game(normalized_game, normalized_game_path, raw_sha256=raw_sha256)
    _write_normalized_pitch_events(
        pitch_events,
        normalized_pitch_events_path,
        game_pk=checked_game_pk,
        raw_sha256=raw_sha256,
    )

    return GameFeedDownloadResult(
        raw_path=raw_path,
        raw_manifest_path=raw_manifest_path,
        normalized_game_path=normalized_game_path,
        normalized_pitch_events_path=normalized_pitch_events_path,
        pitch_event_count=len(pitch_events),
    )


def read_game_pks_parquet(path: Path) -> tuple[int, ...]:
    frame = pl.read_parquet(path)
    if "game_pk" not in frame.columns:
        raise ValueError(f"game_pk parquet must contain a game_pk column: {path}")
    game_pks = frame.select(pl.col("game_pk").drop_nulls().cast(pl.Int64).unique().sort()).to_series()
    return tuple(int(game_pk) for game_pk in game_pks.to_list())


def download_games_from_parquet(
    game_pks_parquet: Path,
    project_root: Path,
) -> list[GameFeedDownloadResult]:
    game_pks = read_game_pks_parquet(game_pks_parquet)
    with httpx.Client(timeout=30) as client:
        return [
            download_game_feed(game_pk, project_root, client=client)
            for game_pk in game_pks
        ]
