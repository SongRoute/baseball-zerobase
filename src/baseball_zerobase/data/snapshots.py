from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from baseball_zerobase.data.contracts import RelativeZone
from baseball_zerobase.data.manifest import RawDataManifest, write_manifest
from baseball_zerobase.data.outcomes import map_outcome
from baseball_zerobase.data.zone_mapper import map_relative_zone

SNAPSHOT_COLUMNS = [
    "game_pk",
    "game_date",
    "game_type",
    "pitch_timestamp",
    "at_bat_number",
    "pitch_number",
    "inning",
    "inning_topbot",
    "pitcher_id",
    "batter_id",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "outs",
    "runners",
    "batting_order_slot",
    "bat_score",
    "fld_score",
    "score_diff",
    "lineup_ids",
    "lineup_stable",
    "is_official_starter_pitch",
    "lineup_stands",
    "timestamp_joined",
    "pitch_type",
    "relative_zone",
    "as_of_timestamp",
    "outcome",
    "balls_after",
    "strikes_after",
    "outs_after",
    "runners_after",
    "runs_scored",
    "plate_appearance_ended",
    "half_inning_ended",
    "terminal_reason",
]

SNAPSHOT_SCHEMA: dict[str, Any] = {
    "game_pk": pl.Int64,
    "game_date": pl.Date,
    "game_type": pl.String,
    "pitch_timestamp": pl.Datetime,
    "at_bat_number": pl.Int64,
    "pitch_number": pl.Int64,
    "inning": pl.Int64,
    "inning_topbot": pl.String,
    "pitcher_id": pl.Int64,
    "batter_id": pl.Int64,
    "stand": pl.String,
    "p_throws": pl.String,
    "balls": pl.Int64,
    "strikes": pl.Int64,
    "outs": pl.Int64,
    "runners": pl.Int64,
    "batting_order_slot": pl.Int64,
    "bat_score": pl.Int64,
    "fld_score": pl.Int64,
    "score_diff": pl.Int64,
    "lineup_ids": pl.List(pl.Int64),
    "lineup_stable": pl.Boolean,
    "is_official_starter_pitch": pl.Boolean,
    "lineup_stands": pl.List(pl.String),
    "timestamp_joined": pl.Boolean,
    "pitch_type": pl.String,
    "relative_zone": pl.String,
    "as_of_timestamp": pl.Datetime,
    "outcome": pl.String,
    "balls_after": pl.Int64,
    "strikes_after": pl.Int64,
    "outs_after": pl.Int64,
    "runners_after": pl.Int64,
    "runs_scored": pl.Int64,
    "plate_appearance_ended": pl.Boolean,
    "half_inning_ended": pl.Boolean,
    "terminal_reason": pl.String,
}

_JOIN_KEYS = ["game_pk", "at_bat_number", "pitch_number"]
_EPOCH = datetime(1970, 1, 1)


def build_snapshots(
    pitch_frame: pl.DataFrame,
    pitch_events_frame: pl.DataFrame | None = None,
) -> pl.DataFrame:
    rows = _prepared_rows(pitch_frame, pitch_events_frame)
    if not rows:
        return pl.DataFrame(schema=SNAPSHOT_SCHEMA)

    snapshots: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        next_row = rows[index + 1] if index + 1 < len(rows) else None
        same_half_next = next_row is not None and _same_half_inning(row, next_row)
        same_pa_next = (
            next_row is not None
            and same_half_next
            and row["at_bat_number"] == next_row["at_bat_number"]
        )
        plate_appearance_ended = not same_pa_next
        half_inning_ended = not same_half_next

        if next_row is not None and same_half_next:
            outs_after = _integer(next_row, "outs", 0)
            runners_after = _runner_bits(next_row)
        elif next_row is not None and row["game_pk"] == next_row["game_pk"]:
            outs_after = 3
            runners_after = 0
        else:
            outs_after = min(_integer(row, "outs", 0), 3)
            runners_after = _runner_bits(row)

        if plate_appearance_ended:
            balls_after = 0
            strikes_after = 0
        elif next_row is not None:
            balls_after = _integer(next_row, "balls", 0)
            strikes_after = _integer(next_row, "strikes", 0)
        else:
            balls_after = 0
            strikes_after = 0

        runs_scored = max(
            0,
            _integer(row, "post_bat_score", _integer(row, "bat_score", 0))
            - _integer(row, "bat_score", 0),
        )

        timestamp_joined = bool(row["timestamp_joined"])
        pitch_timestamp = _datetime(row["pitch_timestamp"])
        as_of_timestamp = _datetime(row["as_of_timestamp"])
        if timestamp_joined and as_of_timestamp >= pitch_timestamp:
            raise ValueError(
                "joined pitch timestamps must make as_of_timestamp strictly before "
                f"pitch_timestamp for {tuple(row[key] for key in _JOIN_KEYS)}"
            )

        snapshots.append(
            {
                "game_pk": row["game_pk"],
                "game_date": _date_or_none(row.get("game_date")),
                "game_type": _string_or_none(row.get("game_type")),
                "pitch_timestamp": pitch_timestamp,
                "at_bat_number": row["at_bat_number"],
                "pitch_number": row["pitch_number"],
                "inning": _integer(row, "inning", 0),
                "inning_topbot": _string_or_none(row.get("inning_topbot")),
                "pitcher_id": _integer_or_none(row.get("pitcher_id")),
                "batter_id": _integer_or_none(row.get("batter_id")),
                "stand": _string_or_none(row.get("stand")),
                "p_throws": _string_or_none(row.get("p_throws")),
                "balls": _integer(row, "balls", 0),
                "strikes": _integer(row, "strikes", 0),
                "outs": _integer(row, "outs", 0),
                "runners": _runner_bits(row),
                "batting_order_slot": _integer_or_none(row.get("batting_order_slot")),
                "bat_score": _integer(row, "bat_score", 0),
                "fld_score": _integer(row, "fld_score", 0),
                "score_diff": _integer(row, "fld_score", 0) - _integer(row, "bat_score", 0),
                "lineup_ids": _list_or_none(row.get("lineup_ids")),
                "lineup_stable": bool(row.get("lineup_stable") or False),
                "is_official_starter_pitch": bool(row.get("is_official_starter_pitch") or False),
                "lineup_stands": _list_or_none(row.get("lineup_stands")),
                "timestamp_joined": timestamp_joined,
                "pitch_type": _string_or_none(row.get("pitch_type")),
                "relative_zone": _relative_zone_value(row),
                "as_of_timestamp": as_of_timestamp,
                "outcome": str(map_outcome(row.get("description"), row.get("events"))),
                "balls_after": balls_after,
                "strikes_after": strikes_after,
                "outs_after": outs_after,
                "runners_after": runners_after,
                "runs_scored": runs_scored,
                "plate_appearance_ended": plate_appearance_ended,
                "half_inning_ended": half_inning_ended,
                "terminal_reason": _terminal_reason(
                    row,
                    next_row,
                    half_inning_ended,
                    outs_after,
                ),
            }
        )

    return pl.DataFrame(snapshots, schema=SNAPSHOT_SCHEMA).select(SNAPSHOT_COLUMNS)


def write_snapshot_dataset(
    snapshots: pl.DataFrame,
    output_path: Path,
    *,
    source: str,
    request: dict[str, Any],
) -> RawDataManifest:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots.write_parquet(output_path)
    return write_manifest(
        output_path,
        source=source,
        request=request,
        row_count=snapshots.height,
        schema_names=snapshots.columns,
    )


def _prepared_rows(
    pitch_frame: pl.DataFrame,
    pitch_events_frame: pl.DataFrame | None,
) -> list[dict[str, Any]]:
    event_lookup = _event_lookup(pitch_events_frame)
    sortable_rows: list[dict[str, Any]] = []
    raw_rows = [
        _normalize_pitch_row(row, index=index, event_lookup=event_lookup)
        for index, row in enumerate(pitch_frame.iter_rows(named=True))
    ]

    previous_completed_by_game: dict[int, datetime] = {}
    for row in sorted(raw_rows, key=_sort_key):
        game_pk = _integer(row, "game_pk", 0)
        game_start = _datetime(row.get("game_start_timestamp") or _EPOCH)
        pitch_timestamp = _datetime(row["pitch_timestamp"])
        completed_timestamp = _datetime(row["completed_event_timestamp"])
        row["as_of_timestamp"] = previous_completed_by_game.get(game_pk, game_start)
        previous_completed_by_game[game_pk] = completed_timestamp
        if row["as_of_timestamp"] >= pitch_timestamp and not bool(row["timestamp_joined"]):
            row["as_of_timestamp"] = pitch_timestamp - timedelta(microseconds=1)
        sortable_rows.append(row)
    return sortable_rows


def _normalize_pitch_row(
    row: dict[str, Any],
    *,
    index: int,
    event_lookup: dict[tuple[int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    normalized = dict(row)
    normalized["game_pk"] = _integer(normalized, "game_pk", 0)
    normalized["at_bat_number"] = _integer(normalized, "at_bat_number", 0)
    normalized["pitch_number"] = _integer(normalized, "pitch_number", 0)
    _copy_column(normalized, "pitcher", "pitcher_id")
    _copy_column(normalized, "batter", "batter_id")
    _copy_column(normalized, "outs_when_up", "outs")
    _copy_column(normalized, "current_lineup_slot", "batting_order_slot")
    _copy_column(normalized, "offense_initial_lineup", "lineup_ids")
    _copy_column(normalized, "offense_initial_lineup_stands", "lineup_stands")

    if "balls" not in normalized or normalized["balls"] is None:
        normalized["balls"] = 0
    if "strikes" not in normalized or normalized["strikes"] is None:
        normalized["strikes"] = max(0, min(_integer(normalized, "pitch_number", 1) - 1, 2))
    if "outs" not in normalized or normalized["outs"] is None:
        normalized["outs"] = 0

    event = event_lookup.get(
        (
            _integer(normalized, "game_pk", 0),
            _integer(normalized, "at_bat_number", 0),
            _integer(normalized, "pitch_number", 0),
        )
    )
    if event is not None:
        normalized["pitch_timestamp"] = event["pitch_timestamp"]
        normalized["completed_event_timestamp"] = event["completed_event_timestamp"]
        normalized["timestamp_joined"] = True
    else:
        normalized["timestamp_joined"] = (
            normalized.get("pitch_timestamp") is not None
            and normalized.get("completed_event_timestamp") is not None
        )

    if normalized.get("pitch_timestamp") is None:
        normalized["pitch_timestamp"] = _fallback_pitch_timestamp(normalized, index)
    if normalized.get("completed_event_timestamp") is None:
        normalized["completed_event_timestamp"] = _datetime(normalized["pitch_timestamp"]) + timedelta(
            milliseconds=1
        )
    if normalized.get("game_start_timestamp") is None:
        normalized["game_start_timestamp"] = _datetime(normalized["pitch_timestamp"]) - timedelta(
            seconds=max(index + 1, 1)
        )
    return normalized


def _event_lookup(pitch_events_frame: pl.DataFrame | None) -> dict[tuple[int, int, int], dict[str, Any]]:
    if pitch_events_frame is None:
        return {}

    missing = [
        key
        for key in [*_JOIN_KEYS, "pitch_timestamp", "completed_event_timestamp"]
        if key not in pitch_events_frame.columns
    ]
    if missing:
        raise ValueError(f"normalized pitch events frame is missing columns: {missing}")

    lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in pitch_events_frame.iter_rows(named=True):
        key = (
            _integer(row, "game_pk", 0),
            _integer(row, "at_bat_number", 0),
            _integer(row, "pitch_number", 0),
        )
        if key in lookup:
            raise ValueError(f"duplicate normalized pitch event key: {key}")
        lookup[key] = {
            "pitch_timestamp": row["pitch_timestamp"],
            "completed_event_timestamp": row["completed_event_timestamp"],
        }
    return lookup


def _copy_column(row: dict[str, Any], source: str, target: str) -> None:
    if target not in row and source in row:
        row[target] = row[source]


def _same_half_inning(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["game_pk"] == right["game_pk"]
        and _integer(left, "inning", 0) == _integer(right, "inning", 0)
        and _half_label(left.get("inning_topbot")) == _half_label(right.get("inning_topbot"))
    )


def _terminal_reason(
    row: dict[str, Any],
    next_row: dict[str, Any] | None,
    half_inning_ended: bool,
    outs_after: int,
) -> str | None:
    if not half_inning_ended:
        return None
    if next_row is not None and row["game_pk"] == next_row["game_pk"]:
        return "three_outs"
    if outs_after < 3:
        return "game_end"
    return "three_outs"


def _sort_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        _integer(row, "game_pk", 0),
        _integer(row, "inning", 0),
        0 if _half_label(row.get("inning_topbot")) == "top" else 1,
        _integer(row, "at_bat_number", 0),
        _integer(row, "pitch_number", 0),
    )


def _half_label(value: Any) -> str:
    normalized = _string_or_none(value)
    return "" if normalized is None else normalized.lower()


def _fallback_pitch_timestamp(row: dict[str, Any], index: int) -> datetime:
    game_start = _datetime(row.get("game_start_timestamp") or _EPOCH)
    return game_start + timedelta(seconds=index + 1)


def _runner_bits(row: dict[str, Any]) -> int:
    runners = 0
    for bit, column in ((1, "on_1b"), (2, "on_2b"), (4, "on_3b")):
        value = row.get(column)
        if value is not None and value is not False:
            runners |= bit
    return runners


def _relative_zone_value(row: dict[str, Any]) -> str | None:
    existing = row.get("relative_zone")
    if existing is not None:
        return str(existing)
    mapped = map_relative_zone(
        row.get("plate_x"),
        row.get("plate_z"),
        row.get("sz_bot"),
        row.get("sz_top"),
        row.get("stand"),
    )
    if mapped is None:
        return None
    if isinstance(mapped, RelativeZone):
        return mapped.value
    return str(mapped)


def _integer(row: dict[str, Any], key: str, default: int) -> int:
    value = row.get(key)
    if value is None:
        return default
    return int(value)


def _integer_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _list_or_none(value: Any) -> list[Any] | None:
    if value is None:
        return None
    return list(value)


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _date_or_none(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])
