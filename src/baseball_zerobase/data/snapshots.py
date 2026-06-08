from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from baseball_zerobase.data.contracts import OutcomeLabel, RelativeZone
from baseball_zerobase.data.manifest import (
    ManifestConflictError,
    RawDataManifest,
    manifest_path_for,
    sha256_file,
    write_manifest,
)
from baseball_zerobase.data.outcomes import map_outcome
from baseball_zerobase.data.splits import DatasetRole, classify_row
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
    "action",
    "description",
    "events",
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
    "action": pl.String,
    "description": pl.String,
    "events": pl.String,
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
_AUTOMATIC_OR_INTENTIONAL_LABELS = {
    "automatic_ball",
    "automatic_strike",
    "intent_ball",
    "intent_walk",
    "intentional_ball",
    "intentional_walk",
}
_SUPPORTED_OUTCOMES = {label.value for label in OutcomeLabel} - {OutcomeLabel.OTHER.value}


@dataclass(frozen=True)
class DevelopmentDataset:
    frame: pl.DataFrame
    filter_counts: dict[str, int]


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
            outs_after = _terminal_outs_after(row)
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

        pitch_type = _string_or_none(row.get("pitch_type"))
        relative_zone = _relative_zone_value(row)
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
                "pitch_type": pitch_type,
                "relative_zone": relative_zone,
                "action": _action_value(pitch_type, relative_zone),
                "description": _string_or_none(row.get("description")),
                "events": _string_or_none(row.get("events")),
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


def build_development_dataset(snapshots: pl.DataFrame) -> DevelopmentDataset:
    _require_columns(
        snapshots,
        [
            "game_date",
            "game_type",
            "is_official_starter_pitch",
            "lineup_stable",
            "starter_eligible",
            "timestamp_joined",
            "outcome",
        ],
        "development dataset snapshots",
    )

    filtered = snapshots.with_columns(
        pl.Series("dataset_role", _dataset_roles(snapshots), dtype=pl.String),
        _action_filter_expr(snapshots).alias("_has_action"),
        _supported_event_expr(snapshots).alias("_supported_strategic_event"),
    )
    filter_counts = {"input_rows": filtered.height}
    for name, expression in [
        ("dev_regular_rows", pl.col("dataset_role") == DatasetRole.DEV_REGULAR.value),
        ("official_starter_pitch_rows", pl.col("is_official_starter_pitch").fill_null(False)),
        ("stable_lineup_rows", pl.col("lineup_stable").fill_null(False)),
        ("starter_eligible_rows", pl.col("starter_eligible").fill_null(False)),
        ("non_null_action_rows", pl.col("_has_action")),
        ("timestamp_joined_rows", pl.col("timestamp_joined").fill_null(False)),
        ("supported_strategic_event_rows", pl.col("_supported_strategic_event")),
    ]:
        filtered = filtered.filter(expression)
        filter_counts[name] = filtered.height

    return DevelopmentDataset(
        frame=filtered.drop(["dataset_role", "_has_action", "_supported_strategic_event"]),
        filter_counts=filter_counts,
    )


def write_snapshot_dataset(
    snapshots: pl.DataFrame,
    output_path: Path,
    *,
    source: str,
    request: dict[str, Any],
) -> RawDataManifest:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        snapshots.write_parquet(temp_path)
        checksum = sha256_file(temp_path)
        installed = _install_immutable_parquet(temp_path, output_path, checksum, "snapshot dataset")
        try:
            return write_manifest(
                output_path,
                source=source,
                request=request,
                row_count=snapshots.height,
                schema_names=snapshots.columns,
                sha256=checksum,
            )
        except Exception:
            if installed:
                output_path.unlink(missing_ok=True)
            raise
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def write_development_dataset(
    dataset: DevelopmentDataset,
    output_path: Path,
    *,
    source: str,
    request: dict[str, Any],
    input_paths: dict[str, Path],
) -> RawDataManifest:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_checksums = {
        label: sha256_file(path.resolve()) for label, path in sorted(input_paths.items())
    }
    manifest_request = {
        **request,
        "input_checksums": input_checksums,
        "filter_counts": dataset.filter_counts,
    }
    with tempfile.NamedTemporaryFile(
        prefix=f"{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        dataset.frame.write_parquet(temp_path)
        checksum = sha256_file(temp_path)
        installed = _install_immutable_parquet(
            temp_path, output_path, checksum, "development dataset"
        )
        try:
            return write_manifest(
                output_path,
                source=source,
                request=manifest_request,
                row_count=dataset.frame.height,
                schema_names=dataset.frame.columns,
                sha256=checksum,
            )
        except Exception:
            if installed:
                output_path.unlink(missing_ok=True)
            raise
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _read_existing_manifest_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("sha256"), str):
        raise ManifestConflictError(f"existing manifest is invalid: {path}")
    return loaded["sha256"]


def _install_immutable_parquet(temp_path: Path, data_path: Path, checksum: str, label: str) -> bool:
    manifest_path = manifest_path_for(data_path)
    if data_path.exists():
        actual_checksum = sha256_file(data_path)
        existing_manifest_checksum = _read_existing_manifest_sha256(manifest_path)
        temp_path.unlink()
        if actual_checksum == checksum and (
            existing_manifest_checksum is None or existing_manifest_checksum == checksum
        ):
            return False
        if actual_checksum != checksum:
            raise ManifestConflictError(
                f"{label} already exists with a different checksum: {data_path}"
            )
        raise ManifestConflictError(
            f"{label} manifest already exists with a different checksum: {manifest_path}"
        )

    existing_manifest_checksum = _read_existing_manifest_sha256(manifest_path)
    if existing_manifest_checksum is not None and existing_manifest_checksum != checksum:
        temp_path.unlink()
        raise ManifestConflictError(
            f"{label} manifest already exists with a different checksum: {manifest_path}"
        )

    try:
        os.link(temp_path, data_path)
    except FileExistsError:
        actual_checksum = sha256_file(data_path)
        existing_manifest_checksum = _read_existing_manifest_sha256(manifest_path)
        temp_path.unlink()
        if actual_checksum == checksum and (
            existing_manifest_checksum is None or existing_manifest_checksum == checksum
        ):
            return False
        if actual_checksum != checksum:
            raise ManifestConflictError(
                f"{label} already exists with a different checksum: {data_path}"
            )
        raise ManifestConflictError(
            f"{label} manifest already exists with a different checksum: {manifest_path}"
        )
    temp_path.unlink()
    return True


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

    previous_completed_by_game: dict[int, tuple[datetime, bool]] = {}
    for row in sorted(raw_rows, key=_sort_key):
        game_pk = _integer(row, "game_pk", 0)
        game_start = _datetime(row.get("game_start_timestamp") or _EPOCH)
        pitch_timestamp = _datetime(row["pitch_timestamp"])
        completed_timestamp = _datetime(row["completed_event_timestamp"])
        previous_completed = previous_completed_by_game.get(game_pk)
        if previous_completed is None:
            row["as_of_timestamp"] = game_start
        else:
            row["as_of_timestamp"], previous_completed_joined = previous_completed
            if not previous_completed_joined:
                row["timestamp_joined"] = False
        if row["as_of_timestamp"] >= pitch_timestamp:
            row["as_of_timestamp"] = pitch_timestamp - timedelta(microseconds=1)
            row["timestamp_joined"] = False
        previous_completed_by_game[game_pk] = (completed_timestamp, bool(row["timestamp_joined"]))
        sortable_rows.append(row)
    return sortable_rows


def _normalize_pitch_row(
    row: dict[str, Any],
    *,
    index: int,
    event_lookup: dict[tuple[int, int, int], dict[str, Any]] | None,
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

    event = None
    if event_lookup is not None:
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
    elif event_lookup is not None:
        normalized["timestamp_joined"] = False
    else:
        normalized["timestamp_joined"] = (
            normalized.get("pitch_timestamp") is not None
            and normalized.get("completed_event_timestamp") is not None
        )

    if normalized.get("pitch_timestamp") is None:
        normalized["pitch_timestamp"] = _fallback_pitch_timestamp(normalized, index)
    if normalized.get("completed_event_timestamp") is None:
        normalized["completed_event_timestamp"] = _datetime(
            normalized["pitch_timestamp"]
        ) + timedelta(milliseconds=1)
    if normalized.get("game_start_timestamp") is None:
        normalized["game_start_timestamp"] = _datetime(normalized["pitch_timestamp"]) - timedelta(
            seconds=max(index + 1, 1)
        )
    return normalized


def _event_lookup(
    pitch_events_frame: pl.DataFrame | None,
) -> dict[tuple[int, int, int], dict[str, Any]] | None:
    if pitch_events_frame is None:
        return None

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


def _terminal_outs_after(row: dict[str, Any]) -> int:
    outs_before = min(_integer(row, "outs", 0), 3)
    normalized_event = _normalized_event(row.get("events"))
    outcome = map_outcome(row.get("description"), row.get("events"))
    if normalized_event in {
        "grounded_into_double_play",
        "double_play",
        "strikeout_double_play",
        "sac_fly_double_play",
    }:
        return 3 if outs_before >= 1 else 2
    if normalized_event == "triple_play":
        return 3
    if outcome is OutcomeLabel.STRIKEOUT:
        return min(outs_before + 1, 3)
    if outcome is OutcomeLabel.IN_PLAY_OUT:
        return min(outs_before + 1, 3)
    return outs_before


def _normalized_event(value: Any) -> str | None:
    normalized = _string_or_none(value)
    if normalized is None:
        return None
    return normalized.strip().lower().replace("-", "_")


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


def _action_value(pitch_type: str | None, relative_zone: str | None) -> str | None:
    if pitch_type is None or relative_zone is None:
        return None
    return f"{pitch_type}:{relative_zone}"


def _require_columns(frame: pl.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _dataset_roles(frame: pl.DataFrame) -> list[str]:
    roles: list[str] = []
    for row in frame.select(["game_date", "game_type"]).iter_rows(named=True):
        game_date = _date_or_none(row["game_date"])
        if game_date is None:
            raise ValueError("development dataset snapshots contain null game_date")
        roles.append(classify_row(game_date, str(row["game_type"])).value)
    return roles


def _action_filter_expr(frame: pl.DataFrame) -> pl.Expr:
    if "action" in frame.columns:
        return _non_empty_string("action")
    _require_columns(frame, ["pitch_type", "relative_zone"], "development dataset action")
    return _non_empty_string("pitch_type") & _non_empty_string("relative_zone")


def _supported_event_expr(frame: pl.DataFrame) -> pl.Expr:
    expression = pl.col("outcome").is_in(sorted(_SUPPORTED_OUTCOMES))
    for column in ("description", "events"):
        if column in frame.columns:
            expression = expression & ~_excluded_label_expr(column)
    return expression.fill_null(False)


def _non_empty_string(column: str) -> pl.Expr:
    return (
        pl.col(column).is_not_null() & (pl.col(column).cast(pl.String).str.len_chars() > 0)
    ).fill_null(False)


def _normalized_label(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all("-", "_")
    )


def _excluded_label_expr(column: str) -> pl.Expr:
    return (
        _normalized_label(column).is_in(sorted(_AUTOMATIC_OR_INTENTIONAL_LABELS)).fill_null(False)
    )


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
