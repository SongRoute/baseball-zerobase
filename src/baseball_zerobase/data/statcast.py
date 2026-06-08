from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from pybaseball import cache as pybaseball_cache  # type: ignore[reportMissingTypeStubs]
from pybaseball import statcast as pybaseball_statcast  # type: ignore[reportMissingTypeStubs]

from baseball_zerobase.data.manifest import (
    ManifestConflictError,
    manifest_path_for,
    sha256_file,
    write_manifest,
)
from baseball_zerobase.data.splits import DatasetRole, classify_row
from baseball_zerobase.paths import statcast_partition

REQUIRED_SOURCE_COLUMNS = frozenset(
    {
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
)


class MissingStatcastColumnsError(ValueError):
    pass


@dataclass(frozen=True)
class StatcastDownloadResult:
    data_path: Path
    manifest_path: Path
    row_count: int
    role: DatasetRole


@dataclass(frozen=True)
class StatcastChunkedDownloadResult:
    data_path: Path
    manifest_path: Path
    row_count: int
    role: DatasetRole
    chunk_paths: tuple[Path, ...]
    downloaded_chunk_count: int
    skipped_chunk_count: int


def _coerce_game_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value)[:10])


def _classify_roles(frame: pl.DataFrame) -> set[DatasetRole]:
    return {
        classify_row(_coerce_game_date(row["game_date"]), str(row["game_type"]))
        for row in frame.select(["game_date", "game_type"]).unique().iter_rows(named=True)
    }


def _validate_required_columns(frame: pl.DataFrame) -> None:
    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(frame.columns))
    if missing:
        raise MissingStatcastColumnsError(f"missing required Statcast source columns: {missing}")


def _read_manifest_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("sha256"), str):
        raise ManifestConflictError(f"existing manifest is invalid: {path}")
    return loaded["sha256"]


def _install_immutable_partition(temp_path: Path, data_path: Path, checksum: str) -> None:
    manifest_path = manifest_path_for(data_path)
    existing_manifest_checksum = _read_manifest_sha256(manifest_path)
    if existing_manifest_checksum is not None and existing_manifest_checksum != checksum:
        temp_path.unlink()
        raise ManifestConflictError(
            f"statcast manifest already exists with a different checksum: {manifest_path}"
        )

    try:
        os.link(temp_path, data_path)
    except FileExistsError:
        actual_checksum = sha256_file(data_path)
        existing_manifest_checksum = _read_manifest_sha256(manifest_path)
        temp_path.unlink()
        if actual_checksum == checksum and (
            existing_manifest_checksum is None or existing_manifest_checksum == checksum
        ):
            return
        if actual_checksum != checksum:
            raise ManifestConflictError(
                f"statcast partition already exists with a different checksum: {data_path}"
            )
        raise ManifestConflictError(
            f"statcast manifest already exists with a different checksum: {manifest_path}"
        )
    temp_path.unlink()


def _download_statcast_frame(start: date, end: date) -> pl.DataFrame:
    pandas_frame = pybaseball_statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
    if not isinstance(pandas_frame, pd.DataFrame):
        raise TypeError("pybaseball.statcast returned a non-DataFrame response")

    frame = pl.from_pandas(pandas_frame)
    _validate_required_columns(frame)
    return frame


def _write_statcast_partition(
    frame: pl.DataFrame,
    data_path: Path,
    *,
    source: str,
    request: dict[str, Any],
):
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
        _install_immutable_partition(temp_path, data_path, checksum)
        return write_manifest(
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


def download_statcast_range(start: date, end: date, project_root: Path) -> StatcastDownloadResult:
    frame = _download_statcast_frame(start, end)
    roles = _classify_roles(frame)
    if len(roles) != 1:
        raise ValueError(f"statcast response contains mixed dataset roles: {sorted(roles)}")
    role = roles.pop()
    if role is DatasetRole.EXCLUDED:
        raise ValueError("statcast response contains only excluded dataset rows")

    data_path = statcast_partition(project_root, role, start, end).resolve()
    manifest = _write_statcast_partition(
        frame,
        data_path,
        source="pybaseball.statcast",
        request={"start": start.isoformat(), "end": end.isoformat()},
    )

    return StatcastDownloadResult(
        data_path=data_path,
        manifest_path=manifest.path,
        row_count=frame.height,
        role=role,
    )


def _date_chunks(start: date, end: date, chunk_days: int) -> tuple[tuple[date, date], ...]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")
    if start > end:
        raise ValueError("start must be on or before end")

    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return tuple(chunks)


def _statcast_dev_regular_chunk_dir(project_root: Path, start: date, end: date) -> Path:
    return (
        project_root
        / "data/raw/statcast_chunks/role=dev_regular"
        / f"start={start.isoformat()}_end={end.isoformat()}"
    ).resolve()


def _statcast_chunk_path(chunk_dir: Path, start: date, end: date) -> Path:
    return chunk_dir / f"chunk_start={start.isoformat()}_end={end.isoformat()}.parquet"


def _enable_pybaseball_cache() -> None:
    enable = getattr(pybaseball_cache, "enable", None)
    if callable(enable):
        enable()


def _require_dev_regular_request_window(start: date, end: date) -> None:
    if start > end:
        raise ValueError("start must be on or before end")
    if start.year < 2022 or end.year > 2025:
        raise ValueError("chunked dev regular Statcast requests must stay within 2022-2025")


def _require_only_dev_regular_rows(frame: pl.DataFrame, label: str) -> None:
    _validate_required_columns(frame)
    game_types = set(frame.get_column("game_type").cast(pl.String).drop_nulls().unique().to_list())
    if game_types != {"R"}:
        raise ValueError(
            f"{label} must contain only regular-season game_type R rows, found {sorted(game_types)}"
        )

    roles = _classify_roles(frame)
    if roles != {DatasetRole.DEV_REGULAR}:
        found = sorted(role.value for role in roles)
        raise ValueError(
            f"{label} must contain only development regular-season rows, found {found}"
        )


def _write_statcast_chunk(frame: pl.DataFrame, chunk_path: Path) -> None:
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{chunk_path.name}.",
        suffix=".tmp",
        dir=chunk_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        frame.write_parquet(temp_path)
        os.link(temp_path, chunk_path)
    except FileExistsError:
        _require_only_dev_regular_rows(
            pl.read_parquet(chunk_path), f"existing Statcast chunk {chunk_path}"
        )
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        temp_path.unlink(missing_ok=True)


def _sort_statcast_frame(frame: pl.DataFrame) -> pl.DataFrame:
    sort_columns = [
        column
        for column in (
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "pitcher",
            "batter",
        )
        if column in frame.columns
    ]
    if not sort_columns:
        return frame
    return frame.sort(sort_columns)


def download_statcast_dev_regular_range_chunked(
    start: date,
    end: date,
    project_root: Path,
    *,
    chunk_days: int = 7,
) -> StatcastChunkedDownloadResult:
    _require_dev_regular_request_window(start, end)
    chunks = _date_chunks(start, end, chunk_days)
    _enable_pybaseball_cache()

    chunk_dir = _statcast_dev_regular_chunk_dir(project_root, start, end)
    chunk_paths: list[Path] = []
    downloaded_chunk_count = 0
    skipped_chunk_count = 0

    for chunk_start, chunk_end in chunks:
        chunk_path = _statcast_chunk_path(chunk_dir, chunk_start, chunk_end)
        if chunk_path.exists():
            _require_only_dev_regular_rows(
                pl.read_parquet(chunk_path), f"existing Statcast chunk {chunk_path}"
            )
            skipped_chunk_count += 1
        else:
            frame = _download_statcast_frame(chunk_start, chunk_end)
            _require_only_dev_regular_rows(
                frame, f"Statcast chunk {chunk_start.isoformat()}..{chunk_end.isoformat()}"
            )
            _write_statcast_chunk(frame, chunk_path)
            downloaded_chunk_count += 1
        chunk_paths.append(chunk_path)

    frames = [pl.read_parquet(chunk_path) for chunk_path in chunk_paths]
    if not frames:
        raise ValueError("no Statcast chunks were produced")
    merged = _sort_statcast_frame(pl.concat(frames, how="vertical_relaxed"))
    _require_only_dev_regular_rows(merged, "merged Statcast chunk output")

    data_path = statcast_partition(project_root, DatasetRole.DEV_REGULAR, start, end).resolve()
    manifest = _write_statcast_partition(
        merged,
        data_path,
        source="pybaseball.statcast.chunked",
        request={
            "start": start.isoformat(),
            "end": end.isoformat(),
            "role": DatasetRole.DEV_REGULAR.value,
            "chunk_days": chunk_days,
            "chunk_paths": [str(path) for path in chunk_paths],
        },
    )

    return StatcastChunkedDownloadResult(
        data_path=data_path,
        manifest_path=manifest.path,
        row_count=merged.height,
        role=DatasetRole.DEV_REGULAR,
        chunk_paths=tuple(chunk_paths),
        downloaded_chunk_count=downloaded_chunk_count,
        skipped_chunk_count=skipped_chunk_count,
    )
