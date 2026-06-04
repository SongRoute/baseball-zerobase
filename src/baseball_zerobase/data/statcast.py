from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
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


def _coerce_game_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value)[:10])


def _classify_roles(frame: pl.DataFrame) -> set[DatasetRole]:
    return {
        classify_row(_coerce_game_date(row["game_date"]), str(row["game_type"]))
        for row in frame.select(["game_date", "game_type"]).iter_rows(named=True)
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


def download_statcast_range(start: date, end: date, project_root: Path) -> StatcastDownloadResult:
    pandas_frame = pybaseball_statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
    if not isinstance(pandas_frame, pd.DataFrame):
        raise TypeError("pybaseball.statcast returned a non-DataFrame response")

    frame = pl.from_pandas(pandas_frame)
    _validate_required_columns(frame)
    roles = _classify_roles(frame)
    if len(roles) != 1:
        raise ValueError(f"statcast response contains mixed dataset roles: {sorted(roles)}")
    role = roles.pop()
    if role is DatasetRole.EXCLUDED:
        raise ValueError("statcast response contains only excluded dataset rows")

    data_path = statcast_partition(project_root, role, start, end).resolve()
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
        manifest = write_manifest(
            data_path,
            source="pybaseball.statcast",
            request={"start": start.isoformat(), "end": end.isoformat()},
            row_count=frame.height,
            schema_names=frame.columns,
            sha256=checksum,
        )
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return StatcastDownloadResult(
        data_path=data_path,
        manifest_path=manifest.path,
        row_count=frame.height,
        role=role,
    )
