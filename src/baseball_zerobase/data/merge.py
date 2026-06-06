from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import polars as pl

from baseball_zerobase.data.manifest import RawDataManifest
from baseball_zerobase.data.snapshots import DevelopmentDataset, write_development_dataset
from baseball_zerobase.data.splits import DatasetRole, LockedDataError, classify_row


_SORT_COLUMNS = ["game_date", "game_pk", "at_bat_number", "pitch_number"]


def merge_dev_regular_datasets(
    input_paths: Sequence[Path],
    output_path: Path,
    *,
    label: str,
) -> RawDataManifest:
    if len(input_paths) < 2:
        raise ValueError("merge requires at least two development regular-season datasets")

    frames: list[pl.DataFrame] = []
    input_row_counts: dict[str, int] = {}
    expected_schema: list[tuple[str, str]] | None = None
    expected_schema_path: Path | None = None

    for input_path in input_paths:
        resolved_path = input_path.resolve()
        frame = pl.read_parquet(resolved_path)
        _require_dev_regular_frame(frame, str(resolved_path))
        schema = [(name, str(dtype)) for name, dtype in frame.schema.items()]
        if expected_schema is None:
            expected_schema = schema
            expected_schema_path = resolved_path
        elif schema != expected_schema:
            raise ValueError(
                "schema mismatch between development regular-season datasets: "
                f"{expected_schema_path} != {resolved_path}"
            )
        frames.append(frame)
        input_row_counts[str(resolved_path)] = frame.height

    merged = pl.concat(frames, how="vertical")
    if all(column in merged.columns for column in _SORT_COLUMNS):
        merged = merged.sort(_SORT_COLUMNS)

    dataset = DevelopmentDataset(
        frame=merged,
        filter_counts={
            "input_dataset_count": len(input_paths),
            "input_rows": sum(input_row_counts.values()),
            "merged_rows": merged.height,
        },
    )
    return write_development_dataset(
        dataset,
        output_path,
        source="baseball-zerobase.dev-dataset-merge",
        request={
            "label": label,
            "role": DatasetRole.DEV_REGULAR.value,
            "input_paths": list(input_row_counts),
            "input_row_counts": input_row_counts,
        },
        input_paths={
            f"input_{index}": path.resolve() for index, path in enumerate(input_paths, start=1)
        },
    )


def _require_dev_regular_frame(frame: pl.DataFrame, label: str) -> None:
    missing = sorted({"game_date", "game_type"}.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing dataset role columns: {missing}")

    roles = {
        classify_row(_coerce_game_date(row["game_date"]), str(row["game_type"]))
        for row in frame.select(["game_date", "game_type"]).iter_rows(named=True)
    }
    if roles != {DatasetRole.DEV_REGULAR}:
        role_values = sorted(role.value for role in roles)
        raise LockedDataError(
            f"{label} must contain only development regular-season rows; found {role_values}"
        )


def _coerce_game_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])
