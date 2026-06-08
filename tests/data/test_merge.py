from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from baseball_zerobase.data.merge import merge_dev_regular_datasets
from baseball_zerobase.data.splits import LockedDataError


def test_merge_dev_regular_datasets_writes_output_and_manifest(tmp_path) -> None:
    first = tmp_path / "dev_dataset_2022_regular.parquet"
    second = tmp_path / "dev_dataset_2023_regular.parquet"
    output = tmp_path / "data/processed/dev_dataset/role=dev_regular/dev_dataset_merged.parquet"
    _frame(2022, game_pk=2).write_parquet(first)
    _frame(2023, game_pk=1).write_parquet(second)

    manifest = merge_dev_regular_datasets([first, second], output, label="merged")

    merged = pl.read_parquet(output)
    assert merged.height == 2
    assert merged.get_column("game_pk").to_list() == [2, 1]
    assert manifest.path.exists()
    assert manifest.row_count == 2
    assert manifest.schema_names == merged.columns


def test_merge_dev_regular_datasets_rejects_schema_mismatch(tmp_path) -> None:
    first = tmp_path / "dev_dataset_2022_regular.parquet"
    second = tmp_path / "dev_dataset_2023_regular.parquet"
    output = tmp_path / "merged.parquet"
    _frame(2022, game_pk=1).write_parquet(first)
    _frame(2023, game_pk=2).with_columns(pl.lit("extra").alias("extra")).write_parquet(second)

    with pytest.raises(ValueError, match="schema mismatch"):
        merge_dev_regular_datasets([first, second], output, label="merged")


def test_merge_dev_regular_datasets_rejects_locked_or_non_dev_rows(tmp_path) -> None:
    first = tmp_path / "dev_dataset_2024_regular.parquet"
    locked = tmp_path / "dev_dataset_locked.parquet"
    output = tmp_path / "merged.parquet"
    _frame(2024, game_pk=1).write_parquet(first)
    _frame(2026, game_pk=2, game_date=date(2026, 4, 1)).write_parquet(locked)

    with pytest.raises(LockedDataError, match="development regular-season"):
        merge_dev_regular_datasets([first, locked], output, label="merged")


def _frame(year: int, *, game_pk: int, game_date: date | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_date": [game_date or date(year, 4, 1)],
            "game_type": ["R"],
            "game_pk": [game_pk],
            "at_bat_number": [1],
            "pitch_number": [1],
            "pitch_type": ["FF"],
        }
    )
