from datetime import date
import json

import polars as pl

from baseball_zerobase.evaluation.rolling import evaluate_fold, evaluate_rolling, rolling_folds


def test_rolling_folds_never_train_on_validation_year() -> None:
    folds = rolling_folds()

    assert folds[0].train_years == (2022,)
    assert folds[0].validation_year == 2023
    assert folds[1].train_years == (2022, 2023)
    assert folds[1].validation_year == 2024
    assert folds[-1].train_years == (2022, 2023, 2024)
    assert folds[-1].validation_year == 2025
    assert all(fold.validation_year not in fold.train_years for fold in folds)
    assert all(max(fold.train_years) < fold.validation_year for fold in folds)


def test_evaluate_fold_fits_only_training_years_and_validates_only_validation_year() -> None:
    report = evaluate_fold(
        _fixture_frame(),
        train_years=(2022,),
        validation_year=2023,
        trials=2,
        dataset_manifest_hash="sha256:test",
    )

    assert report.train_years == (2022,)
    assert report.validation_year == 2023
    assert report.training_row_count == 3
    assert report.validation_row_count == 3
    assert report.behavior_top1_accuracy == 1.0


def test_evaluate_rolling_writes_fold_reports_and_korean_markdown_summary(tmp_path) -> None:
    dataset_path = tmp_path / "dev_dataset.parquet"
    output_dir = tmp_path / "reports"
    _fixture_frame().write_parquet(dataset_path)
    dataset_path.with_name("dev_dataset.parquet.manifest.json").write_text(
        json.dumps({"sha256": "sha256:fixture"}),
        encoding="utf-8",
    )

    summary = evaluate_rolling(dataset_path, output_dir, trials=2)

    assert len(summary.fold_reports) == 3
    assert (output_dir / "fold_2022_to_2023.json").exists()
    assert (output_dir / "fold_2022_2023_2024_to_2025.json").exists()
    markdown = (output_dir / "rolling_summary.md").read_text(encoding="utf-8")
    assert "## Korean Summary / 한국어 요약" in markdown
    assert "검증" in markdown


def _fixture_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for year in (2022, 2023, 2024, 2025):
        rows.extend(
            [
                _row(year, pitch_number=1, outs=0, outs_after=1, half_inning_ended=False),
                _row(year, pitch_number=2, outs=1, outs_after=2, half_inning_ended=False),
                _row(year, pitch_number=3, outs=2, outs_after=3, half_inning_ended=True),
            ]
        )
    return pl.DataFrame(rows)


def _row(
    year: int,
    *,
    pitch_number: int,
    outs: int,
    outs_after: int,
    half_inning_ended: bool,
) -> dict[str, object]:
    return {
        "game_pk": year,
        "game_date": date(year, 4, 1),
        "game_type": "R",
        "at_bat_number": pitch_number,
        "pitch_number": pitch_number,
        "inning": 1,
        "balls": 0,
        "strikes": 0,
        "outs": outs,
        "runners": 0,
        "batting_order_slot": pitch_number,
        "score_diff": 0,
        "lineup_ids": [101, 102, 103],
        "lineup_stands": ["R", "L", "R"],
        "stand": "R" if pitch_number != 2 else "L",
        "p_throws": "R",
        "pitch_type": "FF",
        "relative_zone": "middle_middle",
        "action": "FF:middle_middle",
        "outcome": "in_play_out",
        "balls_after": 0,
        "strikes_after": 0,
        "outs_after": outs_after,
        "runners_after": 0,
        "runs_scored": 0,
        "plate_appearance_ended": True,
        "half_inning_ended": half_inning_ended,
        "terminal_reason": "three_outs" if half_inning_ended else None,
    }
