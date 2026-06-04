import json
from datetime import date, datetime
from pathlib import Path

import polars as pl
import typer

from baseball_zerobase.config import Settings
from baseball_zerobase.data.eligibility import add_starter_eligibility
from baseball_zerobase.data.game_feed import download_games_from_parquet
from baseball_zerobase.data.snapshots import (
    build_development_dataset,
    build_snapshots,
    write_development_dataset,
    write_snapshot_dataset,
)
from baseball_zerobase.data.splits import DatasetRole, LockedDataError, classify_row, require_dev_role
from baseball_zerobase.data.statcast import download_statcast_range
from baseball_zerobase.data.validation import LeakageError, ValidationReport, audit_snapshots
from baseball_zerobase.paths import require_dev_input

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def version() -> None:
    typer.echo("0.1.0")


def load_settings(config: Path) -> Settings:
    config_path = config.resolve()
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    return Settings(project_root=project_root)


@app.command("download-statcast")
def download_statcast_command(
    start: date = typer.Option(..., parser=date.fromisoformat),
    end: date = typer.Option(..., parser=date.fromisoformat),
    config: Path = typer.Option(Path("configs/base.yaml")),
) -> None:
    settings = load_settings(config)
    result = download_statcast_range(start, end, settings.project_root)
    typer.echo(result.data_path)


@app.command("download-games")
def download_games_command(
    game_pks_parquet: Path = typer.Option(...),
    config: Path = typer.Option(Path("configs/base.yaml")),
) -> None:
    settings = load_settings(config)
    results = download_games_from_parquet(game_pks_parquet, settings.project_root)
    output_dir = settings.project_root / "data/normalized/games"
    total_pitch_events = sum(result.pitch_event_count for result in results)
    typer.echo(
        f"Downloaded {len(results)} game feeds to {output_dir} "
        f"({total_pitch_events} pitch events)."
    )


def _coerce_game_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _require_dev_regular_frame(frame: pl.DataFrame, label: str) -> None:
    missing = sorted({"game_date", "game_type"}.difference(frame.columns))
    if missing:
        raise typer.BadParameter(f"{label} is missing dataset role columns: {missing}")

    roles = {
        classify_row(_coerce_game_date(row["game_date"]), str(row["game_type"]))
        for row in frame.select(["game_date", "game_type"]).iter_rows(named=True)
    }
    if roles != {DatasetRole.DEV_REGULAR}:
        raise typer.BadParameter(f"{label} must contain only development regular-season rows")
    require_dev_role(DatasetRole.DEV_REGULAR)


@app.command("build-snapshots")
def build_snapshots_command(
    prepared_pitch_parquet: Path = typer.Option(...),
    normalized_pitch_events_parquet: Path = typer.Option(...),
    output_parquet: Path = typer.Option(
        Path("data/processed/snapshots/role=dev_regular/snapshots.parquet")
    ),
    config: Path = typer.Option(Path("configs/base.yaml")),
) -> None:
    settings = load_settings(config)
    pitch_path = require_dev_input(prepared_pitch_parquet, settings)
    events_path = require_dev_input(normalized_pitch_events_parquet, settings)
    output_path = require_dev_input(output_parquet, settings)

    pitch_frame = pl.read_parquet(pitch_path)
    _require_dev_regular_frame(pitch_frame, str(pitch_path))
    pitch_events_frame = pl.read_parquet(events_path)

    snapshots = build_snapshots(pitch_frame, pitch_events_frame)
    manifest = write_snapshot_dataset(
        snapshots,
        output_path,
        source="baseball-zerobase.snapshots",
        request={
            "prepared_pitch_parquet": str(pitch_path.resolve()),
            "normalized_pitch_events_parquet": str(events_path.resolve()),
            "role": DatasetRole.DEV_REGULAR.value,
        },
    )
    typer.echo(f"Wrote {snapshots.height} snapshots to {output_path} ({manifest.path})")


@app.command("build-dev-dataset")
def build_dev_dataset_command(
    snapshots_parquet: Path = typer.Option(...),
    output_parquet: Path = typer.Option(
        Path("data/processed/dev_dataset/role=dev_regular/dev_dataset.parquet")
    ),
    min_prior_pitches: int | None = typer.Option(None),
    config: Path = typer.Option(Path("configs/base.yaml")),
) -> None:
    settings = load_settings(config)
    snapshots_path = require_dev_input(snapshots_parquet, settings)
    output_path = require_dev_input(output_parquet, settings)

    snapshots = pl.read_parquet(snapshots_path)
    _require_dev_regular_frame(snapshots, str(snapshots_path))
    eligibility_threshold = (
        settings.starter.prior_two_season_pitches
        if min_prior_pitches is None
        else min_prior_pitches
    )
    eligible_snapshots = add_starter_eligibility(
        snapshots,
        min_prior_pitches=eligibility_threshold,
    )
    dataset = build_development_dataset(eligible_snapshots)
    manifest = write_development_dataset(
        dataset,
        output_path,
        source="baseball-zerobase.dev-dataset",
        request={
            "snapshots_parquet": str(snapshots_path.resolve()),
            "role": DatasetRole.DEV_REGULAR.value,
            "min_prior_pitches": eligibility_threshold,
        },
        input_paths={"snapshots": snapshots_path},
    )
    typer.echo(f"Wrote {dataset.frame.height} development rows to {output_path} ({manifest.path})")


@app.command("validate-dataset")
def validate_dataset_command(
    input_path: Path = typer.Option(..., "--input"),
    report_path: Path = typer.Option(..., "--report"),
    config: Path = typer.Option(Path("configs/base.yaml")),
) -> None:
    settings = load_settings(config)
    try:
        dataset_path = require_dev_input(input_path, settings)
        output_path = require_dev_input(report_path, settings)
        snapshots = pl.read_parquet(dataset_path)
        report = audit_snapshots(snapshots)
    except (LockedDataError, LeakageError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _write_validation_report(report, output_path)
    typer.echo(f"Wrote validation report to {output_path}")


def _write_validation_report(report: ValidationReport, report_path: Path) -> None:
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    if report_path.suffix.lower() in {".md", ".markdown"}:
        report_path.write_text(_validation_report_markdown(payload), encoding="utf-8")
        return
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _validation_report_markdown(payload: dict[str, object]) -> str:
    lines = ["# Validation Report", ""]
    for key, value in sorted(payload.items()):
        if isinstance(value, dict):
            lines.extend([f"## {key}", ""])
            for nested_key, nested_value in sorted(value.items()):
                lines.append(f"- `{nested_key}`: {nested_value}")
            lines.append("")
        else:
            lines.append(f"- `{key}`: {value}")
    return "\n".join(lines).rstrip() + "\n"
