from datetime import date
from pathlib import Path

import typer

from baseball_zerobase.config import Settings
from baseball_zerobase.data.game_feed import download_games_from_parquet
from baseball_zerobase.data.statcast import download_statcast_range

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
