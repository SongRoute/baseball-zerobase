from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StarterSettings(BaseModel):
    prior_two_season_pitches: int = 1000


class BaselineSettings(BaseModel):
    transition_min_support: int = 50
    behavior_min_support: int = 25


class SimulationSettings(BaseModel):
    trials: int = 2000
    max_pitches_per_inning: int = 150


class PathSettings(BaseModel):
    data: Path = Path("data")
    artifacts: Path = Path("artifacts")
    reports: Path = Path("reports/generated")


def _load_base_yaml(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "configs" / "base.yaml"
    if not config_path.exists():
        return {}

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    return loaded


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BASEBALL_ZEROBASE_", nested_model_default_partial_update=True)

    project_root: Path = Field(default_factory=Path.cwd)
    paths: PathSettings = Field(default_factory=PathSettings)
    starter: StarterSettings = Field(default_factory=StarterSettings)
    baseline: BaselineSettings = Field(default_factory=BaselineSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    random_seed: int = 42

    def __init__(self, **data: Any) -> None:
        project_root = Path(data.get("project_root", Path.cwd())).resolve()
        config_data = _load_base_yaml(project_root)
        settings_data = {
            key: config_data[key]
            for key in ("paths", "starter", "baseline", "simulation", "random_seed")
            if key in config_data
        }
        settings_data.update(data)
        settings_data["project_root"] = project_root
        super().__init__(**settings_data)

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def data_dir(self) -> Path:
        return self._resolve_path(self.paths.data)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def locked_dir(self) -> Path:
        return self.data_dir / "locked"

    @property
    def artifacts_dir(self) -> Path:
        return self._resolve_path(self.paths.artifacts)

    @property
    def reports_dir(self) -> Path:
        return self._resolve_path(self.paths.reports)
