from datetime import date
from pathlib import Path

from baseball_zerobase.config import Settings
from baseball_zerobase.data.splits import DatasetRole, guard_dev_path


def statcast_partition(root: Path, role: DatasetRole, start: date, end: date) -> Path:
    base = (
        root
        / "data"
        / ("locked/raw/statcast" if role is not DatasetRole.DEV_REGULAR else "raw/statcast")
    )
    return base / f"start={start.isoformat()}_end={end.isoformat()}.parquet"


def require_dev_input(path: Path, settings: Settings) -> Path:
    guard_dev_path(path, settings.locked_dir)
    return path
