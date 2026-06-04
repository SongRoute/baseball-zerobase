from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ManifestConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawDataManifest:
    source: str
    request: dict[str, Any]
    retrieved_at: str
    byte_size: int
    row_count: int | None
    schema_names: list[str] | None
    sha256: str
    data_path: Path
    path: Path

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        payload["path"] = str(self.path)
        return payload


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def manifest_path_for(data_path: Path) -> Path:
    return data_path.with_name(f"{data_path.name}.manifest.json")


def _read_existing_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("sha256"), str):
        raise ManifestConflictError(f"existing manifest is invalid: {path}")
    return loaded["sha256"]


def write_manifest(
    data_path: Path,
    *,
    source: str,
    request: dict[str, Any],
    row_count: int | None = None,
    schema_names: list[str] | None = None,
    sha256: str | None = None,
    retrieved_at: datetime | None = None,
) -> RawDataManifest:
    data_path = data_path.resolve()
    manifest_path = manifest_path_for(data_path)
    checksum = sha256 if sha256 is not None else sha256_file(data_path)
    existing_checksum = _read_existing_sha256(manifest_path)
    if existing_checksum is not None and existing_checksum != checksum:
        raise ManifestConflictError(f"manifest checksum differs for {data_path}")

    manifest = RawDataManifest(
        source=source,
        request=request,
        retrieved_at=(retrieved_at or datetime.now(UTC)).isoformat(),
        byte_size=data_path.stat().st_size,
        row_count=row_count,
        schema_names=schema_names,
        sha256=checksum,
        data_path=data_path,
        path=manifest_path,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
