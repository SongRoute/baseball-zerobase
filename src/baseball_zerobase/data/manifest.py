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


def _read_existing_manifest_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ManifestConflictError(f"existing manifest is invalid: {path}")
    return loaded


def _manifest_from_payload(
    payload: dict[str, Any], *, data_path: Path, manifest_path: Path
) -> RawDataManifest:
    source = payload.get("source")
    request = payload.get("request")
    retrieved_at = payload.get("retrieved_at")
    byte_size = payload.get("byte_size")
    row_count = payload.get("row_count")
    schema_names = payload.get("schema_names")
    sha256 = payload.get("sha256")
    if (
        not isinstance(source, str)
        or not isinstance(request, dict)
        or not isinstance(retrieved_at, str)
        or not isinstance(byte_size, int)
        or not (row_count is None or isinstance(row_count, int))
        or not (schema_names is None or isinstance(schema_names, list))
        or not isinstance(sha256, str)
    ):
        raise ManifestConflictError(f"existing manifest is invalid: {manifest_path}")
    if schema_names is not None and not all(isinstance(name, str) for name in schema_names):
        raise ManifestConflictError(f"existing manifest is invalid: {manifest_path}")
    return RawDataManifest(
        source=source,
        request=request,
        retrieved_at=retrieved_at,
        byte_size=byte_size,
        row_count=row_count,
        schema_names=schema_names,
        sha256=sha256,
        data_path=data_path,
        path=manifest_path,
    )


def _ensure_existing_manifest_matches(
    existing: RawDataManifest, candidate: RawDataManifest
) -> RawDataManifest:
    if (
        existing.source != candidate.source
        or existing.request != candidate.request
        or existing.byte_size != candidate.byte_size
        or existing.row_count != candidate.row_count
        or existing.schema_names != candidate.schema_names
        or existing.sha256 != candidate.sha256
    ):
        raise ManifestConflictError(f"manifest metadata differs for {candidate.data_path}")
    return existing


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
    actual_checksum = sha256_file(data_path)
    if sha256 is not None and sha256 != actual_checksum:
        raise ManifestConflictError(f"supplied checksum differs from file bytes for {data_path}")
    checksum = actual_checksum

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
    existing_payload = _read_existing_manifest_payload(manifest_path)
    if existing_payload is not None:
        existing_manifest = _manifest_from_payload(
            existing_payload, data_path=data_path, manifest_path=manifest_path
        )
        return _ensure_existing_manifest_matches(existing_manifest, manifest)

    manifest_path.write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
