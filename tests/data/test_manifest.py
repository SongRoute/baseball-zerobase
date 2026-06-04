import hashlib
import json

import pytest

from baseball_zerobase.data import manifest as manifest_module
from baseball_zerobase.data.manifest import ManifestConflictError, sha256_file, write_manifest


def test_manifest_records_file_checksum(tmp_path) -> None:
    raw = tmp_path / "sample.bin"
    raw.write_bytes(b"immutable")

    manifest = write_manifest(raw, source="test", request={"start": "2024-04-01"})

    assert manifest.sha256 == sha256_file(raw)
    assert manifest.row_count is None
    assert manifest.byte_size == len(b"immutable")
    assert manifest.schema_names is None
    assert manifest.path == raw.with_name("sample.bin.manifest.json")


def test_manifest_allows_idempotent_rewrite_for_same_file(tmp_path) -> None:
    raw = tmp_path / "sample.bin"
    raw.write_bytes(b"immutable")

    first = write_manifest(
        raw,
        source="test",
        request={"start": "2024-04-01"},
        row_count=1,
        schema_names=["col"],
    )
    first_payload = first.path.read_text(encoding="utf-8")
    second = write_manifest(
        raw,
        source="test",
        request={"start": "2024-04-01"},
        row_count=1,
        schema_names=["col"],
    )

    assert second.path == first.path
    assert second.path.read_text(encoding="utf-8") == first_payload


@pytest.mark.parametrize(
    ("manifest_request", "row_count", "match"),
    [
        ({"start": "2024-04-02"}, 1, "manifest metadata differs"),
        ({"start": "2024-04-01"}, 2, "manifest metadata differs"),
    ],
)
def test_manifest_rejects_metadata_change_for_same_file_and_leaves_existing_manifest(
    tmp_path, manifest_request: dict[str, str], row_count: int, match: str
) -> None:
    raw = tmp_path / "sample.bin"
    raw.write_bytes(b"immutable")
    manifest = write_manifest(raw, source="test", request={"start": "2024-04-01"}, row_count=1)
    first_payload = manifest.path.read_text(encoding="utf-8")

    with pytest.raises(ManifestConflictError, match=match):
        write_manifest(raw, source="test", request=manifest_request, row_count=row_count)

    assert manifest.path.read_text(encoding="utf-8") == first_payload


def test_manifest_rejects_overwrite_when_checksum_differs(tmp_path) -> None:
    raw = tmp_path / "sample.bin"
    raw.write_bytes(b"immutable")
    write_manifest(raw, source="test", request={"start": "2024-04-01"})
    raw.write_bytes(b"changed")

    with pytest.raises(ManifestConflictError):
        write_manifest(raw, source="test", request={"start": "2024-04-01"})


def test_manifest_rejects_racing_manifest_creation_and_leaves_competing_manifest(
    tmp_path, monkeypatch
) -> None:
    raw = tmp_path / "sample.bin"
    raw.write_bytes(b"immutable")
    manifest_path = raw.with_name("sample.bin.manifest.json")
    competing_payload = {
        "source": "test",
        "request": {"start": "2024-04-02"},
        "retrieved_at": "2024-04-02T00:00:00+00:00",
        "byte_size": len(b"immutable"),
        "row_count": 2,
        "schema_names": None,
        "sha256": sha256_file(raw),
        "data_path": str(raw.resolve()),
        "path": str(manifest_path.resolve()),
    }
    competing_text = json.dumps(competing_payload, indent=2, sort_keys=True) + "\n"
    original_read_existing = manifest_module._read_existing_manifest_payload
    calls = 0

    def create_conflict_then_report_missing(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            path.write_text(competing_text, encoding="utf-8")
            return None
        return original_read_existing(path)

    monkeypatch.setattr(
        manifest_module, "_read_existing_manifest_payload", create_conflict_then_report_missing
    )

    with pytest.raises(ManifestConflictError, match="manifest metadata differs"):
        write_manifest(raw, source="test", request={"start": "2024-04-01"}, row_count=1)

    assert manifest_path.read_text(encoding="utf-8") == competing_text


def test_manifest_rejects_supplied_checksum_that_differs_from_file_bytes(tmp_path) -> None:
    raw = tmp_path / "sample.bin"
    raw.write_bytes(b"immutable")
    wrong_checksum = hashlib.sha256(b"changed").hexdigest()

    with pytest.raises(ManifestConflictError, match="supplied checksum differs"):
        write_manifest(raw, source="test", request={"start": "2024-04-01"}, sha256=wrong_checksum)

    assert not raw.with_name("sample.bin.manifest.json").exists()
