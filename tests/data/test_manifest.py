import json

import pytest

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

    first = write_manifest(raw, source="test", request={"start": "2024-04-01"})
    second = write_manifest(raw, source="test", request={"start": "2024-04-01"})

    assert second.path == first.path
    assert json.loads(second.path.read_text(encoding="utf-8"))["sha256"] == sha256_file(raw)


def test_manifest_rejects_overwrite_when_checksum_differs(tmp_path) -> None:
    raw = tmp_path / "sample.bin"
    raw.write_bytes(b"immutable")
    write_manifest(raw, source="test", request={"start": "2024-04-01"})
    raw.write_bytes(b"changed")

    with pytest.raises(ManifestConflictError):
        write_manifest(raw, source="test", request={"start": "2024-04-01"})
