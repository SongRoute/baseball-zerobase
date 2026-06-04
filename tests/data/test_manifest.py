import hashlib

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


def test_manifest_rejects_supplied_checksum_that_differs_from_file_bytes(tmp_path) -> None:
    raw = tmp_path / "sample.bin"
    raw.write_bytes(b"immutable")
    wrong_checksum = hashlib.sha256(b"changed").hexdigest()

    with pytest.raises(ManifestConflictError, match="supplied checksum differs"):
        write_manifest(raw, source="test", request={"start": "2024-04-01"}, sha256=wrong_checksum)

    assert not raw.with_name("sample.bin.manifest.json").exists()
