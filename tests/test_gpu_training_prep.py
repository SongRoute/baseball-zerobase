from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPU_PREP_SCRIPTS = (
    ROOT / "scripts/gpu_smoke.sh",
    ROOT / "scripts/build_dev_regular_dataset.sh",
    ROOT / "scripts/train_transition_baseline.sh",
)


def test_gpu_prep_scripts_exist_and_are_executable() -> None:
    for path in GPU_PREP_SCRIPTS:
        assert path.exists(), f"{path} should exist"
        assert path.stat().st_mode & 0o111, f"{path} should be executable"


def test_gpu_prep_scripts_refuse_locked_inputs() -> None:
    for path in GPU_PREP_SCRIPTS:
        text = path.read_text(encoding="utf-8")
        assert "data/locked" in text
        assert "2026-03-25" in text
        assert "2026-05-31" in text
        assert "reject_locked_path" in text or "reject_locked_args" in text


def test_dataset_builder_downloads_only_missing_normalized_game_feeds() -> None:
    text = (ROOT / "scripts/build_dev_regular_dataset.sh").read_text(encoding="utf-8")

    assert "MISSING_GAME_PKS_PATH" in text
    assert "missing normalized game feeds" in text
    assert 'download-games --game-pks-parquet "$MISSING_GAME_PKS_PATH"' in text


def test_dataset_builder_uses_resumable_statcast_chunks() -> None:
    text = (ROOT / "scripts/build_dev_regular_dataset.sh").read_text(encoding="utf-8")

    assert "download-statcast-dev-regular-chunked" in text
    assert "data/raw/statcast_chunks/role=dev_regular" in text
    assert "--chunk-days" in text


def test_gpu_training_runbooks_are_bilingual_and_include_next_stage() -> None:
    english = ROOT / "docs/gpu-training-runbook.md"
    korean = ROOT / "docs/gpu-training-runbook.ko.md"

    assert english.exists()
    assert korean.exists()

    english_text = english.read_text(encoding="utf-8")
    korean_text = korean.read_text(encoding="utf-8")
    assert "GPU Server Quickstart" in english_text
    assert "GPU 서버 빠른 시작" in korean_text
    assert "2022-2024" in english_text
    assert "2022-2024" in korean_text
    assert "data/locked" in english_text
    assert "data/locked" in korean_text
