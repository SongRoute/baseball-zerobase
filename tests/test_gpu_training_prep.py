from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPU_PREP_SCRIPTS = (
    ROOT / "scripts/gpu_smoke.sh",
    ROOT / "scripts/build_dev_regular_dataset.sh",
    ROOT / "scripts/train_transition_baseline.sh",
    ROOT / "scripts/merge_dev_datasets.sh",
    ROOT / "scripts/report_transition_diagnostics.sh",
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


def test_merge_dev_datasets_contract() -> None:
    text = (ROOT / "scripts/merge_dev_datasets.sh").read_text(encoding="utf-8")

    assert "dev_dataset_${LABEL}.parquet" in text
    assert "reports/generated/validation/dev_dataset_${LABEL}.json" in text
    assert "schema mismatch" in text
    assert "development regular-season" in text
    assert "validate-dataset" in text
    assert "write_development_dataset" in text


def test_transition_diagnostics_contract() -> None:
    text = (ROOT / "scripts/report_transition_diagnostics.sh").read_text(encoding="utf-8")

    assert "transition_diagnostics" in text
    assert "pitch_type_distribution" in text
    assert "relative_zone_distribution" in text
    assert "batter_weakness_archetype_distribution" in text
    assert "batter_threat_score_bucket_distribution" in text
    assert "pitcher_profile_reliability_weight_bucket_distribution" in text
    assert "profile_feature_null_rates" in text
    assert "pitcher_pitch_type_owned_true_rate" in text
    assert "daily_state_count_summary" in text
    assert "label_outcome_distribution" in text
    assert "full_scale_${LABEL}_" in text


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
    assert "scripts/merge_dev_datasets.sh 2022_2024_regular" in english_text
    assert "scripts/merge_dev_datasets.sh 2022_2024_regular" in korean_text
    assert "candidate pitch type x 13-zone" in english_text
    assert "후보 제거 금지" in korean_text
