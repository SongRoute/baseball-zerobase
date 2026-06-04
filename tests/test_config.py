from baseball_zerobase.config import Settings


def test_settings_resolve_paths_under_project_root(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    assert settings.raw_dir == tmp_path / "data" / "raw"
    assert settings.locked_dir == tmp_path / "data" / "locked"
