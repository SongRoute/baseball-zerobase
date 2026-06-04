from pathlib import Path


def test_korean_readme_exists_for_user_review() -> None:
    readme_ko = Path("README.ko.md")
    assert readme_ko.exists()
    assert "클린룸" in readme_ko.read_text(encoding="utf-8")
