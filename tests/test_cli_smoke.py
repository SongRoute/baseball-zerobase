from typer.testing import CliRunner

from baseball_zerobase.cli import app


def test_cli_help_lists_pipeline_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.stdout
