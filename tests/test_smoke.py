from typer.testing import CliRunner

from medvfm_inspector import __version__
from medvfm_inspector.cli import app


def test_package_exposes_version() -> None:
    assert __version__


def test_cli_help_works() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    expected_help = "Inspect and compare vision foundation model representations."

    assert expected_help in result.output
