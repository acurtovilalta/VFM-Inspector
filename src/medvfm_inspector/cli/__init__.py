"""Command-line interface for MedVFM-Inspector."""

import typer

from medvfm_inspector import __version__

app = typer.Typer(
    add_completion=False,
    help="Inspect and compare vision foundation model representations.",
)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed MedVFM-Inspector version and exit.",
    ),
) -> None:
    """Run the MedVFM-Inspector command-line interface."""
    if version:
        typer.echo(f"medvfm {__version__}")
        raise typer.Exit()


@app.command()
def version() -> None:
    """Show the installed MedVFM-Inspector version."""
    typer.echo(f"medvfm {__version__}")


__all__ = ["app"]
