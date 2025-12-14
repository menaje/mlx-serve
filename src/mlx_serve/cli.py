"""CLI for mlx-serve."""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from mlx_serve import __version__
from mlx_serve.config import settings
from mlx_serve.core.model_manager import ModelType, model_manager

app = typer.Typer(
    name="mlx-serve",
    help="MLX-based embedding and reranking server with OpenAI-compatible API",
    no_args_is_help=True,
)
service_app = typer.Typer(help="Manage mlx-serve as a system service")
app.add_typer(service_app, name="service")

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"mlx-serve version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """MLX-based embedding and reranking server."""
    pass


@app.command()
def start(
    host: str = typer.Option(
        settings.host,
        "--host",
        "-h",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        settings.port,
        "--port",
        "-p",
        help="Port to bind to",
    ),
    foreground: bool = typer.Option(
        False,
        "--foreground",
        "-f",
        help="Run in foreground (default: background)",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development",
    ),
) -> None:
    """Start the mlx-serve server."""
    import uvicorn

    console.print(f"[green]Starting mlx-serve on {host}:{port}[/green]")

    if foreground:
        uvicorn.run(
            "mlx_serve.server:app",
            host=host,
            port=port,
            reload=reload,
        )
    else:
        # Run in background
        console.print("[yellow]Running in background mode...[/yellow]")
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "mlx_serve.server:app",
                "--host",
                host,
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        console.print(f"[green]Server started at http://{host}:{port}[/green]")


@app.command()
def pull(
    model: str = typer.Argument(
        ...,
        help="Hugging Face model repository (e.g., Qwen/Qwen3-Embedding-0.6B)",
    ),
    model_type: ModelType = typer.Option(
        "embedding",
        "--type",
        "-t",
        help="Model type (embedding or reranker)",
    ),
) -> None:
    """Download and convert a model from Hugging Face."""
    console.print(f"[blue]Pulling model: {model}[/blue]")

    async def _pull():
        async for status in model_manager.pull_model(model, model_type):
            if status["status"] == "downloading":
                console.print("[yellow]Downloading...[/yellow]")
            elif status["status"] == "converting":
                console.print("[yellow]Converting to MLX format...[/yellow]")
            elif status["status"] == "success":
                console.print(f"[green]Successfully pulled {status['name']}[/green]")
            elif status["status"] == "error":
                console.print(f"[red]Error: {status['message']}[/red]")
                raise typer.Exit(1)

    asyncio.run(_pull())


@app.command("list")
def list_models() -> None:
    """List installed models."""
    models = model_manager.list_models()

    if not models:
        console.print("[yellow]No models installed[/yellow]")
        console.print("Use 'mlx-serve pull <model>' to download a model")
        return

    table = Table(title="Installed Models")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Size", style="yellow")
    table.add_column("Modified", style="dim")

    for m in models:
        size_mb = m.size / (1024 * 1024)
        table.add_row(
            m.name,
            m.model_type,
            f"{size_mb:.1f} MB",
            m.modified_at[:19] if m.modified_at else "-",
        )

    console.print(table)


@app.command()
def remove(
    model: str = typer.Argument(..., help="Model name to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Remove an installed model."""
    if not model_manager.is_model_installed(model):
        console.print(f"[red]Model '{model}' not found[/red]")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Remove model '{model}'?")
        if not confirm:
            raise typer.Abort()

    model_manager.delete_model(model)
    console.print(f"[green]Removed model: {model}[/green]")


# Service management commands
PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mlx-serve.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>mlx_serve</string>
        <string>start</string>
        <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/mlx-serve.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/mlx-serve.error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
"""


def _get_plist_path() -> Path:
    """Get the path to the launchd plist file."""
    return Path.home() / "Library" / "LaunchAgents" / "com.mlx-serve.server.plist"


def _get_log_dir() -> Path:
    """Get the log directory."""
    log_dir = Path.home() / ".mlx-serve" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


@service_app.command("install")
def service_install() -> None:
    """Install mlx-serve as a launchd service."""
    plist_path = _get_plist_path()
    log_dir = _get_log_dir()

    plist_content = PLIST_TEMPLATE.format(
        python_path=sys.executable,
        log_dir=str(log_dir),
    )

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)

    console.print(f"[green]Service installed at {plist_path}[/green]")
    console.print("Use 'mlx-serve service start' to start the service")


@service_app.command("uninstall")
def service_uninstall() -> None:
    """Uninstall the mlx-serve launchd service."""
    plist_path = _get_plist_path()

    # Stop service first
    subprocess.run(
        ["launchctl", "unload", str(plist_path)],
        capture_output=True,
    )

    if plist_path.exists():
        plist_path.unlink()
        console.print("[green]Service uninstalled[/green]")
    else:
        console.print("[yellow]Service was not installed[/yellow]")


@service_app.command("start")
def service_start() -> None:
    """Start the mlx-serve launchd service."""
    plist_path = _get_plist_path()

    if not plist_path.exists():
        console.print("[red]Service not installed. Run 'mlx-serve service install' first[/red]")
        raise typer.Exit(1)

    result = subprocess.run(
        ["launchctl", "load", str(plist_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print("[green]Service started[/green]")
    else:
        console.print(f"[red]Failed to start service: {result.stderr}[/red]")
        raise typer.Exit(1)


@service_app.command("stop")
def service_stop() -> None:
    """Stop the mlx-serve launchd service."""
    plist_path = _get_plist_path()

    result = subprocess.run(
        ["launchctl", "unload", str(plist_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print("[green]Service stopped[/green]")
    else:
        console.print(f"[yellow]Service may not be running: {result.stderr}[/yellow]")


@service_app.command("status")
def service_status() -> None:
    """Check the status of the mlx-serve service."""
    result = subprocess.run(
        ["launchctl", "list", "com.mlx-serve.server"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print("[green]Service is running[/green]")
        # Parse PID from output
        for line in result.stdout.strip().split("\n"):
            if "PID" not in line and line.strip():
                parts = line.split()
                if len(parts) >= 1 and parts[0].isdigit():
                    console.print(f"  PID: {parts[0]}")
    else:
        plist_path = _get_plist_path()
        if plist_path.exists():
            console.print("[yellow]Service is installed but not running[/yellow]")
        else:
            console.print("[red]Service is not installed[/red]")


if __name__ == "__main__":
    app()
