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
from mlx_serve.core.pid_manager import PIDManager, get_all_instances

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
    preload: Optional[list[str]] = typer.Option(
        None,
        "--preload",
        help="Model names to preload at startup (can be specified multiple times)",
    ),
) -> None:
    """Start the mlx-serve server."""
    import os

    import uvicorn

    pid_manager = PIDManager(port)

    # Check for existing server
    if pid_manager.is_process_running():
        console.print(f"[yellow]Server already running on port {port} (PID: {pid_manager.read_pid()})[/yellow]")
        console.print("Use 'mlx-serve stop' to stop it first")
        raise typer.Exit(1)

    # Cleanup stale PID file if exists
    pid_manager.cleanup_stale()

    console.print(f"[green]Starting mlx-serve on {host}:{port}[/green]")

    # Set preload environment variable if specified
    env = os.environ.copy()
    if preload:
        env["MLX_SERVE_PRELOAD_MODELS"] = ",".join(preload)
        console.print(f"[blue]Preloading models: {', '.join(preload)}[/blue]")

    if foreground:
        # Apply preload to current environment
        if preload:
            os.environ["MLX_SERVE_PRELOAD_MODELS"] = ",".join(preload)

        # Write PID for foreground mode
        pid_manager.write_pid()
        try:
            uvicorn.run(
                "mlx_serve.server:app",
                host=host,
                port=port,
                reload=reload,
            )
        finally:
            pid_manager.remove_pid()
    else:
        # Run in background
        console.print("[yellow]Running in background mode...[/yellow]")
        process = subprocess.Popen(
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
            env=env,
        )
        # Write background process PID
        pid_manager.write_pid(process.pid)
        console.print(f"[green]Server started at http://{host}:{port} (PID: {process.pid})[/green]")


@app.command()
def stop(
    port: int = typer.Option(
        settings.port,
        "--port",
        "-p",
        help="Port of the server to stop",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force kill the server immediately",
    ),
    all_instances: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Stop all running instances",
    ),
) -> None:
    """Stop the mlx-serve server."""
    if all_instances:
        instances = get_all_instances()
        if not instances:
            console.print("[yellow]No running mlx-serve instances found[/yellow]")
            return

        for inst_port, inst_pid in instances:
            pid_manager = PIDManager(inst_port)
            if pid_manager.stop_server(force=force):
                console.print(f"[green]Stopped server on port {inst_port} (PID: {inst_pid})[/green]")
            else:
                console.print(f"[red]Failed to stop server on port {inst_port}[/red]")
        return

    pid_manager = PIDManager(port)
    pid = pid_manager.read_pid()

    if pid is None:
        console.print(f"[yellow]No server running on port {port}[/yellow]")
        return

    if not pid_manager.is_process_running(pid):
        pid_manager.remove_pid()
        console.print(f"[yellow]Server on port {port} is not running (stale PID file cleaned)[/yellow]")
        return

    if pid_manager.stop_server(force=force):
        console.print(f"[green]Server stopped on port {port} (PID: {pid})[/green]")
    else:
        console.print(f"[red]Failed to stop server on port {port}[/red]")
        raise typer.Exit(1)


@app.command("status")
def server_status(
    port: int = typer.Option(
        None,
        "--port",
        "-p",
        help="Port to check (if not specified, show all)",
    ),
) -> None:
    """Check the status of mlx-serve server(s)."""
    if port is not None:
        # Check specific port
        pid_manager = PIDManager(port)
        pid = pid_manager.read_pid()

        if pid is None:
            console.print(f"[yellow]No server registered on port {port}[/yellow]")
            return

        if pid_manager.is_process_running(pid):
            console.print(f"[green]Server running on port {port} (PID: {pid})[/green]")
        else:
            console.print(f"[yellow]Server not running on port {port} (stale PID file)[/yellow]")
            pid_manager.remove_pid()
        return

    # Show all instances
    instances = get_all_instances()

    if not instances:
        console.print("[yellow]No running mlx-serve instances[/yellow]")
        console.print("Use 'mlx-serve start' to start a server")
        return

    table = Table(title="Running mlx-serve Instances")
    table.add_column("Port", style="cyan")
    table.add_column("PID", style="green")
    table.add_column("URL", style="blue")

    for inst_port, inst_pid in instances:
        table.add_row(str(inst_port), str(inst_pid), f"http://localhost:{inst_port}")

    console.print(table)


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Show current configuration",
    ),
    example: bool = typer.Option(
        False,
        "--example",
        "-e",
        help="Print example config file",
    ),
    path: bool = typer.Option(
        False,
        "--path",
        "-p",
        help="Show config file path",
    ),
) -> None:
    """Manage mlx-serve configuration."""
    from mlx_serve.core.config_loader import DEFAULT_CONFIG_PATH, get_example_config

    if example:
        console.print(get_example_config())
        return

    if path:
        console.print(f"Config file: {DEFAULT_CONFIG_PATH}")
        console.print(f"Exists: {DEFAULT_CONFIG_PATH.exists()}")
        return

    if show or not any([example, path]):
        # Show current configuration
        table = Table(title="Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Source", style="dim")

        # Determine source for each setting
        from mlx_serve.core.config_loader import get_config_values
        import os

        yaml_config = get_config_values()

        for field_name, field_info in settings.model_fields.items():
            value = getattr(settings, field_name)

            # Determine source
            env_var = f"MLX_SERVE_{field_name.upper()}"
            if env_var in os.environ:
                source = "env"
            elif field_name in yaml_config:
                source = "yaml"
            else:
                source = "default"

            # Format value
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value) if value else "(empty)"
            elif isinstance(value, Path):
                value_str = str(value)
            else:
                value_str = str(value)

            table.add_row(field_name, value_str, source)

        console.print(table)


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
    quantize_bits: Optional[int] = typer.Option(
        None,
        "--quantize",
        "-q",
        help="Quantize model after download (4 or 8 bits)",
    ),
) -> None:
    """Download and convert a model from Hugging Face."""
    if quantize_bits is not None and quantize_bits not in [4, 8]:
        console.print("[red]Quantize bits must be 4 or 8[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Pulling model: {model}[/blue]")

    model_name = None

    async def _pull():
        nonlocal model_name
        async for status in model_manager.pull_model(model, model_type):
            if status["status"] == "downloading":
                console.print("[yellow]Downloading...[/yellow]")
            elif status["status"] == "converting":
                console.print("[yellow]Converting to MLX format...[/yellow]")
            elif status["status"] == "success":
                model_name = status["name"]
                console.print(f"[green]Successfully pulled {status['name']}[/green]")
            elif status["status"] == "error":
                console.print(f"[red]Error: {status['message']}[/red]")
                raise typer.Exit(1)

    asyncio.run(_pull())

    # Quantize if requested
    if quantize_bits is not None and model_name is not None:
        from mlx_serve.core.quantizer import get_quantized_model_name, quantize_model

        console.print(f"[blue]Quantizing to {quantize_bits}-bit...[/blue]")
        success, message = quantize_model(model_name, bits=quantize_bits)
        if success:
            quantized_name = get_quantized_model_name(model_name, quantize_bits)
            console.print(f"[green]{message}[/green]")
            console.print(f"[dim]Use model name: {quantized_name}[/dim]")
        else:
            console.print(f"[red]Quantization failed: {message}[/red]")


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


@app.command()
def quantize(
    model: str = typer.Argument(..., help="Model name to quantize"),
    bits: int = typer.Option(
        4,
        "--bits",
        "-b",
        help="Number of bits for quantization (4 or 8)",
    ),
    group_size: int = typer.Option(
        64,
        "--group-size",
        "-g",
        help="Group size for quantization",
    ),
) -> None:
    """Quantize a model to reduce memory usage."""
    from mlx_serve.core.quantizer import (
        get_quantized_model_name,
        list_quantization_options,
        quantize_model,
    )

    if bits not in [4, 8]:
        console.print("[red]Bits must be 4 or 8[/red]")
        raise typer.Exit(1)

    if not model_manager.is_model_installed(model):
        console.print(f"[red]Model '{model}' not found[/red]")
        console.print("Use 'mlx-serve list' to see installed models")
        raise typer.Exit(1)

    quantized_name = get_quantized_model_name(model, bits)
    console.print(f"[blue]Quantizing {model} to {bits}-bit...[/blue]")
    console.print(f"[dim]Output: {quantized_name}[/dim]")

    success, message = quantize_model(model, bits=bits, group_size=group_size)

    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)


# Service management commands
from mlx_serve.core.service_manager import get_platform_name, get_service_manager


def _check_service_support() -> None:
    """Check if service management is supported on this platform."""
    manager = get_service_manager()
    if manager is None:
        console.print("[red]Service management is not supported on this platform[/red]")
        console.print("Supported platforms: macOS (launchd), Linux (systemd)")
        raise typer.Exit(1)


@service_app.command("install")
def service_install() -> None:
    """Install mlx-serve as a system service."""
    _check_service_support()
    manager = get_service_manager()

    console.print(f"[blue]Installing service for {get_platform_name()}...[/blue]")
    success, message = manager.install()

    if success:
        console.print(f"[green]{message}[/green]")
        console.print("Use 'mlx-serve service start' to start the service")
    else:
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)


@service_app.command("uninstall")
def service_uninstall() -> None:
    """Uninstall the mlx-serve system service."""
    _check_service_support()
    manager = get_service_manager()

    success, message = manager.uninstall()
    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[yellow]{message}[/yellow]")


@service_app.command("start")
def service_start() -> None:
    """Start the mlx-serve system service."""
    _check_service_support()
    manager = get_service_manager()

    success, message = manager.start()
    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)


@service_app.command("stop")
def service_stop() -> None:
    """Stop the mlx-serve system service."""
    _check_service_support()
    manager = get_service_manager()

    success, message = manager.stop()
    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[yellow]{message}[/yellow]")


@service_app.command("status")
def service_status() -> None:
    """Check the status of the mlx-serve service."""
    _check_service_support()
    manager = get_service_manager()

    status = manager.status()
    console.print(f"[blue]Platform: {get_platform_name()}[/blue]")

    if status["running"]:
        console.print(f"[green]Service is running ({status['service_name']})[/green]")
    elif status["installed"]:
        console.print("[yellow]Service is installed but not running[/yellow]")
    else:
        console.print("[red]Service is not installed[/red]")

    if status["installed"]:
        if status["enabled"]:
            console.print("  Auto-start at login: [green]enabled[/green]")
        else:
            console.print("  Auto-start at login: [yellow]disabled[/yellow]")


@service_app.command("enable")
def service_enable() -> None:
    """Enable mlx-serve to start automatically at login."""
    _check_service_support()
    manager = get_service_manager()

    success, message = manager.enable()
    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)


@service_app.command("disable")
def service_disable() -> None:
    """Disable mlx-serve from starting automatically at login."""
    _check_service_support()
    manager = get_service_manager()

    success, message = manager.disable()
    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
