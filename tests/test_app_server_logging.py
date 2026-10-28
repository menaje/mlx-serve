"""Exercise configured log levels against the real app-owned server runner."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("level", "retrieval_workers", "generation_workers"),
    [("DEBUG", False, False), ("INFO", False, False),
     ("WARNING", False, False), ("ERROR", False, False),
     ("DEBUG", True, False), ("WARNING", True, False),
     ("DEBUG", False, True), ("WARNING", False, True)],
)
def test_app_server_applies_saved_log_level(tmp_path, level, retrieval_workers, generation_workers):
    config_dir = tmp_path / ".mlx-serve"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(f"logging:\n  level: {level}\n")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    env = {key: value for key, value in os.environ.items() if not key.startswith("MLX_SERVE_")}
    env.update({
        "HOME": str(tmp_path),
        "PYTHONUNBUFFERED": "1",
        "MLX_SERVE_RETRIEVAL_WORKER_ISOLATION_ENABLED": str(retrieval_workers).lower(),
        "MLX_SERVE_GENERATION_WORKER_ISOLATION_ENABLED": str(generation_workers).lower(),
    })
    root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "mlx_serve.app_server_runner", "--host", "127.0.0.1",
         "--port", str(port), "--parent", str(os.getpid())],
        cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 25
        while True:
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=1):
                    break
            except (OSError, urllib.error.URLError):
                if process.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail("App-owned server did not start")
                time.sleep(0.1)

        secret = "private_input_in_logging_test_1234"
        for path, body in (
            ("/v1/completions", {"model": "not-installed", "prompt": secret}),
            ("/v1/decisions", {"state": secret}),
        ):
            request = urllib.request.Request(
                base_url + path, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(request, timeout=10).close()
            except urllib.error.HTTPError:
                pass  # Both fixture requests are intentionally invalid.
    finally:
        process.terminate()
        output, _ = process.communicate(timeout=20)

    assert ("Received legacy completion request" in output) == (level == "DEBUG")
    assert ("mlx-serve server started" in output) == (level in {"DEBUG", "INFO"})
    assert ("Started server process" in output) == (level in {"DEBUG", "INFO"})
    if retrieval_workers and level == "DEBUG":
        assert output.count("Started server process") == 4
    if generation_workers and level == "DEBUG":
        assert output.count("Started server process") == 3
    assert ("Request validation failed" in output) == (level != "ERROR")
    assert secret not in output
