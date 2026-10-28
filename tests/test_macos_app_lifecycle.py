"""Check the native app's owned and external server lifetimes without a GUI."""

import asyncio
import http.server
import json
import os
import select
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def native_harness(tmp_path_factory):
    if sys.platform != "darwin":
        pytest.skip("The native lifecycle test requires macOS")
    import shutil

    if shutil.which("swiftc") is None:
        pytest.skip("The native lifecycle test requires the Swift compiler")
    root = Path(__file__).resolve().parents[1] / "macos/Sources/MLXServeApp"
    target = tmp_path_factory.mktemp("native-lifecycle")
    main = target / "main.swift"
    main.write_text(
        '''
import Foundation

@main struct Smoke {
    @MainActor static func main() async throws {
        let store = AppStore()
        for _ in 0..<200 {
            if store.agentReady { break }
            if let error = store.message { fatalError(error) }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        guard store.agentReady else { fatalError("Agent did not start") }
        if ProcessInfo.processInfo.environment["LIFECYCLE_TEST_EXTERNAL"] != "1" {
            await store.startServer()
        }
        for _ in 0..<300 {
            await store.refresh()
            if ["connected", "degraded"].contains(store.state), store.ownership != nil { break }
            if let error = store.message { fatalError(error) }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        guard let owner = store.ownership,
              ["connected", "degraded"].contains(store.state) else {
            fatalError("Server did not become available")
        }
        if ProcessInfo.processInfo.environment["LIFECYCLE_TEST_JOB"] == "1" {
            await store.startJob(repo: "LiquidAI/d1-omni-600M", name: "quit-job",
                                 revision: "", keepOriginal: false, loadAfter: false)
            if let error = store.message { fatalError(error) }
        }
        print("READY \\(owner)")
        fflush(stdout)
        _ = readLine()
    }
}
''',
        encoding="utf-8",
    )
    executable = target / "lifecycle-test"
    sources = ["AppStore.swift", "LocalSession.swift", "Notifications.swift",
               "NotificationPolicy.swift", "ServerStartPolicy.swift"]
    subprocess.run(
        ["swiftc", "-parse-as-library", *(str(root / item) for item in sources),
         str(main), "-o", str(executable)],
        check=True, capture_output=True, timeout=90,
    )
    return executable


def _processes():
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True, check=True,
        timeout=5,
    )
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            yield int(parts[0]), int(parts[1]), parts[2]


def _find_child(parent: int, name: str) -> int | None:
    return next((pid for pid, ppid, cmd in _processes() if ppid == parent and name in cmd), None)


def _has_process(pid: int, name: str) -> bool:
    return any(process == pid and name in cmd for process, _, cmd in _processes())


def _is_server(cmd: str) -> bool:
    # The gateway may replace its original command with a short process title.
    return cmd.startswith("mlx-serve") or "uvicorn mlx_serve.server:app" in cmd


def _find_server(parent: int) -> int | None:
    return next(
        (pid for pid, ppid, cmd in _processes() if ppid == parent and _is_server(cmd)), None
    )


def _has_server(pid: int) -> bool:
    return any(process == pid and _is_server(cmd) for process, _, cmd in _processes())


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize(
    ("external", "abrupt", "active_job", "agent_crash"),
    [(False, False, False, False), (False, True, False, False),
     (True, False, False, False), (False, False, True, False),
     (False, True, True, False), (False, False, False, True),
     (False, False, True, True)],
)
def test_app_exit_stops_only_its_own_server(
    native_harness, tmp_path, external, abrupt, active_job, agent_crash,
):
    home = tmp_path / "home"
    home.mkdir()
    models = tmp_path / "data" / "models"
    models.mkdir(parents=True)
    server = None
    hub = None
    hub_started = threading.Event()
    hub_release = threading.Event()
    if active_job:
        class SlowHub(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                hub_started.set()
                hub_release.wait(timeout=60)
                self.send_error(503)

            def log_message(self, *args):
                pass

        hub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SlowHub)
        threading.Thread(target=hub.serve_forever, daemon=True).start()
    if external:
        class Health(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    body = json.dumps({"role": "gateway", "version": "test", "status": "healthy"})
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(body.encode())
                else:
                    self.send_error(404)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Health)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port if server is not None else _port()
    env = os.environ.copy()
    env.update({
        "HOME": str(home), "HF_HOME": str(home / "hf"),
        "MLX_SERVE_PYTHON": os.environ.get("MLX_SERVE_TEST_PYTHON", sys.executable),
        "MLX_SERVE_HOST": "127.0.0.1",
        "MLX_SERVE_PORT": str(port), "MLX_SERVE_MODELS_DIR": str(models),
        "LIFECYCLE_TEST_EXTERNAL": "1" if external else "0",
        "LIFECYCLE_TEST_JOB": "1" if active_job else "0",
    })
    if hub is not None:
        env["HF_ENDPOINT"] = f"http://127.0.0.1:{hub.server_port}"
    app = subprocess.Popen(
        [str(native_harness)], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    agent_pid = server_pid = job_pid = None
    worker_pids = []
    try:
        assert app.stdout is not None
        ready, _, _ = select.select([app.stdout], [], [], 45)
        assert ready, "Native test app did not start"
        output = app.stdout.readline().strip()
        assert output == f"READY {'external' if external else 'app'}", output
        agent_pid = _find_child(app.pid, "mlx_serve.management_agent")
        assert agent_pid is not None
        if not external:
            server_pid = _find_server(agent_pid)
            assert server_pid is not None
            worker_pids = [
                pid for pid, ppid, command in _processes()
                if ppid == server_pid and _is_server(command)
            ]
        if active_job:
            assert hub_started.wait(timeout=25), "Model preparation did not start"
            job_pid = _find_child(agent_pid, "mlx_serve.core.managed_job_runner")
            assert job_pid is not None
        if agent_crash:
            os.kill(agent_pid, signal.SIGKILL)
        if abrupt:
            app.kill()
        else:
            assert app.stdin is not None
            app.stdin.write("quit\n")
            app.stdin.flush()
        app.wait(timeout=10)
        assert (app.returncode != 0) if abrupt else (app.returncode == 0)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if not _has_process(agent_pid, "mlx_serve.management_agent") and (
                server_pid is None or not _has_server(server_pid)
            ) and (job_pid is None or not _has_process(job_pid, "managed_job_runner")) and all(
                not _has_server(pid) for pid in worker_pids
            ):
                break
            time.sleep(0.2)
        assert not _has_process(agent_pid, "mlx_serve.management_agent")
        if server_pid is not None:
            assert not _has_server(server_pid)
            logs = list((home / ".mlx-serve" / "app-server-logs").glob("*.log"))
            assert len(logs) == 1
            assert logs[0].name.startswith("running-" if agent_crash else "server-")
            assert logs[0].stat().st_mode & 0o077 == 0
        assert all(not _has_server(pid) for pid in worker_pids)
        if job_pid is not None:
            assert not _has_process(job_pid, "managed_job_runner")
            job_dir = home / ".mlx-serve" / "jobs"
            if agent_crash:
                from mlx_serve.core.managed_jobs import JobManager

                async def recover_job():
                    manager = JobManager(job_dir)
                    try:
                        await manager.recover()
                        return manager.list()[0]["status"]
                    finally:
                        await manager.close()

                assert asyncio.run(recover_job()) == "interrupted"
            else:
                with sqlite3.connect(job_dir / "jobs.sqlite3") as db:
                    assert db.execute("SELECT status FROM jobs").fetchone() == ("cancelled",)
            assert not list(models.glob(".decision-prepare-*"))
        if server is not None:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                pass
    finally:
        if app.poll() is None:
            app.terminate()
            app.wait(timeout=10)
        if agent_pid is not None and _has_process(agent_pid, "mlx_serve.management_agent"):
            os.kill(agent_pid, signal.SIGTERM)
        if server_pid is not None and _has_server(server_pid):
            os.kill(server_pid, signal.SIGTERM)
        for pid in worker_pids:
            if _has_server(pid):
                os.kill(pid, signal.SIGTERM)
        if job_pid is not None and _has_process(job_pid, "managed_job_runner"):
            os.kill(job_pid, signal.SIGKILL)
        if server is not None:
            server.shutdown()
            server.server_close()
        if hub is not None:
            hub_release.set()
            hub.shutdown()
            hub.server_close()
