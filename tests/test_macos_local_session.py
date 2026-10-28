"""Exercise the native app's authenticated loopback networking boundary."""

import http.server
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swiftc") is None,
    reason="The native networking test requires the macOS Swift compiler",
)
def test_native_management_session_does_not_use_proxy_or_follow_redirect(tmp_path):
    source = Path(__file__).resolve().parents[1] / "macos/Sources/MLXServeApp/LocalSession.swift"
    main = tmp_path / "main.swift"
    main.write_text(
        """
import Foundation
@main struct Smoke {
    static func main() async throws {
        let session = makeLocalSession()
        let proxies = session.configuration.connectionProxyDictionary ?? [:]
        precondition(proxies.count == 5 && proxies.values.allSatisfy { ($0 as? Int) == 0 })
        var request = URLRequest(url: URL(string: CommandLine.arguments[1])!)
        request.setValue("Bearer local-test-secret", forHTTPHeaderField: "Authorization")
        let (_, response) = try await session.data(for: request)
        precondition((response as? HTTPURLResponse)?.statusCode == 302)
    }
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "local-session-test"
    subprocess.run(
        ["swiftc", "-parse-as-library", str(source), str(main), "-o", str(executable)],
        check=True, capture_output=True, timeout=60,
    )
    hits: dict[str, list[str | None]] = {"redirect": [], "target": [], "proxy": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            kind = self.server.kind
            hits[kind].append(self.headers.get("Authorization"))
            self.send_response(302 if kind == "redirect" else 200)
            if kind == "redirect":
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/")
            self.end_headers()

        def log_message(self, *args):
            pass

    servers = []
    try:
        for kind in ("target", "proxy", "redirect"):
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.kind = kind
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
        target, proxy, redirect = servers
        env = os.environ.copy()
        env["HTTP_PROXY"] = env["http_proxy"] = f"http://127.0.0.1:{proxy.server_port}"
        env["NO_PROXY"] = env["no_proxy"] = ""
        subprocess.run(
            [str(executable), f"http://127.0.0.1:{redirect.server_port}/"],
            env=env, check=True, capture_output=True, timeout=20,
        )
        assert hits == {"redirect": ["Bearer local-test-secret"], "proxy": [], "target": []}
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
