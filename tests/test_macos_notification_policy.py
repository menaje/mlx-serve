"""Check which native app transitions may result in a user notification."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swiftc") is None,
    reason="The native policy test requires the macOS Swift compiler",
)
def test_notifications_report_new_transitions_without_replaying_history(tmp_path):
    source = (
        Path(__file__).resolve().parents[1] / "macos/Sources/MLXServeApp/NotificationPolicy.swift"
    )
    main = tmp_path / "main.swift"
    main.write_text(
        """
import Foundation
@main struct Smoke {
    static func main() {
        var policy = NotificationPolicy()
        let old: [[String: Any]] = [["id": "old", "status": "succeeded"]]
        precondition(policy.observe(serverState: "connected", ownership: "app", jobs: old).isEmpty)
        let running: [[String: Any]] = old + [["id": "a", "status": "running"]]
        precondition(policy.observe(
            serverState: "connected", ownership: "app", jobs: running
        ).isEmpty)
        let done: [[String: Any]] = old + [
            ["id": "a", "status": "succeeded"], ["id": "fast", "status": "failed"]
        ]
        let result = policy.observe(serverState: "failed", ownership: nil, jobs: done)
        precondition(result == [.serverFailed, .jobSucceeded("a"), .jobFailed("fast")])
        precondition(policy.observe(serverState: "failed", ownership: nil, jobs: done).isEmpty)
        precondition(policy.observe(serverState: "connected", ownership: "app", jobs: done).isEmpty)
        precondition(policy.observe(serverState: "stopped", ownership: nil, jobs: done).isEmpty)
        precondition(policy.observe(serverState: "failed", ownership: nil, jobs: done).isEmpty)
        precondition(policy.observe(
            serverState: "connected", ownership: "external", jobs: done
        ).isEmpty)
        precondition(policy.observe(serverState: "failed", ownership: nil, jobs: done).isEmpty)
    }
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "notification-policy-test"
    subprocess.run(
        ["swiftc", "-parse-as-library", str(source), str(main), "-o", str(executable)],
        check=True, capture_output=True, timeout=60,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=20)
