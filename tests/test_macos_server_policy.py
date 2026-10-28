"""Exercise the native app's bounded server start and recovery policy."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swiftc") is None,
    reason="The native policy test requires the macOS Swift compiler",
)
def test_automatic_start_and_recovery_respect_ownership_and_retry_limit(tmp_path):
    source = (
        Path(__file__).resolve().parents[1] / "macos/Sources/MLXServeApp/ServerStartPolicy.swift"
    )
    main = tmp_path / "main.swift"
    main.write_text(
        """
import Foundation
@main struct Smoke {
    static func step(
        _ policy: inout ServerStartPolicy, _ state: String, _ owner: String?, _ time: Double,
        start: Bool = true, recover: Bool = true
    ) -> ServerStartPolicy.Decision? {
        policy.evaluate(
            state: state, ownership: owner, autoStart: start, autoRecover: recover, now: time
        )
    }
    static func main() {
        var external = ServerStartPolicy()
        precondition(step(&external, "connecting", nil, 0) == nil)
        precondition(step(&external, "connected", "external", 1) == nil)
        precondition(step(&external, "stopped", nil, 2) == nil)

        var policy = ServerStartPolicy()
        precondition(step(&policy, "stopped", nil, 0) == .startOnLaunch)
        precondition(step(&policy, "failed", nil, 1) == nil)
        precondition(step(&policy, "failed", nil, 5) == nil)
        precondition(step(&policy, "failed", nil, 6) == .recover(attempt: 1))
        precondition(step(&policy, "starting", "app", 7) == nil)
        precondition(step(&policy, "connected", "app", 8) == nil)
        precondition(step(&policy, "failed", nil, 9) == nil)
        precondition(step(&policy, "failed", nil, 24) == .recover(attempt: 2))
        precondition(step(&policy, "starting", "app", 25) == nil)
        precondition(step(&policy, "failed", nil, 26) == nil)
        precondition(step(&policy, "failed", nil, 56) == .recover(attempt: 3))
        precondition(step(&policy, "starting", "app", 57) == nil)
        precondition(step(&policy, "failed", nil, 58) == nil)
        precondition(step(&policy, "failed", nil, 1000) == nil)
        precondition(policy.attempts == 3)

        precondition(step(&policy, "connected", "app", 1001) == nil)
        precondition(step(&policy, "connected", "app", 1061) == nil)
        precondition(policy.attempts == 0)
        precondition(step(&policy, "failed", nil, 1062) == nil)
        precondition(step(&policy, "failed", nil, 1067) == .recover(attempt: 1))
        policy.manualStopRequested()
        precondition(step(&policy, "failed", nil, 2000) == nil)

        var disabled = ServerStartPolicy()
        disabled.manualStartRequested()
        precondition(step(&disabled, "failed", nil, 0, start: false, recover: false) == nil)
        precondition(step(&disabled, "failed", nil, 100, start: false, recover: false) == nil)
        precondition(disabled.attempts == 0)
        precondition(step(&disabled, "port_conflict", nil, 101) == nil)
    }
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "server-policy-test"
    subprocess.run(
        ["swiftc", "-parse-as-library", str(source), str(main), "-o", str(executable)],
        check=True, capture_output=True, timeout=60,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=20)
