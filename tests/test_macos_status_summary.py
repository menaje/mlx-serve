"""Check the shared native summaries for healthy and unavailable workers."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swiftc") is None,
    reason="The native summary test requires the macOS Swift compiler",
)
def test_status_summary_preserves_unknown_model_and_memory_state(tmp_path):
    source = (
        Path(__file__).resolve().parents[1] / "macos/Sources/MLXServeApp/AppStatusSummary.swift"
    )
    main = tmp_path / "main.swift"
    main.write_text(
        """
import Foundation
@main struct Smoke {
    static func main() throws {
        let missing = ModelStatusSummary(modelState: [:])
        precondition(missing.installedCount == nil && missing.loadedCount == nil)
        precondition(missing.loadedText == "적재 상태 확인 불가")

        let models: [[String: Any]] = [
            ["load_state": "loaded"], ["load_state": "not_loaded"],
            ["load_state": "unknown"], [:]
        ]
        let partial = ModelStatusSummary(modelState: ["models": models])
        precondition(partial.installedCount == 4 && partial.loadedCount == 1)
        precondition(partial.unknownCount == 2)
        precondition(partial.loadedText.contains("1개 적재 확인"))
        precondition(partial.loadedText.contains("2개 상태 확인 불가"))
        let known = ModelStatusSummary(modelState: ["models": Array(models.prefix(2))])
        precondition(known.unknownCount == 0 && !known.loadedText.contains("확인 불가"))

        let absent = MemoryStatusSummary(server: [:])
        precondition(absent.systemText == "시스템 메모리 확인 불가")
        precondition(absent.gatewayText == "게이트웨이 프로세스 RSS 확인 불가")
        let json = #"{"health":{"memory":{"snapshot":{"system_available_bytes":4096,"system_total_bytes":8192,"process_rss_bytes":2048}}}}"#
        let server = try JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
        let memory = MemoryStatusSummary(server: server)
        precondition(memory.systemAvailableBytes == 4096 && memory.systemTotalBytes == 8192)
        precondition(memory.gatewayRSSBytes == 2048)
        precondition(!memory.systemText.contains("확인 불가"))
        precondition(memory.gatewayText.contains("추론 워커는 포함하지 않습니다"))
    }
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "status-summary-test"
    subprocess.run(
        ["swiftc", "-parse-as-library", str(source), str(main), "-o", str(executable)],
        check=True, capture_output=True, timeout=60,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=20)
