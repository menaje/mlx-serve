import Foundation

/// Summaries shared by the main window and the menu bar.
struct ModelStatusSummary {
    let installedCount: Int?
    let loadedCount: Int?
    let unknownCount: Int?

    init(modelState: [String: Any]) {
        guard let models = modelState["models"] as? [[String: Any]] else {
            installedCount = nil
            loadedCount = nil
            unknownCount = nil
            return
        }
        let states = models.map { $0["load_state"] as? String ?? "unknown" }
        installedCount = models.count
        loadedCount = states.filter { $0 == "loaded" }.count
        unknownCount = states.filter { !["loaded", "not_loaded"].contains($0) }.count
    }

    var loadedText: String {
        guard let loadedCount, let unknownCount else { return "적재 상태 확인 불가" }
        if unknownCount > 0 {
            return "\(loadedCount)개 적재 확인 · \(unknownCount)개 상태 확인 불가"
        }
        return "\(loadedCount)개 메모리에 적재됨"
    }
}

struct MemoryStatusSummary {
    let systemAvailableBytes: Int64?
    let systemTotalBytes: Int64?
    let gatewayRSSBytes: Int64?

    init(server: [String: Any]) {
        let health = server["health"] as? [String: Any]
        let memory = health?["memory"] as? [String: Any]
        let snapshot = memory?["snapshot"] as? [String: Any]
        systemAvailableBytes = snapshot?["system_available_bytes"] as? Int64
        systemTotalBytes = snapshot?["system_total_bytes"] as? Int64
        gatewayRSSBytes = snapshot?["process_rss_bytes"] as? Int64
    }

    var systemText: String {
        guard let available = systemAvailableBytes, let total = systemTotalBytes else {
            return "시스템 메모리 확인 불가"
        }
        return "시스템 여유 \(ByteCountFormatter.string(fromByteCount: available, countStyle: .memory)) / 전체 \(ByteCountFormatter.string(fromByteCount: total, countStyle: .memory))"
    }

    var gatewayText: String {
        guard let rss = gatewayRSSBytes else { return "게이트웨이 프로세스 RSS 확인 불가" }
        return "게이트웨이 프로세스 RSS \(ByteCountFormatter.string(fromByteCount: rss, countStyle: .memory)); 추론 워커는 포함하지 않습니다."
    }
}
