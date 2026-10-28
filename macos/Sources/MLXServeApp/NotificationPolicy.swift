import Foundation

struct NotificationPolicy {
    enum Event: Equatable {
        case jobSucceeded(String)
        case jobFailed(String)
        case serverFailed
    }

    private var previousJobs: [String: String]?
    private var appServerObserved = false
    private var serverObserved = false

    mutating func observe(
        serverState: String, ownership: String?, jobs: [[String: Any]]
    ) -> [Event] {
        var events: [Event] = []
        if serverObserved && appServerObserved && serverState == "failed" {
            events.append(.serverFailed)
            appServerObserved = false
        } else if ownership == "app" {
            appServerObserved = true
        } else if ownership == "external" || serverState == "stopped" {
            appServerObserved = false
        }
        serverObserved = true

        var currentJobs: [String: String] = [:]
        for job in jobs {
            guard let id = job["id"] as? String, let status = job["status"] as? String else { continue }
            currentJobs[id] = status
            guard let previousJobs else { continue }
            if let previous = previousJobs[id],
               !["queued", "running", "cancelling", "recovery_pending"].contains(previous) {
                continue
            }
            if status == "succeeded" { events.append(.jobSucceeded(id)) }
            if ["failed", "interrupted"].contains(status) { events.append(.jobFailed(id)) }
        }
        previousJobs = currentJobs
        return events
    }
}
