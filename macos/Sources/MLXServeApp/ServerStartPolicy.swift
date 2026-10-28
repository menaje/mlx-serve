import Foundation

// This policy only schedules starts; the agent still checks port ownership,
// configuration and storage before creating a server process.
struct ServerStartPolicy {
    enum Decision: Equatable {
        case startOnLaunch
        case recover(attempt: Int)
    }

    private let delays: [TimeInterval] = [5, 15, 30]
    private(set) var attempts = 0
    private var initialStatusSeen = false
    private var mayRecoverOwnedServer = false
    private var nextAttemptAt: TimeInterval?
    private var healthySince: TimeInterval?

    mutating func manualStartRequested() {
        mayRecoverOwnedServer = true
        attempts = 0
        nextAttemptAt = nil
        healthySince = nil
    }

    mutating func manualStopRequested() {
        mayRecoverOwnedServer = false
        attempts = 0
        nextAttemptAt = nil
        healthySince = nil
    }

    mutating func evaluate(
        state: String, ownership: String?, autoStart: Bool, autoRecover: Bool, now: TimeInterval
    ) -> Decision? {
        if !initialStatusSeen, ["connected", "degraded", "stopped", "failed", "port_conflict"].contains(state) {
            initialStatusSeen = true
            if state == "stopped" && autoStart {
                mayRecoverOwnedServer = true
                return .startOnLaunch
            }
        }

        if ownership == "external" {
            manualStopRequested()
            return nil
        }
        if ownership == "app" {
            mayRecoverOwnedServer = true
            if state == "connected" {
                if healthySince == nil { healthySince = now }
                // A briefly healthy but repeatedly crashing server is still
                // subject to the retry limit.
                if now - (healthySince ?? now) >= 60 {
                    attempts = 0
                    nextAttemptAt = nil
                }
            } else {
                healthySince = nil
            }
            return nil
        }
        healthySince = nil
        guard state == "failed", mayRecoverOwnedServer, autoRecover else {
            nextAttemptAt = nil
            return nil
        }
        guard attempts < delays.count else { return nil }
        if nextAttemptAt == nil {
            nextAttemptAt = now + delays[attempts]
            return nil
        }
        guard let deadline = nextAttemptAt, now >= deadline else { return nil }
        attempts += 1
        nextAttemptAt = nil
        return .recover(attempt: attempts)
    }
}
