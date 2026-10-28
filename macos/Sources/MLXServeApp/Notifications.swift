import Foundation
import UserNotifications

func notificationAuthorizationText(_ status: UNAuthorizationStatus) -> String {
    switch status {
    case .authorized, .provisional: return "허용됨"
    case .denied: return "시스템에서 알림이 꺼져 있습니다"
    case .notDetermined: return "시스템 승인 필요"
    @unknown default: return "확인 불가"
    }
}

func deliverNotification(_ event: NotificationPolicy.Event) async {
    guard UserDefaults.standard.bool(forKey: "jobNotifications") else { return }
    let center = UNUserNotificationCenter.current()
    let settings = await center.notificationSettings()
    guard [.authorized, .provisional].contains(settings.authorizationStatus),
          UserDefaults.standard.bool(forKey: "jobNotifications") else { return }

    let content = UNMutableNotificationContent()
    switch event {
    case .jobSucceeded(let id):
        content.title = "모델 준비 완료"
        content.body = "작업 \(id.prefix(8))의 준비가 완료되었습니다."
    case .jobFailed(let id):
        content.title = "모델 준비 확인 필요"
        content.body = "작업 \(id.prefix(8))이 실패하거나 중단되었습니다. 앱에서 원인을 확인하세요."
    case .serverFailed:
        content.title = "서버가 중단되었습니다"
        content.body = "앱에서 서버 상태와 복구 결과를 확인하세요."
    }
    // No request text, credentials, or model input is included in notifications.
    let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
    try? await center.add(request)
}
