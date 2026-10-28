import AppKit
import Foundation
import Security

@MainActor
final class AppStore: ObservableObject {
    @Published var server: [String: Any] = [:]
    @Published var modelState: [String: Any] = [:]
    @Published var jobs: [[String: Any]] = []
    @Published var jobDetail: [String: Any] = [:]
    @Published var settings: [String: Any] = [:]
    @Published var runtime: [String: Any] = [:]
    @Published var selectedJob: String?
    @Published var message: String?
    @Published var agentReady = false
    @Published var modelError: String?
    @Published var busy = false
    @Published var automaticStartNote: String?

    private var agent: Process?
    private var agentPort: Int?
    private var secret = ""
    private var autoLoad = Set<String>()
    private let localSession = makeLocalSession()
    private var serverStartPolicy = ServerStartPolicy()
    private var notificationPolicy = NotificationPolicy()

    var state: String {
        let serverState = server["state"] as? String ?? "connecting"
        return serverState == "connected" && (modelError != nil || modelState["state"] as? String == "degraded")
            ? "degraded" : serverState
    }
    var ownership: String? { server["ownership"] as? String }
    var baseURL: String { server["url"] as? String ?? "http://127.0.0.1:8000" }
    var models: [[String: Any]] { modelState["models"] as? [[String: Any]] ?? [] }
    var modelStateAvailable: Bool { modelState["models"] is [[String: Any]] }
    var activeJobs: Int {
        jobs.filter { ["queued", "running", "cancelling", "recovery_pending"].contains($0["status"] as? String ?? "") }.count
    }

    init() { Task { await launchAgent() } }

    private func launchAgent() async {
        let bundled = Bundle.main.resourceURL?.appendingPathComponent("runtime/bin/python3").path
        let development = ProcessInfo.processInfo.environment["MLX_SERVE_PYTHON"]
        guard let executable = [bundled, development].compactMap({ $0 }).first(where: {
            FileManager.default.isExecutableFile(atPath: $0)
        }) else {
            message = "실행 환경을 찾을 수 없습니다. 앱에 포함된 Python 런타임을 확인하세요."
            return
        }
        var random = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, random.count, &random) == errSecSuccess else {
            message = "로컬 관리 연결을 준비하지 못했습니다."
            return
        }
        secret = random.map { String(format: "%02x", $0) }.joined()
        let process = Process()
        let input = Pipe()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = ["-m", "mlx_serve.management_agent", "--parent", "\(ProcessInfo.processInfo.processIdentifier)"]
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "HF_TOKEN")
        environment.removeValue(forKey: "HUGGING_FACE_HUB_TOKEN")
        environment.removeValue(forKey: "PYTHONPATH")
        environment.removeValue(forKey: "PYTHONHOME")
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        // Python bytecode written inside a signed app bundle invalidates its seal.
        // This is inherited by the engine, workers, and preparation subprocesses.
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process.environment = environment
        process.standardInput = input
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            agent = process
            input.fileHandleForWriting.write(Data((secret + "\n").utf8))
            try? input.fileHandleForWriting.close()
            let line = try await Task.detached {
                var line = Data()
                while line.count < 4096 {
                    guard let byte = try output.fileHandleForReading.read(upToCount: 1),
                          !byte.isEmpty else { break }
                    line.append(byte)
                    if byte.last == 10 { break }
                }
                return line
            }.value
            guard let result = try JSONSerialization.jsonObject(with: line) as? [String: Any],
                  let port = result["port"] as? Int else {
                throw AppError("로컬 관리 구성요소를 시작하지 못했습니다.")
            }
            agentPort = port
            agentReady = true
            Task {
                while !Task.isCancelled && process.isRunning {
                    await refresh()
                    try? await Task.sleep(nanoseconds: 3_000_000_000)
                }
                agentReady = false
                message = "로컬 관리 구성요소가 종료됐습니다. 앱을 다시 실행하세요."
            }
        } catch {
            message = error.localizedDescription
        }
    }

    struct AppError: LocalizedError {
        let text: String
        init(_ text: String) { self.text = text }
        var errorDescription: String? { text }
    }

    func request(_ path: String, method: String = "GET", body: [String: Any]? = nil) async throws -> Any {
        guard let port = agentPort, let url = URL(string: "http://127.0.0.1:\(port)\(path)") else {
            throw AppError("관리 구성요소에 연결하고 있습니다.")
        }
        let timeout: TimeInterval = path == "/status" ? 6 :
            (method == "GET" && (path == "/models" || path.hasPrefix("/jobs")) ? 20 : 220)
        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.httpMethod = method
        request.setValue("Bearer \(secret)", forHTTPHeaderField: "Authorization")
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let (data, response) = try await localSession.data(for: request)
        let json = (try? JSONSerialization.jsonObject(with: data)) ?? [:]
        guard let response = response as? HTTPURLResponse, (200..<300).contains(response.statusCode) else {
            let detail = (json as? [String: Any])?["detail"]
            let message = detail.map { String(describing: $0) } ?? "관리 요청에 실패했습니다."
            throw AppError(localizedMessage(message))
        }
        return json
    }

    private func localizedMessage(_ message: String) -> String {
        if message.hasPrefix("Cleanup was incomplete; inspect ") {
            return "정리가 끝나지 않았습니다. 남은 파일 위치를 확인하세요: " +
                message.replacingOccurrences(of: "Cleanup was incomplete; inspect ", with: "")
        }
        let translations = [
            "Existing model files must be migrated before changing storage": "기존 모델 파일이 있어 저장 위치를 바꿀 수 없습니다. 모델 이동 기능이 준비될 때까지 현재 위치를 유지하세요.",
            "Existing registered models must be migrated before changing storage": "등록된 모델이 있어 저장 위치를 바꿀 수 없습니다. 모델 이동 기능이 준비될 때까지 현재 위치를 유지하세요.",
            "The current model storage is unavailable; reconnect it first": "현재 모델 저장소에 접근할 수 없습니다. 저장 장치를 다시 연결한 뒤 시도하세요.",
            "Model storage volume is unavailable; reconnect it first": "모델 저장 장치가 연결되지 않았습니다. 장치를 다시 연결한 뒤 서버를 시작하세요.",
            "Model storage is unavailable": "모델 저장소에 접근할 수 없습니다. 장치 연결과 권한을 확인하세요.",
            "The selected model is not installed": "선택한 모델의 파일이나 등록 정보를 찾을 수 없습니다. 모델 목록을 확인하세요.",
            "Default model settings could not be read": "기본 모델 설정을 읽을 수 없습니다. 저장소 상태를 확인하세요.",
            "Default model settings are invalid": "기본 모델 설정 형식을 확인할 수 없습니다. 설정 파일을 점검하세요.",
            "This model is not the current default": "기본 모델이 변경되었습니다. 현재 상태를 다시 확인하세요.",
            "Model registry could not be checked": "모델 등록 정보를 확인할 수 없어 정리를 중지했습니다. 저장소와 권한을 확인하세요.",
            "Original storage path is invalid": "원본 저장 경로를 안전하게 확인할 수 없습니다. 저장소 설정을 확인하세요.",
            "Cleanup targets changed; review them again": "정리 대상이 변경되었습니다. 목록을 다시 확인하세요.",
            "Finish or cancel model preparation before cleaning originals": "진행 중인 모델 준비를 마치거나 취소한 뒤 정리하세요.",
            "The existing model registry could not be checked": "기존 모델 목록을 확인할 수 없습니다. 저장 장치와 파일 권한을 확인하세요.",
            "Finish or cancel model preparation before changing storage": "진행 중인 모델 준비를 마치거나 취소한 뒤 저장 위치를 바꾸세요.",
            "Stop the server before changing model storage": "저장 위치를 바꾸기 전에 서버를 중지하세요.",
            "External server settings are read-only": "기존 서버의 설정은 이 앱에서 변경할 수 없습니다.",
            "The configured port is already in use": "설정된 포트를 다른 서비스가 사용 중입니다. 포트나 해당 서비스를 확인하세요.",
            "The app does not own a running server": "이 앱이 시작한 서버가 아닙니다. 기존 서비스에서 서버를 관리하세요.",
            "Cancel or finish active jobs first": "진행 중인 작업을 마치거나 취소한 뒤 서버를 중지하세요.",
            "Finish or cancel active jobs first": "진행 중인 작업을 마치거나 취소한 뒤 다시 시도하세요.",
            "Start or connect to the server first": "먼저 서버를 시작하거나 기존 서버에 연결하세요.",
            "Invalid request. Check the entered values.": "입력한 값을 확인한 뒤 다시 시도하세요.",
            "Server is unavailable": "서버에 연결할 수 없습니다. 서버 상태와 포트를 확인하세요.",
            "Worker is unavailable": "추론 워커에 연결할 수 없습니다. 진단 화면에서 워커 상태를 확인하세요."
        ]
        return translations[message] ?? message
    }

    func refresh() async {
        do {
            server = try await request("/status") as? [String: Any] ?? [:]
            jobs = try await request("/jobs") as? [[String: Any]] ?? []
            let events = notificationPolicy.observe(
                serverState: server["state"] as? String ?? "unknown", ownership: ownership, jobs: jobs
            )
            for event in events { Task { await deliverNotification(event) } }
            if ["connected", "degraded"].contains(state) {
                do {
                    modelState = try await request("/models") as? [String: Any] ?? [:]
                    modelError = nil
                } catch {
                    modelState = [:]
                    modelError = "이 서버의 모델 상태를 확인할 수 없습니다: \(error.localizedDescription)"
                }
            } else {
                modelState = [:]
                modelError = nil
            }
            if let id = selectedJob {
                jobDetail = (try? await request("/jobs/\(id)")) as? [String: Any] ?? [:]
            }
            for job in jobs {
                guard let id = job["id"] as? String, autoLoad.contains(id) else { continue }
                if job["status"] as? String == "succeeded" {
                    autoLoad.remove(id)
                    if let result = job["result"] as? [String: Any], let name = result["name"] as? String {
                        await modelCommand("load", name: name, type: "decision")
                    }
                } else if ["failed", "cancelled", "interrupted"].contains(job["status"] as? String ?? "") {
                    autoLoad.remove(id)
                }
            }
            await evaluateAutomaticStart()
        } catch {
            server = ["state": "unknown"]
            modelState = [:]
            modelError = "관리 연결을 확인할 수 없습니다. 서버 및 앱 상태를 다시 확인하세요."
            if agent?.isRunning == false { message = "관리 구성요소가 종료됐습니다. 앱을 다시 실행하세요." }
        }
    }

    private func evaluateAutomaticStart() async {
        guard !busy else { return }
        let autoStart = UserDefaults.standard.bool(forKey: "automaticStartServer")
        let autoRecover = UserDefaults.standard.bool(forKey: "automaticRecoverServer")
        let decision = serverStartPolicy.evaluate(
            state: state, ownership: ownership, autoStart: autoStart, autoRecover: autoRecover,
            now: ProcessInfo.processInfo.systemUptime
        )
        if state == "connected" { automaticStartNote = nil }
        if state == "failed" && autoRecover && serverStartPolicy.attempts >= 3 {
            automaticStartNote = "자동 복구를 세 번 시도했습니다. 원인을 확인한 뒤 직접 다시 시작하세요."
        }
        guard let decision else { return }
        switch decision {
        case .startOnLaunch:
            automaticStartNote = "앱 설정에 따라 서버를 시작합니다."
        case .recover(let attempt):
            automaticStartNote = "서버 자동 복구를 시도합니다 (\(attempt)/3)."
        }
        await action { _ = try await self.request("/server/start", method: "POST") }
    }

    func startServer() async {
        serverStartPolicy.manualStartRequested()
        automaticStartNote = nil
        await action { _ = try await self.request("/server/start", method: "POST") }
    }

    func stopServer() async {
        serverStartPolicy.manualStopRequested()
        automaticStartNote = nil
        await action { _ = try await self.request("/server/stop", method: "POST") }
        // A concurrent refresh while stop was in progress must not re-arm it.
        if state == "stopped" { serverStartPolicy.manualStopRequested() }
    }

    func modelCommand(_ operation: String, name: String, type: String) async {
        await action {
            _ = try await self.request("/models/\(operation)", method: "POST", body: ["name": name, "type": type])
        }
    }

    func startJob(repo: String, name: String, revision: String, keepOriginal: Bool, loadAfter: Bool) async {
        await action {
            var body: [String: Any] = ["repo": repo, "keep_original": keepOriginal]
            if !name.isEmpty { body["name"] = name }
            if !revision.isEmpty { body["revision"] = revision }
            if let token = HFKeychain.read(), !token.isEmpty { body["token"] = token }
            let result = try await self.request("/jobs", method: "POST", body: body) as? [String: Any] ?? [:]
            if let id = result["id"] as? String {
                self.selectedJob = id
                if loadAfter { self.autoLoad.insert(id) }
            }
        }
    }

    func cancelJob(_ id: String) async {
        await action { _ = try await self.request("/jobs/\(id)/cancel", method: "POST") }
    }

    func reloadSettings() async {
        do {
            settings = try await request("/settings") as? [String: Any] ?? [:]
            runtime = try await request("/runtime") as? [String: Any] ?? [:]
        } catch { message = error.localizedDescription }
    }

    func saveSettings(_ values: [String: Any]) async {
        await action {
            self.settings = try await self.request("/settings", method: "PUT", body: ["values": values]) as? [String: Any] ?? [:]
        }
    }

    func field(_ name: String) -> [String: Any] {
        let fields = settings["fields"] as? [String: [String: Any]] ?? [:]
        return fields[name] ?? [:]
    }

    func action(_ body: () async throws -> Void) async {
        busy = true
        defer { busy = false }
        do { try await body(); await refresh() }
        catch { message = error.localizedDescription }
    }

    func exportDiagnostics() async {
        do {
            let data = try JSONSerialization.data(withJSONObject: try await request("/diagnostics"), options: [.prettyPrinted, .sortedKeys])
            let panel = NSSavePanel()
            panel.nameFieldStringValue = "mlx-serve-diagnostics.json"
            panel.message = "서버 상태와 작업 이력만 포함합니다. 요청 본문, 인증정보와 로그는 제외됩니다."
            if panel.runModal() == .OK, let url = panel.url { try data.write(to: url, options: .atomic) }
        } catch { message = error.localizedDescription }
    }
}

func pretty(_ value: Any) -> String {
    guard JSONSerialization.isValidJSONObject(value),
          let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]),
          let text = String(data: data, encoding: .utf8) else { return String(describing: value) }
    return text
}

func copyText(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
}

func readableState(_ state: String) -> String {
    let translations = ["connecting": "연결 중", "connected": "정상", "degraded": "부분 장애", "starting": "시작 중",
     "stopped": "중지됨", "port_conflict": "포트 사용 중", "queued": "대기 중", "running": "진행 중",
     "cancelling": "취소 중", "cancelled": "취소됨", "succeeded": "준비 완료", "failed": "실패",
     "recovery_pending": "정리 확인 필요",
     "interrupted": "중단됨", "loaded": "적재됨", "not_loaded": "미적재", "unknown": "확인 불가",
     "idle": "유휴", "installed": "설치됨", "downloading": "다운로드", "checking": "호환성 확인",
     "converting": "MLX 변환", "validating": "검증", "resolving": "버전 확인",
     "registering": "모델 등록", "resources": "임시 파일 측정", "complete": "완료"]
    return translations[state] ?? state
}
