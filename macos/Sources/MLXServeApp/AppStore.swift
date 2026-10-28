import AppKit
import CryptoKit
import Darwin
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
    @Published var storageMigration: [String: Any] = [:]
    @Published var selectedJob: String?
    @Published var message: String?
    @Published var agentReady = false
    @Published var modelError: String?
    @Published var busy = false
    @Published var automaticStartNote: String?
    @Published var backgroundEnabled = false

    private var agent: Process?
    private var agentPort: Int?
    private var secret = ""
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
    var migratingStorage: Bool { storageMigration["status"] as? String == "running" }
    var activeJobs: Int {
        jobs.filter { ["queued", "running", "cancelling", "recovery_pending"].contains($0["status"] as? String ?? "") }.count
    }

    init() { Task { await launchAgent() } }

    private func launchAgent() async {
        do {
            if try await reconnectBackgroundAgent() {
                startPolling(nil)
                return
            }
        } catch {
            message = error.localizedDescription
            return
        }
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
            var confirmed = false
            for _ in 0..<12 where process.isRunning {
                if (try? await request("/status")) != nil {
                    confirmed = true
                    break
                }
                try? await Task.sleep(nanoseconds: 250_000_000)
            }
            guard confirmed else {
                throw AppError("로컬 관리 구성요소를 확인하지 못했습니다. 기존 관리 프로세스와 작업을 확인하세요.")
            }
            agentReady = true
            startPolling(process)
            if UserDefaults.standard.bool(forKey: "backgroundServer") {
                await setBackground(true)
            }
        } catch {
            message = error.localizedDescription
        }
    }

    private struct AgentRecord: Decodable {
        let version: Int
        let pid: Int32
        let port: Int
        let session: String
        let keyID: String

        enum CodingKeys: String, CodingKey {
            case version, pid, port, session
            case keyID = "key_id"
        }
    }

    private var agentKeyID: String {
        let input = Data("mlx-serve-agent-key-id-v1:".utf8) + Data(secret.utf8)
        return SHA256.hash(data: input).map { String(format: "%02x", $0) }.joined()
    }

    private func reconnectBackgroundAgent() async throws -> Bool {
        let home = ProcessInfo.processInfo.environment["HOME"] ?? NSHomeDirectory()
        let file = URL(fileURLWithPath: home).appendingPathComponent(".mlx-serve/app-agent.json")
        guard FileManager.default.fileExists(atPath: file.path) else { return false }
        let values = try file.resourceValues(forKeys: [.isSymbolicLinkKey, .fileSizeKey])
        guard values.isSymbolicLink != true, (values.fileSize ?? 4097) <= 4096,
              let record = try? JSONDecoder().decode(AgentRecord.self, from: Data(contentsOf: file)),
              record.version == 1, record.pid > 1, (1...65535).contains(record.port),
              !record.session.isEmpty, record.keyID.count == 64,
              record.keyID.allSatisfy({ $0.isHexDigit }) else {
            throw AppError("백그라운드 연결 정보를 확인할 수 없습니다. 저장소와 실행 중인 서비스를 확인하세요.")
        }
        errno = 0
        if kill(record.pid, 0) != 0 && errno == ESRCH { return false }
        guard let saved = try AgentKeychain.read(record.keyID) else {
            throw AppError("백그라운드 서버의 Keychain 연결 정보를 찾을 수 없습니다.")
        }
        secret = saved
        guard agentKeyID == record.keyID else {
            secret = ""
            throw AppError("백그라운드 서버의 Keychain 연결 정보를 확인하지 못했습니다.")
        }
        agentPort = record.port
        do {
            _ = try await request("/background/attach", method: "POST", body: [
                "pid": Int(ProcessInfo.processInfo.processIdentifier)
            ])
        } catch {
            agentPort = nil
            secret = ""
            throw AppError("실행 중인 백그라운드 서버를 안전하게 확인하지 못했습니다. 기존 서비스와 Keychain 상태를 확인하세요.")
        }
        backgroundEnabled = true
        agentReady = true
        UserDefaults.standard.set(true, forKey: "backgroundServer")
        return true
    }

    private func startPolling(_ process: Process?) {
        Task {
            while !Task.isCancelled {
                if let process, !process.isRunning { break }
                await refresh()
                try? await Task.sleep(nanoseconds: 3_000_000_000)
            }
            if process != nil {
                agentReady = false
                message = "로컬 관리 구성요소가 종료됐습니다. 앱을 다시 실행하세요."
            }
        }
    }

    struct AppError: LocalizedError {
        let text: String
        init(_ text: String) { self.text = text }
        var errorDescription: String? { text }
    }

    func request(_ path: String, method: String = "GET", body: [String: Any]? = nil) async throws -> Any {
        guard let port = agentPort, let url = URL(string: "http://127.0.0.1:\(port)/secure") else {
            throw AppError("관리 구성요소에 연결하고 있습니다.")
        }
        let timeout: TimeInterval = path == "/status" ? 6 :
            (method == "GET" && (path == "/models" || path.hasPrefix("/jobs")) ? 20 : 220)
        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        var envelope: [String: Any] = [
            "method": method, "path": path, "timestamp": Date().timeIntervalSince1970
        ]
        if let body { envelope["body"] = body }
        let material = Data("mlx-serve-agent-aes-key-v1:".utf8) + Data(secret.utf8)
        let key = SymmetricKey(data: Data(SHA256.hash(data: material)))
        let additionalData = Data("mlx-serve-agent-v1".utf8)
        let plaintext = try JSONSerialization.data(withJSONObject: envelope)
        let box = try AES.GCM.seal(plaintext, using: key, authenticating: additionalData)
        guard let combined = box.combined else { throw AppError("관리 연결을 암호화하지 못했습니다.") }
        let nonce = Data(combined.prefix(12))
        request.httpBody = try JSONSerialization.data(withJSONObject: ["payload": combined.base64EncodedString()])
        let (data, response) = try await localSession.data(for: request)
        guard let response = response as? HTTPURLResponse, response.statusCode == 200,
              let outer = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let encoded = outer["payload"] as? String, let sealed = Data(base64Encoded: encoded),
              let responseBox = try? AES.GCM.SealedBox(combined: sealed),
              let decrypted = try? AES.GCM.open(responseBox, using: key, authenticating: additionalData),
              let result = try? JSONSerialization.jsonObject(with: decrypted) as? [String: Any],
              let requestNonce = result["request_nonce"] as? String,
              Data(base64Encoded: requestNonce) == nonce, let status = result["status"] as? Int else {
            throw AppError("관리 구성요소의 보안 연결을 확인하지 못했습니다. 앱을 다시 실행하세요.")
        }
        let json = result["body"] ?? [:]
        guard (200..<300).contains(status) else {
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
            "Hugging Face cache is unavailable": "Hugging Face 캐시에 접근할 수 없습니다. 저장 장치와 권한을 확인하세요.",
            "Existing model files must be migrated before changing storage": "기존 모델이 있습니다. ‘기존 모델 복사 및 전환’을 이용하세요.",
            "Existing registered models must be migrated before changing storage": "등록된 모델이 있습니다. ‘기존 모델 복사 및 전환’을 이용하세요.",
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
            "Model storage migration is in progress": "모델 저장소를 복사하는 중입니다. 완료될 때까지 기다리세요.",
            "Model storage is overridden by the environment": "환경 변수 또는 .env 파일이 저장 위치를 지정하고 있습니다. 해당 설정을 제거한 뒤 다시 시도하세요.",
            "Model storage settings changed during migration": "복사하는 동안 저장 위치 설정이 변경되어 전환을 중지했습니다. 기존 파일과 복사본은 보존됩니다.",
            "Model storage changed during migration": "복사하는 동안 기존 저장소가 변경되어 전환을 중지했습니다. 기존 파일과 복사본을 확인하세요.",
            "Model storage is being modified by another process": "다른 프로세스가 모델 저장소를 사용 중입니다. 다른 서버와 모델 작업을 종료하세요.",
            "The destination parent directory already exists": "새 저장 위치의 상위 폴더가 이미 있습니다. 아직 없는 새 폴더 안의 경로를 지정하세요.",
            "The destination parent must be a new folder inside a writable folder": "쓰기 가능한 기존 폴더 안에 새 폴더를 지정하세요.",
            "Choose a destination outside the existing model storage": "기존 저장소 바깥에 새 위치를 지정하세요.",
            "Choose a separate model directory": "모델 파일을 저장할 별도 폴더 이름을 지정하세요.",
            "Model directory must be an absolute path": "모델 저장 위치를 절대 경로로 입력하세요.",
            "Model storage contains a link or another mounted volume": "저장소에 심볼릭 링크 또는 중첩된 저장 장치가 있어 자동 복사할 수 없습니다.",
            "Model storage contains an unsupported file": "저장소에 자동 복사를 지원하지 않는 파일이 있습니다.",
            "An unfinished preparation directory needs review": "완료되지 않은 준비 작업 폴더가 있습니다. 작업 이력과 저장소를 확인하세요.",
            "Not enough free disk space for the model copy": "새 저장소의 여유 공간이 부족합니다. 공간을 확보한 뒤 다시 시도하세요.",
            "Copied model files failed verification": "복사본 검증에 실패해 저장 위치를 바꾸지 않았습니다. 기존 파일과 복사본을 확인하세요.",
            "Copied model files changed during migration": "검증 중 복사본이 변경되어 저장 위치를 바꾸지 않았습니다. 기존 파일과 복사본을 확인하세요.",
            "Model storage copy was interrupted": "모델 복사가 중단되어 기존 저장 위치를 유지했습니다. 복사 대상 폴더를 확인하세요.",
            "Model storage copy failed": "모델 복사에 실패했습니다. 기존 파일과 복사 대상 폴더를 확인하세요.",
            "Background agent is unavailable": "백그라운드 관리 구성요소를 사용할 수 없습니다. 앱을 다시 실행하세요.",
            "Another background agent is already running": "다른 백그라운드 관리 구성요소가 실행 중입니다. 기존 앱을 확인하세요.",
            "Another background agent may still be running": "기존 백그라운드 관리 구성요소가 남아 있을 수 있습니다. 실행 중인 서비스를 확인하세요.",
            "Background agent record is invalid": "백그라운드 연결 정보가 올바르지 않습니다. 저장소와 실행 중인 서비스를 확인하세요.",
            "Background agent lock is invalid": "백그라운드 연결 잠금 파일을 확인할 수 없습니다. 저장소 권한을 확인하세요.",
            "The application process could not be verified": "앱 프로세스를 확인하지 못해 백그라운드 연결을 중지했습니다.",
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
            backgroundEnabled = server["background_enabled"] as? Bool ?? false
            jobs = try await request("/jobs") as? [[String: Any]] ?? []
            let previousMigration = storageMigration["status"] as? String
            storageMigration = try await request("/storage/migration") as? [String: Any] ?? [:]
            let migrationStatus = storageMigration["status"] as? String
            if previousMigration == "running" && migrationStatus == "succeeded" {
                await reloadSettings()
                message = "모델 복사와 저장 위치 전환을 완료했습니다. 기존 파일은 이전 위치에 그대로 남아 있습니다."
            } else if previousMigration == "running" && migrationStatus == "failed" {
                let reason = storageMigration["error"] as? String ?? "Model storage copy failed"
                message = localizedMessage(reason) + " 기존 파일은 보존되며, 복사 대상 폴더가 남을 수 있습니다."
            }
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
            await evaluateAutomaticStart()
        } catch {
            server = ["state": "unknown"]
            modelState = [:]
            modelError = "관리 연결을 확인할 수 없습니다. 서버 및 앱 상태를 다시 확인하세요."
            if agent?.isRunning == false { message = "관리 구성요소가 종료됐습니다. 앱을 다시 실행하세요." }
        }
    }

    private func evaluateAutomaticStart() async {
        guard !busy && !migratingStorage else { return }
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

    func setBackground(_ enabled: Bool) async {
        guard agentReady, !busy else { return }
        busy = true
        defer { busy = false }
        do {
            if enabled { try AgentKeychain.save(secret, keyID: agentKeyID) }
            _ = try await request("/background", method: "POST", body: ["enabled": enabled])
            backgroundEnabled = enabled
            UserDefaults.standard.set(enabled, forKey: "backgroundServer")
            if !enabled { try AgentKeychain.remove(agentKeyID) }
            await refresh()
        } catch { message = error.localizedDescription }
    }

    func modelCommand(_ operation: String, name: String, type: String) async {
        await action {
            _ = try await self.request("/models/\(operation)", method: "POST", body: ["name": name, "type": type])
        }
    }

    func startJob(repo: String, name: String, revision: String, keepOriginal: Bool, loadAfter: Bool) async {
        await action {
            var body: [String: Any] = [
                "repo": repo, "keep_original": keepOriginal, "load_after": loadAfter
            ]
            if !name.isEmpty { body["name"] = name }
            if !revision.isEmpty { body["revision"] = revision }
            if let token = HFKeychain.read(), !token.isEmpty { body["token"] = token }
            let result = try await self.request("/jobs", method: "POST", body: body) as? [String: Any] ?? [:]
            if let id = result["id"] as? String {
                self.selectedJob = id
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

    func previewStorageMigration(_ destination: String) async throws -> [String: Any] {
        try await request("/storage/migration/preview", method: "POST", body: ["destination": destination])
            as? [String: Any] ?? [:]
    }

    func startStorageMigration(_ destination: String) async {
        await action {
            self.storageMigration = try await self.request(
                "/storage/migration", method: "POST", body: ["destination": destination]
            ) as? [String: Any] ?? [:]
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
     "registering": "모델 등록", "resources": "임시 파일 측정", "complete": "완료",
     "loading": "모델 로드", "load_succeeded": "자동 로드 완료", "load_failed": "자동 로드 실패"]
    return translations[state] ?? state
}
