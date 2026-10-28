import ServiceManagement
import SwiftUI
import UserNotifications

struct JobsView: View {
    @EnvironmentObject var store: AppStore
    var selected: [String: Any]? { store.jobs.first { $0["id"] as? String == store.selectedJob } }
    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            VStack(alignment: .leading) {
                PageHeader(title: "작업", subtitle: "준비 이력과 실제 취소 상태를 확인합니다.")
                List(selection: $store.selectedJob) {
                    ForEach(store.jobs.indices, id: \.self) { index in
                        let job = store.jobs[index]
                        let params = job["params"] as? [String: Any] ?? [:]
                        let id = job["id"] as? String ?? ""
                        VStack(alignment: .leading, spacing: 4) {
                            Text(params["repo"] as? String ?? "모델 준비").fontWeight(.medium)
                            Text("\(readableState(job["status"] as? String ?? "unknown")) · \(readableState(job["stage"] as? String ?? "unknown"))")
                                .font(.caption).foregroundStyle(.secondary)
                        }.tag(id)
                    }
                }.frame(minHeight: 200, maxHeight: .infinity)
            }.padding(20).frame(width: 320).frame(maxHeight: .infinity, alignment: .topLeading)
            Divider()
            Group {
              if let job = selected {
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            Text("작업 상세").font(.title2.bold()); Spacer()
                            StatusBadge(value: job["status"] as? String ?? "unknown")
                        }
                        LabeledContent("작업 ID", value: job["id"] as? String ?? "")
                        LabeledContent("현재 단계", value: readableState(job["stage"] as? String ?? "unknown"))
                        LabeledContent("시작", value: job["created_at"] as? String ?? "")
                        if ["queued", "running", "cancelling", "recovery_pending"].contains(job["status"] as? String ?? ""),
                           let timestamp = job["created_at"] as? String,
                           let started = jobDate(timestamp) {
                            LabeledContent("경과 시간") { Text(started, style: .timer) }
                        }
                        if let error = job["error"] as? String {
                            Text(error).foregroundStyle(.orange).textSelection(.enabled)
                            Text(jobGuidance(job["error_category"] as? String ?? ""))
                                .font(.callout).foregroundStyle(.secondary)
                        }
                        let id = job["id"] as? String ?? ""
                        let status = job["status"] as? String ?? ""
                        if ["queued", "running", "cancelling", "recovery_pending"].contains(status) {
                            Button(status == "recovery_pending" ? "정리 다시 확인" : "작업 취소") { Task { await store.cancelJob(id) } }
                                .disabled(store.busy || status == "cancelling")
                            Text("취소는 준비 프로세스와 임시 파일 정리가 끝난 뒤 확정됩니다.")
                                .font(.caption).foregroundStyle(.secondary)
                        } else if ["failed", "cancelled", "interrupted"].contains(status) {
                            Button("다시 시도") {
                                let params = job["params"] as? [String: Any] ?? [:]
                                Task {
                                    await store.startJob(
                                        repo: params["repo"] as? String ?? "",
                                        name: params["name"] as? String ?? "",
                                        revision: params["revision"] as? String ?? "",
                                        keepOriginal: params["keep_original"] as? Bool ?? false,
                                        loadAfter: false
                                    )
                                }
                            }.disabled(store.busy)
                            Text("재시도할 때 기존 검증 산출물의 재사용 가능 여부를 서버가 확인합니다.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Divider()
                        Text("단계와 기록").font(.headline)
                        let events = store.jobDetail["events"] as? [[String: Any]] ?? []
                        if events.isEmpty { Text("기록을 불러오는 중입니다.").foregroundStyle(.secondary) }
                        ForEach(events.indices, id: \.self) { index in
                            let event = events[index]
                            VStack(alignment: .leading, spacing: 3) {
                                Text(readableState(event["status"] as? String ?? ""))
                                if let detail = event["detail"] as? String { Text(detail).foregroundStyle(.secondary) }
                                if let estimate = event["estimated_disk_bytes"] as? Int64,
                                   let available = event["free_disk_bytes"] as? Int64 {
                                    Text("준비 공간 추정 \(ByteCountFormatter.string(fromByteCount: estimate, countStyle: .file)) · 확인 당시 여유 \(ByteCountFormatter.string(fromByteCount: available, countStyle: .file))")
                                        .foregroundStyle(.secondary)
                                    if event["estimate_basis"] as? String == "partial_repository_metadata" {
                                        Text("저장소의 파일 크기 정보가 일부 없어 실제 필요 공간이 더 클 수 있습니다.")
                                            .foregroundStyle(.orange)
                                    } else if event["estimate_basis"] as? String == "repository_metadata" {
                                        Text("저장소 파일 크기를 기준으로 변환 및 임시 공간을 더한 추정치입니다.")
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                if let size = event["temporary_file_bytes"] as? Int64 {
                                    Text("작업 임시 파일 크기 \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file)) (측정값)")
                                        .foregroundStyle(.secondary)
                                }
                                if let message = event["message"] as? String { Text(message).foregroundStyle(.orange) }
                                Text(event["time"] as? String ?? "").font(.caption2).foregroundStyle(.secondary)
                            }.font(.callout)
                        }
                    }.padding(24).frame(maxWidth: .infinity, alignment: .leading)
                }
              } else {
                ContentUnavailableView("작업 선택", systemImage: "clock.arrow.circlepath", description: Text("목록에서 작업을 선택하세요."))
              }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private func jobDate(_ timestamp: String) -> Date? {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter.date(from: timestamp)
}

private func jobGuidance(_ category: String) -> String {
    switch category {
    case "authentication": return "Hugging Face 토큰을 확인하거나 다시 저장한 뒤 시도하세요."
    case "access_approval": return "Hugging Face에서 모델 접근 승인과 토큰 권한을 확인하세요."
    case "storage_permission": return "모델 저장 위치의 권한과 장치 연결 상태를 확인하세요."
    case "disk": return "저장 공간을 확보하고 모델 저장 장치의 연결을 확인하세요."
    case "memory": return "다른 모델을 언로드하거나 메모리를 확보한 뒤 시도하세요."
    case "network": return "네트워크 연결을 확인한 뒤 다시 시도하세요."
    case "missing_file": return "저장소 ID, revision과 필요한 모델 파일을 확인하세요."
    case "timeout": return "네트워크와 실행 환경을 확인한 뒤 다시 시도하세요."
    case "unsupported": return "지원되는 d1 원본 또는 공식 MLX FP16 모델을 선택하세요."
    case "validation": return "원본과 revision을 확인하고 검증된 파일을 다시 준비하세요."
    case "conversion": return "런타임 버전과 사용 가능한 자원을 확인한 뒤 다시 시도하세요."
    default: return "작업 기록에서 오류를 확인하고 원인을 수정한 뒤 다시 시도하세요."
    }
}

struct DiagnosticsView: View {
    @EnvironmentObject var store: AppStore
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                PageHeader(title: "진단", subtitle: "게이트웨이와 워커의 상태를 따로 확인합니다.")
                Button("진단 자료 내보내기…") { Task { await store.exportDiagnostics() } }
                Text("내보내기에는 서버 상태와 작업 이력이 포함됩니다. 요청 본문, 인증정보와 서버 로그는 제외됩니다.")
                    .font(.callout).foregroundStyle(.secondary)
                GroupBox("서버") { Text(pretty(store.server)).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(8) }
                GroupBox("게이트웨이 및 워커") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("프로세스 RSS와 MLX 활성·캐시 메모리는 서로 겹칠 수 있어 합산하지 않습니다.").font(.caption).foregroundStyle(.secondary)
                        Text(pretty(["gateway": store.modelState["gateway"] ?? [:], "workers": store.modelState["workers"] ?? [:]]))
                            .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                }
            }.padding(24)
        }
    }
}

struct PreferencesView: View {
    @EnvironmentObject var store: AppStore
    @State private var port = "8000"
    @State private var host = "127.0.0.1"
    @State private var modelsDir = ""
    @State private var preload = ""
    @State private var ttl = "1800"
    @State private var queue = "8"
    @State private var concurrency = "1"
    @State private var memoryGuard = true
    @State private var preset = "custom"
    @State private var embeddingCache = "3"
    @State private var rerankerCache = "2"
    @State private var decisionCache = "1"
    @State private var embeddingBatch = "32"
    @State private var rerankerDocuments = "4"
    @State private var rerankerTokens = "2048"
    @State private var logLevel = "INFO"
    @State private var logBodies = false
    @State private var token = ""
    @State private var loginStatus = "확인 중"
    @State private var notificationStatus = "확인 중"
    @State private var reviewingOriginals = false
    @AppStorage("automaticStartServer") private var automaticStartServer = false
    @AppStorage("automaticRecoverServer") private var automaticRecoverServer = false
    @AppStorage("jobNotifications") private var jobNotifications = false

    var body: some View {
        TabView {
            general.tabItem { Label("일반", systemImage: "gear") }
            server.tabItem { Label("서버", systemImage: "network") }
            storage.tabItem { Label("모델", systemImage: "externaldrive") }
            performance.tabItem { Label("성능", systemImage: "speedometer") }
            runtime.tabItem { Label("런타임", systemImage: "cpu") }
            logs.tabItem { Label("진단", systemImage: "stethoscope") }
        }
        .padding(20).frame(width: 690, height: 510)
        .task {
            await store.reloadSettings(); populate(); updateLoginStatus()
            await updateNotificationStatus()
        }
        .sheet(isPresented: $reviewingOriginals) { OriginalCleanupView().environmentObject(store) }
        .alert("알림", isPresented: Binding(
            get: { store.message != nil }, set: { if !$0 { store.message = nil } }
        )) { Button("확인") { store.message = nil } } message: { Text(store.message ?? "") }
    }

    var general: some View {
        Form {
            Section("시작과 종료") {
                LabeledContent("로그인 항목", value: loginStatus)
                HStack {
                    Button("로그인 시 시작 등록") { login(register: true) }
                    Button("등록 해제") { login(register: false) }
                    Button("시스템 설정 열기") { SMAppService.openSystemSettingsLoginItems() }
                }
                Toggle("앱을 열 때 서버 자동 시작", isOn: $automaticStartServer)
                Toggle("앱이 시작한 서버 종료 시 자동 복구", isOn: $automaticRecoverServer)
                Text("자동 시작은 서버가 중지된 경우에만 실행합니다. 자동 복구는 앱 실행 중 최대 세 번 시도하며, 직접 중지한 서버나 기존 서버에는 적용하지 않습니다.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("서버 중단·모델 준비 완료 및 실패 알림", isOn: Binding(
                    get: { jobNotifications },
                    set: { enabled in
                        jobNotifications = enabled
                        Task {
                            if enabled {
                                _ = try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert])
                            }
                            await updateNotificationStatus()
                        }
                    }
                ))
                LabeledContent("시스템 알림 상태", value: notificationStatus)
                Text("알림에는 입력 내용이나 인증정보를 포함하지 않습니다. 앱이 실행 중일 때 발생한 변경을 알립니다.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("창을 닫아도 앱과 서버는 유지됩니다. 앱을 종료하면 앱이 시작한 작업과 서버를 정리합니다.")
                    .foregroundStyle(.secondary)
            }
        }.formStyle(.grouped)
    }

    var server: some View {
        Form {
            Section("서버 연결") {
                Picker("접근 범위", selection: $host) {
                    Text("이 Mac만").tag("127.0.0.1")
                    Text("이 Mac만 (localhost)").tag("localhost")
                    Text("로컬 네트워크").tag("0.0.0.0")
                    Text("이 Mac만 (IPv6)").tag("::1")
                    Text("로컬 네트워크 (IPv6)").tag("::")
                }
                TextField("포트", text: $port)
                fieldState("host")
                fieldState("port")
                Button("연결 설정 저장") {
                    guard let value = Int(port) else { store.message = "포트는 숫자로 입력하세요."; return }
                    Task { await store.saveSettings(["host": host, "port": value]) }
                }.disabled(store.ownership == "external")
                Text("앱이 시작한 이 버전의 서버는 LAN을 켜도 관리 기능을 로컬로 제한합니다.")
                    .foregroundStyle(.secondary)
            }
            Section("대기열") {
                TextField("모델당 대기 요청 수", text: $queue)
                fieldState("inference_max_queue_per_model")
                Button("대기열 저장") {
                    guard let value = Int(queue) else { store.message = "대기열 크기를 숫자로 입력하세요."; return }
                    Task { await store.saveSettings(["inference_max_queue_per_model": value]) }
                }.disabled(store.ownership == "external")
            }
        }.formStyle(.grouped)
    }

    var storage: some View {
        Form {
            Section("저장소") {
                TextField("새 모델 저장 위치", text: $modelsDir)
                fieldState("models_dir")
                if let disk = store.modelState["disk"] as? [String: Any],
                   let available = disk["free_bytes"] as? Int64,
                   let total = disk["total_bytes"] as? Int64 {
                    LabeledContent("현재 저장 장치", value: "여유 \(ByteCountFormatter.string(fromByteCount: available, countStyle: .file)) / 전체 \(ByteCountFormatter.string(fromByteCount: total, countStyle: .file))")
                } else {
                    LabeledContent("현재 저장 장치", value: "서버 연결 후 확인")
                }
                Text("경로 변경 전 서버를 중지하세요. 기존 모델이나 작업이 있으면 변경할 수 없으며, 모델 이동 기능은 준비 중입니다.")
                    .foregroundStyle(.secondary)
                Button("저장 위치 저장") { Task { await store.saveSettings(["models_dir": modelsDir]) } }
                    .disabled(store.ownership == "external")
                Button("참조되지 않는 원본 정리…") { reviewingOriginals = true }
                    .disabled(!store.modelStateAvailable || store.activeJobs > 0)
                Text("다른 모델이 참조하는 원본과 변환된 모델은 정리하지 않습니다.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Hugging Face") {
                SecureField("토큰", text: $token)
                HStack {
                    Button("Keychain에 저장") {
                        do { try HFKeychain.save(token); token = ""; store.message = "Keychain에 저장했습니다." }
                        catch { store.message = error.localizedDescription }
                    }
                    Button("토큰 제거") {
                        do { try HFKeychain.save(""); token = ""; store.message = "Keychain에서 제거했습니다." }
                        catch { store.message = error.localizedDescription }
                    }
                }
                Text("토큰은 설정 파일이나 작업 이력에 저장하지 않습니다.").foregroundStyle(.secondary)
            }
        }.formStyle(.grouped)
    }

    var performance: some View {
        Form {
            Section("모델과 메모리") {
                Picker("입력 프리셋", selection: $preset) {
                    Text("사용자 지정").tag("custom")
                    Text("권장 기본값").tag("recommended")
                    Text("메모리 절약").tag("conserve")
                }
                .onChange(of: preset) { _, selection in applyPreset(selection) }
                if preset == "recommended" {
                    Text("캐시: 임베딩 3 / 리랭커 2 / 결정 1개, 유휴 1800초, 동시 요청 1개.")
                        .font(.caption).foregroundStyle(.secondary)
                } else if preset == "conserve" {
                    Text("캐시: 유형별 1개, 유휴 300초, 동시 요청 1개, 배치 크기 축소.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Text("프리셋은 입력값만 채웁니다. 값을 확인한 후 저장하세요.")
                    .font(.caption).foregroundStyle(.secondary)
                TextField("사전 로드 모델 ID (쉼표로 구분)", text: $preload)
                TextField("유휴 캐시 시간 (초)", text: $ttl)
                TextField("모델당 동시 요청 수", text: $concurrency)
                Toggle("메모리 보호 사용", isOn: $memoryGuard)
                fieldState("preload_models")
                fieldState("cache_ttl_seconds")
                fieldState("inference_max_concurrency_per_model")
                fieldState("memory_guard_enabled")
                DisclosureGroup("캐시와 배치 고급 설정") {
                    TextField("임베딩 캐시 모델 수", text: $embeddingCache)
                    fieldState("cache_max_embedding_models")
                    TextField("리랭커 캐시 모델 수", text: $rerankerCache)
                    fieldState("cache_max_reranker_models")
                    TextField("결정 모델 캐시 수", text: $decisionCache)
                    fieldState("cache_max_decision_models")
                    TextField("임베딩 배치 텍스트 수", text: $embeddingBatch)
                    fieldState("embedding_batch_max_texts")
                    TextField("리랭커 배치 문서 수", text: $rerankerDocuments)
                    fieldState("rerank_batch_max_documents")
                    TextField("리랭커 배치 토큰 예산", text: $rerankerTokens)
                    fieldState("rerank_batch_max_tokens")
                }
                Button("성능 설정 저장") {
                    guard let seconds = Int(ttl), let limit = Int(concurrency),
                          let embedding = Int(embeddingCache), let reranker = Int(rerankerCache),
                          let decision = Int(decisionCache), let batch = Int(embeddingBatch),
                          let documents = Int(rerankerDocuments), let tokens = Int(rerankerTokens) else {
                        store.message = "시간, 캐시, 배치와 동시 요청 수를 숫자로 입력하세요."; return
                    }
                    let names = preload.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                    Task { await store.saveSettings(["preload_models": names, "cache_ttl_seconds": seconds,
                                                     "inference_max_concurrency_per_model": limit, "memory_guard_enabled": memoryGuard,
                                                     "cache_max_embedding_models": embedding, "cache_max_reranker_models": reranker,
                                                     "cache_max_decision_models": decision, "embedding_batch_max_texts": batch,
                                                     "rerank_batch_max_documents": documents, "rerank_batch_max_tokens": tokens]) }
                }.disabled(store.ownership == "external")
                Text("메모리 보호는 요청을 제한하는 기준이며 macOS의 강제 메모리 상한이 아닙니다.")
                    .foregroundStyle(.secondary)
            }
        }.formStyle(.grouped)
    }

    var runtime: some View {
        Form {
            Section("변환과 런타임") {
                LabeledContent("지원 모델", value: "d1 원본 / 공식 d1 MLX FP16")
                LabeledContent("변환 정밀도", value: "FP16")
                LabeledContent("동시 변환", value: "1개")
                Text("다운로드, 구조 확인, 필요한 변환, 검증과 등록을 서버에서 수행합니다.").foregroundStyle(.secondary)
                Text(pretty(store.runtime)).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
            }
        }.formStyle(.grouped)
    }

    var logs: some View {
        Form {
            Section("로그와 진단") {
                Picker("로그 수준", selection: $logLevel) {
                    ForEach(["DEBUG", "INFO", "WARNING", "ERROR"], id: \.self) { Text($0).tag($0) }
                }
                Toggle("채팅 요청 본문 디버그 기록", isOn: $logBodies)
                fieldState("log_level")
                fieldState("debug_log_chat_request_bodies")
                Button("로그 설정 저장") {
                    Task { await store.saveSettings(["log_level": logLevel, "debug_log_chat_request_bodies": logBodies]) }
                }.disabled(store.ownership == "external")
                Button("진단 자료 내보내기…") { Task { await store.exportDiagnostics() } }
                Text("요청 본문은 기본적으로 기록하지 않습니다. 진단 자료에는 본문, 인증정보와 로그가 포함되지 않습니다.")
                    .foregroundStyle(.secondary)
            }
        }.formStyle(.grouped)
    }

    @ViewBuilder func fieldState(_ name: String) -> some View {
        let value = store.field(name)
        let saved = displayValue(value["saved"], field: name)
        let actual = displayValue(value["applied"], field: name)
        let next = displayValue(value["effective_on_restart"], field: name)
        let sources = ["yaml": "설정 파일", "environment": "환경 변수", "dotenv": "환경 파일", "default": "기본값"]
        let source = sources[value["source"] as? String ?? ""] ?? "확인 불가"
        Text("저장: \(saved) · 적용: \(actual) · 다음 실행: \(next) · 출처: \(source)\((value["restart_required"] as? Bool == true) ? " · 재시작 필요" : "")")
            .font(.caption).foregroundStyle(.secondary)
    }

    func displayValue(_ raw: Any?, field: String) -> String {
        guard let raw, !(raw is NSNull) else { return "확인 불가" }
        if field == "preload_models", let models = raw as? [String] {
            return models.isEmpty ? "없음" : models.joined(separator: ", ")
        }
        if ["memory_guard_enabled", "debug_log_chat_request_bodies"].contains(field),
           let enabled = raw as? Bool {
            return enabled ? "사용" : "사용 안 함"
        }
        return String(describing: raw)
    }

    func populate() {
        func saved(_ key: String) -> Any? { store.field(key)["saved"] }
        host = saved("host") as? String ?? "127.0.0.1"
        port = String(saved("port") as? Int ?? 8000)
        modelsDir = saved("models_dir") as? String ?? ""
        preload = (saved("preload_models") as? [String] ?? []).joined(separator: ", ")
        ttl = String(saved("cache_ttl_seconds") as? Int ?? 1800)
        queue = String(saved("inference_max_queue_per_model") as? Int ?? 8)
        concurrency = String(saved("inference_max_concurrency_per_model") as? Int ?? 1)
        memoryGuard = saved("memory_guard_enabled") as? Bool ?? true
        embeddingCache = String(saved("cache_max_embedding_models") as? Int ?? 3)
        rerankerCache = String(saved("cache_max_reranker_models") as? Int ?? 2)
        decisionCache = String(saved("cache_max_decision_models") as? Int ?? 1)
        embeddingBatch = String(saved("embedding_batch_max_texts") as? Int ?? 32)
        rerankerDocuments = String(saved("rerank_batch_max_documents") as? Int ?? 4)
        rerankerTokens = String(saved("rerank_batch_max_tokens") as? Int ?? 2048)
        logLevel = saved("log_level") as? String ?? "INFO"
        logBodies = saved("debug_log_chat_request_bodies") as? Bool ?? false
    }

    func applyPreset(_ selection: String) {
        guard selection != "custom" else { return }
        let conserve = selection == "conserve"
        ttl = conserve ? "300" : "1800"
        concurrency = "1"
        memoryGuard = true
        embeddingCache = conserve ? "1" : "3"
        rerankerCache = conserve ? "1" : "2"
        decisionCache = "1"
        embeddingBatch = conserve ? "16" : "32"
        rerankerDocuments = conserve ? "2" : "4"
        rerankerTokens = conserve ? "1024" : "2048"
    }

    func updateLoginStatus() {
        switch SMAppService.mainApp.status {
        case .enabled: loginStatus = "등록됨"
        case .requiresApproval: loginStatus = "시스템 설정에서 승인 필요"
        case .notRegistered: loginStatus = "등록되지 않음"
        case .notFound: loginStatus = "앱 등록 정보를 찾을 수 없음"
        @unknown default: loginStatus = "확인 불가"
        }
    }

    func updateNotificationStatus() async {
        let settings = await UNUserNotificationCenter.current().notificationSettings()
        notificationStatus = notificationAuthorizationText(settings.authorizationStatus)
    }
    func login(register: Bool) {
        do {
            if register { try SMAppService.mainApp.register() }
            else { try SMAppService.mainApp.unregister() }
            updateLoginStatus()
        } catch { store.message = error.localizedDescription; updateLoginStatus() }
    }
}

struct OriginalCleanupView: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.dismiss) private var dismiss
    @State private var candidates: [[String: Any]] = []
    @State private var loading = false
    @State private var cleaning = false
    @State private var confirming = false
    @State private var error: String?

    private var displayed: [[String: Any]] { Array(candidates.prefix(100)) }
    private var logicalBytes: Int64 {
        displayed.reduce(0) { $0 + ($1["size_bytes"] as? Int64 ?? 0) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("참조되지 않는 원본 정리").font(.title2.bold())
            Text("등록된 모델이 참조하지 않는 보존 원본만 표시합니다. 변환된 모델과 공유 원본은 유지합니다.")
                .foregroundStyle(.secondary)
            if loading {
                ProgressView("정리 대상을 확인하고 있습니다…")
            } else if let error {
                Text(error).foregroundStyle(.orange).textSelection(.enabled)
            } else if candidates.isEmpty {
                Text("정리할 수 있는 원본이 없습니다.").foregroundStyle(.secondary)
            } else {
                Text("\(displayed.count)개 · 파일 크기 합계 \(ByteCountFormatter.string(fromByteCount: logicalBytes, countStyle: .file))")
                Text("파일의 논리 크기입니다. 실제 확보 공간은 저장 장치와 공유 파일에 따라 다를 수 있습니다.")
                    .font(.caption).foregroundStyle(.secondary)
                if candidates.count > displayed.count {
                    Text("한 번에 최대 100개를 정리합니다. 나머지는 정리 후 다시 확인하세요.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                List(displayed.indices, id: \.self) { index in
                    let entry = displayed[index]
                    VStack(alignment: .leading) {
                        Text(entry["name"] as? String ?? "").textSelection(.enabled)
                        if let size = entry["size_bytes"] as? Int64 {
                            Text(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }.frame(minHeight: 140, maxHeight: 250)
            }
            HStack {
                Button("다시 확인") { Task { await scan() } }.disabled(loading || cleaning)
                Spacer()
                Button("닫기") { dismiss() }.disabled(cleaning)
                Button("표시된 원본 정리…", role: .destructive) { confirming = true }
                    .disabled(loading || cleaning || displayed.isEmpty || error != nil)
            }
        }
        .padding(24).frame(width: 610).frame(minHeight: 300)
        .task { await scan() }
        .confirmationDialog("표시된 원본을 정리할까요?", isPresented: $confirming) {
            Button("원본 정리", role: .destructive) { Task { await clean() } }
        } message: {
            Text("원본 파일은 삭제되며 다시 필요하면 다운로드해야 합니다. 정리 직전에 참조와 파일 변경 여부를 다시 확인합니다.")
        }
    }

    private func scan() async {
        loading = true
        defer { loading = false }
        do {
            let result = try await store.request("/originals") as? [String: Any] ?? [:]
            candidates = result["candidates"] as? [[String: Any]] ?? []
            error = nil
        } catch {
            candidates = []
            self.error = error.localizedDescription
        }
    }

    private func clean() async {
        cleaning = true
        defer { cleaning = false }
        do {
            _ = try await store.request("/originals/cleanup", method: "POST", body: ["entries": displayed])
            await scan()
        } catch {
            self.error = error.localizedDescription
        }
    }
}
