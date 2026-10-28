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
                        if let result = job["result"] as? [String: Any],
                           result["auto_load"] as? String == "failed" {
                            Text("준비는 완료했지만 자동 로드에 실패했습니다. 서버를 확인한 뒤 모델 화면에서 직접 로드하세요.")
                                .foregroundStyle(.orange)
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
                                        loadAfter: params["load_after"] as? Bool ?? false
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
                                        Text("저장소 파일 크기와 준비에 필요한 여유 공간을 반영한 추정치입니다.")
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                if let size = event["temporary_file_bytes"] as? Int64 {
                                    Text("작업 임시 파일 크기 \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file)) (측정값)")
                                        .foregroundStyle(.secondary)
                                }
                                if let message = event["message"] as? String {
                                    Text(event["status"] as? String == "load_failed"
                                        ? "자동 로드에 실패했습니다. 서버를 확인한 뒤 모델 화면에서 직접 로드하세요."
                                        : message).foregroundStyle(.orange)
                                }
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
    case "authentication", "access_approval": return "이 앱은 로그인이나 접근 승인 없이 받을 수 있는 공개 모델을 지원합니다. 모델 공개 상태를 확인하세요."
    case "storage_permission": return "모델 저장 위치의 권한과 장치 연결 상태를 확인하세요."
    case "disk": return "저장 공간을 확보하고 모델 저장 장치의 연결을 확인하세요."
    case "memory": return "다른 모델을 언로드하거나 메모리를 확보한 뒤 시도하세요."
    case "network": return "네트워크 연결을 확인한 뒤 다시 시도하세요."
    case "missing_file": return "저장소 ID, revision과 필요한 모델 파일을 확인하세요."
    case "timeout": return "네트워크와 실행 환경을 확인한 뒤 다시 시도하세요."
    case "unsupported": return "지원되는 결정·임베딩·리랭커 모델 중에서 선택하세요."
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
    @State private var queueTimeout = "180"
    @State private var concurrency = "1"
    @State private var memoryGuard = true
    @State private var processMemoryFraction = "0.75"
    @State private var availableMemoryFraction = "0.1"
    @State private var loadHeadroomFraction = "0.1"
    @State private var preset = "custom"
    @State private var embeddingCache = "3"
    @State private var rerankerCache = "2"
    @State private var decisionCache = "1"
    @State private var embeddingBatch = "32"
    @State private var rerankerDocuments = "4"
    @State private var rerankerTokens = "2048"
    @State private var logLevel = "INFO"
    @State private var logRetention = ""
    @State private var logBodies = false
    @State private var loginStatus = "확인 중"
    @State private var notificationStatus = "확인 중"
    @State private var reviewingOriginals = false
    @State private var reviewingHFCache = false
    @State private var reviewingMigration = false
    @State private var previewingMigration = false
    @State private var migrationPreview: [String: Any] = [:]
    @AppStorage("automaticStartServer") private var automaticStartServer = false
    @AppStorage("automaticRecoverServer") private var automaticRecoverServer = false
    @AppStorage("jobNotifications") private var jobNotifications = false

    private var settingsReady: Bool { store.settings["fields"] as? [String: Any] != nil }

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
        .onChange(of: store.agentReady) { _, ready in
            if ready && !settingsReady {
                Task { await store.reloadSettings(); populate() }
            }
        }
        .sheet(isPresented: $reviewingOriginals) { OriginalCleanupView().environmentObject(store) }
        .sheet(isPresented: $reviewingHFCache) { HFCacheInventoryView().environmentObject(store) }
        .sheet(isPresented: $reviewingMigration) {
            StorageMigrationPreviewView(preview: migrationPreview).environmentObject(store)
        }
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
                Text("창을 닫아도 앱과 서버는 유지됩니다. 앱을 종료하면 앱이 시작한 작업과 서버를 종료합니다. 기존 외부 서버는 계속 실행됩니다.")
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
                }.disabled(store.ownership == "external" || !settingsReady || store.busy)
                Text("앱이 시작한 이 버전의 서버는 LAN을 켜도 관리 기능을 로컬로 제한합니다.")
                    .foregroundStyle(.secondary)
            }
            Section("대기열") {
                TextField("모델당 대기 요청 수", text: $queue)
                TextField("요청 대기 시간 제한 (초)", text: $queueTimeout)
                fieldState("inference_max_queue_per_model")
                fieldState("inference_queue_timeout_seconds")
                Text("빈칸은 해당 제한을 사용하지 않습니다. 요청 수를 0으로 설정하면 대기 없이 초과 요청을 거절합니다. 시간 제한에는 배치 처리 시간도 포함됩니다.")
                    .font(.caption).foregroundStyle(.secondary)
                Button("대기열 저장") {
                    let trimmedQueue = queue.trimmingCharacters(in: .whitespacesAndNewlines)
                    let queueValue: Any
                    if trimmedQueue.isEmpty {
                        queueValue = NSNull()
                    } else if let value = Int(trimmedQueue), value >= 0 {
                        queueValue = value
                    } else {
                        store.message = "대기 요청 수는 0 이상의 정수로 입력하거나 비워 두세요."; return
                    }
                    let timeout = optionalDecimal(queueTimeout, allowsZero: false)
                    guard timeout.valid else {
                        store.message = "대기 시간은 0보다 큰 초 단위 숫자로 입력하거나 비워 두세요."; return
                    }
                    Task { await store.saveSettings(["inference_max_queue_per_model": queueValue,
                                                     "inference_queue_timeout_seconds": timeout.value]) }
                }.disabled(store.ownership == "external" || !settingsReady || store.busy)
            }
        }.formStyle(.grouped)
    }

    var storage: some View {
        Form {
            Section("저장소") {
                TextField("새 모델 저장 위치", text: $modelsDir)
                    .disabled(store.migratingStorage)
                fieldState("models_dir")
                if let disk = store.modelState["disk"] as? [String: Any],
                   let available = disk["free_bytes"] as? Int64,
                   let total = disk["total_bytes"] as? Int64 {
                    LabeledContent("현재 저장 장치", value: "여유 \(ByteCountFormatter.string(fromByteCount: available, countStyle: .file)) / 전체 \(ByteCountFormatter.string(fromByteCount: total, countStyle: .file))")
                } else {
                    LabeledContent("현재 저장 장치", value: "서버 연결 후 확인")
                }
                Text("서버와 다른 CLI 서비스를 중지한 뒤 변경하세요. 기존 모델이 없다면 새 위치를 저장하고, 기존 모델이 있다면 복사 및 전환을 선택하세요.")
                    .foregroundStyle(.secondary)
                HStack {
                    Button("새 모델 위치 저장") { Task { await store.saveSettings(["models_dir": modelsDir]) } }
                    Button(previewingMigration ? "확인 중…" : "기존 모델 복사 및 전환…") {
                        previewingMigration = true
                        Task {
                            defer { previewingMigration = false }
                            do {
                                migrationPreview = try await store.previewStorageMigration(modelsDir)
                                reviewingMigration = true
                            } catch { store.message = error.localizedDescription }
                        }
                    }
                }.disabled(store.ownership == "external" || !settingsReady || store.busy || store.migratingStorage || previewingMigration)
                Text("복사할 때는 아직 없는 새 상위 폴더 안의 경로를 지정하세요. 예: /Volumes/외장디스크/MLXServe/models")
                    .font(.caption).foregroundStyle(.secondary)
                if store.migratingStorage {
                    let copied = (store.storageMigration["copied_bytes"] as? NSNumber)?.doubleValue ?? 0
                    let total = (store.storageMigration["size_bytes"] as? NSNumber)?.doubleValue ?? 0
                    let phase = store.storageMigration["phase"] as? String ?? "checking"
                    if total > 0 && phase == "copying" {
                        ProgressView("모델 파일 복사 중", value: min(copied, total), total: total)
                    } else {
                        ProgressView(phase == "applying" ? "저장 위치 적용 중" : "파일 확인 중")
                    }
                    Text("복사와 검증이 끝날 때까지 앱을 열어 두세요. 기존 파일은 삭제되지 않습니다.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Button("참조되지 않는 원본 정리…") { reviewingOriginals = true }
                    .disabled(!store.modelStateAvailable || store.activeJobs > 0 || store.migratingStorage)
                Text("다른 모델이 참조하는 원본과 변환된 모델은 정리하지 않습니다.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Hugging Face") {
                Button("공유 캐시 사용량 확인…") { reviewingHFCache = true }
                    .disabled(!store.agentReady)
                Text("공유 캐시는 다른 앱에서도 사용할 수 있어 자동으로 삭제하지 않습니다.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("로그인이나 접근 승인 없이 받을 수 있는 공개 모델을 사용합니다. 토큰은 사용하지 않습니다.")
                    .foregroundStyle(.secondary)
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
                DisclosureGroup("메모리 보호 기준") {
                    TextField("프로세스 RSS 기준 비율", text: $processMemoryFraction)
                    fieldState("memory_process_limit_fraction")
                    TextField("시스템 가용 메모리 최소 비율", text: $availableMemoryFraction)
                    fieldState("memory_min_available_fraction")
                    TextField("로드·생성 시 확보할 여유 비율", text: $loadHeadroomFraction)
                    fieldState("memory_load_headroom_fraction")
                    Text("전체 메모리 대비 비율입니다. 예: 0.75는 75%. 빈칸은 해당 기준을 사용하지 않습니다. 로드·생성 시 여유 기준은 요청 보호와 별도로 적용됩니다.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Button("성능 설정 저장") {
                    guard let seconds = Int(ttl), let limit = Int(concurrency),
                          let embedding = Int(embeddingCache), let reranker = Int(rerankerCache),
                          let decision = Int(decisionCache), let batch = Int(embeddingBatch),
                          let documents = Int(rerankerDocuments), let tokens = Int(rerankerTokens) else {
                        store.message = "시간, 캐시, 배치와 동시 요청 수를 숫자로 입력하세요."; return
                    }
                    let processFraction = optionalDecimal(processMemoryFraction, allowsZero: false, maximum: 1)
                    let availableFraction = optionalDecimal(availableMemoryFraction, allowsZero: false, maximum: 1)
                    let headroomFraction = optionalDecimal(loadHeadroomFraction, allowsZero: true, maximum: 1)
                    guard processFraction.valid, availableFraction.valid, headroomFraction.valid else {
                        store.message = "메모리 비율은 0보다 크고 1보다 작은 숫자로 입력하세요. 로드 여유 비율에는 0도 사용할 수 있습니다."; return
                    }
                    let names = preload.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                    Task { await store.saveSettings(["preload_models": names, "cache_ttl_seconds": seconds,
                                                     "inference_max_concurrency_per_model": limit, "memory_guard_enabled": memoryGuard,
                                                     "memory_process_limit_fraction": processFraction.value,
                                                     "memory_min_available_fraction": availableFraction.value,
                                                     "memory_load_headroom_fraction": headroomFraction.value,
                                                     "cache_max_embedding_models": embedding, "cache_max_reranker_models": reranker,
                                                     "cache_max_decision_models": decision, "embedding_batch_max_texts": batch,
                                                     "rerank_batch_max_documents": documents, "rerank_batch_max_tokens": tokens]) }
                }.disabled(store.ownership == "external" || !settingsReady || store.busy)
                Text("메모리 보호는 요청을 제한하는 기준이며 macOS의 강제 메모리 상한이 아닙니다.")
                    .foregroundStyle(.secondary)
            }
        }.formStyle(.grouped)
    }

    var runtime: some View {
        Form {
            Section("변환과 런타임") {
                LabeledContent("지원 모델", value: "d1 / d3-lite / all-MiniLM / Qwen3 리랭커")
                LabeledContent("d1 원본 변환 정밀도", value: "FP16")
                LabeledContent("d3 정밀도", value: "BF16 / readout FP32")
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
                TextField("앱 서버 로그 보관 기간 (일, 빈칸은 무기한)", text: $logRetention)
                Toggle("채팅 요청 본문 디버그 기록", isOn: $logBodies)
                fieldState("log_level")
                fieldState("app_log_retention_days")
                fieldState("debug_log_chat_request_bodies")
                Button("로그 설정 저장") {
                    let input = logRetention.trimmingCharacters(in: .whitespacesAndNewlines)
                    let days: Any
                    if input.isEmpty {
                        days = NSNull()
                    } else if let value = Int(input), (1...36500).contains(value) {
                        days = value
                    } else {
                        store.message = "로그 보관 기간은 1~36500일로 입력하거나 비워 두세요."; return
                    }
                    Task { await store.saveSettings(["log_level": logLevel, "app_log_retention_days": days,
                                                     "debug_log_chat_request_bodies": logBodies]) }
                }.disabled(store.ownership == "external" || !settingsReady || store.busy)
                LabeledContent("앱 서버 로그", value: "~/.mlx-serve/app-server-logs")
                Text("기간을 지정하면 서버를 시작할 때 종료된 앱 서버의 오래된 로그만 삭제합니다. 실행 중이거나 종료를 확인하지 못한 로그와 이전 app-server.log는 보존합니다.")
                    .font(.caption).foregroundStyle(.secondary)
                if (store.server["log_cleanup_failures"] as? Int ?? 0) > 0 {
                    Text("일부 오래된 로그를 정리하지 못했습니다. 로그 폴더의 접근 권한을 확인하세요.")
                        .foregroundStyle(.orange)
                }
                Button("진단 자료 내보내기…") { Task { await store.exportDiagnostics() } }
                Text("요청 본문은 기본적으로 기록하지 않습니다. 진단 자료에는 본문, 인증정보와 로그가 포함되지 않습니다.")
                    .foregroundStyle(.secondary)
            }
        }.formStyle(.grouped)
    }

    @ViewBuilder func fieldState(_ name: String) -> some View {
        let value = store.field(name)
        let saved = displayValue(value["saved"], field: name)
        let actual = store.settings["server_settings_available"] as? Bool == false
            ? "확인 불가" : displayValue(value["applied"], field: name)
        let next = displayValue(value["effective_on_restart"], field: name)
        let sources = ["yaml": "설정 파일", "environment": "환경 변수", "dotenv": "환경 파일", "default": "기본값"]
        let source = sources[value["source"] as? String ?? ""] ?? "확인 불가"
        Text("저장: \(saved) · 적용: \(actual) · 다음 실행: \(next) · 출처: \(source)\((value["restart_required"] as? Bool == true) ? " · 재시작 필요" : "")")
            .font(.caption).foregroundStyle(.secondary)
    }

    func displayValue(_ raw: Any?, field: String) -> String {
        guard let raw else { return "확인 불가" }
        if raw is NSNull {
            if field == "app_log_retention_days" { return "무기한 보관" }
            return ["inference_max_queue_per_model", "inference_queue_timeout_seconds",
                    "memory_process_limit_fraction", "memory_min_available_fraction",
                    "memory_load_headroom_fraction"].contains(field) ? "제한 없음" : "확인 불가"
        }
        if field == "preload_models", let models = raw as? [String] {
            return models.isEmpty ? "없음" : models.joined(separator: ", ")
        }
        if ["memory_guard_enabled", "debug_log_chat_request_bodies"].contains(field),
           let enabled = raw as? Bool {
            return enabled ? "사용" : "사용 안 함"
        }
        return String(describing: raw)
    }

    func optionalDecimal(_ text: String, allowsZero: Bool, maximum: Double? = nil) -> (valid: Bool, value: Any) {
        let input = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if input.isEmpty { return (true, NSNull()) }
        guard let number = Double(input), number.isFinite,
              (allowsZero ? number >= 0 : number > 0),
              maximum.map({ number < $0 }) ?? true else { return (false, NSNull()) }
        return (true, number)
    }

    func populate() {
        func saved(_ key: String) -> Any? { store.field(key)["saved"] }
        host = saved("host") as? String ?? "127.0.0.1"
        port = String(saved("port") as? Int ?? 8000)
        modelsDir = saved("models_dir") as? String ?? ""
        preload = (saved("preload_models") as? [String] ?? []).joined(separator: ", ")
        ttl = String(saved("cache_ttl_seconds") as? Int ?? 1800)
        queue = (saved("inference_max_queue_per_model") as? NSNumber).map { String($0.intValue) } ?? ""
        queueTimeout = (saved("inference_queue_timeout_seconds") as? NSNumber).map { String($0.doubleValue) } ?? ""
        concurrency = String(saved("inference_max_concurrency_per_model") as? Int ?? 1)
        memoryGuard = saved("memory_guard_enabled") as? Bool ?? true
        processMemoryFraction = (saved("memory_process_limit_fraction") as? NSNumber).map { String($0.doubleValue) } ?? ""
        availableMemoryFraction = (saved("memory_min_available_fraction") as? NSNumber).map { String($0.doubleValue) } ?? ""
        loadHeadroomFraction = (saved("memory_load_headroom_fraction") as? NSNumber).map { String($0.doubleValue) } ?? ""
        embeddingCache = String(saved("cache_max_embedding_models") as? Int ?? 3)
        rerankerCache = String(saved("cache_max_reranker_models") as? Int ?? 2)
        decisionCache = String(saved("cache_max_decision_models") as? Int ?? 1)
        embeddingBatch = String(saved("embedding_batch_max_texts") as? Int ?? 32)
        rerankerDocuments = String(saved("rerank_batch_max_documents") as? Int ?? 4)
        rerankerTokens = String(saved("rerank_batch_max_tokens") as? Int ?? 2048)
        logLevel = saved("log_level") as? String ?? "INFO"
        logRetention = (saved("app_log_retention_days") as? NSNumber).map { String($0.intValue) } ?? ""
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

struct StorageMigrationPreviewView: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.dismiss) private var dismiss
    let preview: [String: Any]

    private var size: Int64 { (preview["size_bytes"] as? NSNumber)?.int64Value ?? 0 }
    private var free: Int64 { (preview["free_bytes"] as? NSNumber)?.int64Value ?? 0 }
    private var enoughSpace: Bool { free >= size + max(16 * 1024 * 1024, size / 100) }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("기존 모델 복사 및 전환").font(.title2.bold())
            LabeledContent("현재 위치", value: preview["source"] as? String ?? "")
            LabeledContent("새 위치", value: preview["destination"] as? String ?? "")
            LabeledContent("복사할 파일", value: ByteCountFormatter.string(fromByteCount: size, countStyle: .file))
            LabeledContent("대상 디스크 여유", value: ByteCountFormatter.string(fromByteCount: free, countStyle: .file))
            Text("모델 파일, 등록 정보, 기본 모델 선택 및 보관된 원본을 복사하고 검증한 뒤 저장 위치를 전환합니다. 기존 파일은 삭제하지 않습니다.")
            Text("다른 CLI나 서비스가 기존 저장소를 수정하지 않도록 모두 중지하세요. 복사가 실패하거나 앱을 종료하면 기존 설정과 파일을 유지하며, 불완전한 복사 대상 폴더가 남을 수 있습니다.")
                .foregroundStyle(.secondary)
            if !enoughSpace {
                Text("복사와 검증에 필요한 여유 공간이 부족합니다.").foregroundStyle(.orange)
            }
            HStack {
                Spacer()
                Button("취소") { dismiss() }
                Button("복사 시작") {
                    let destination = preview["destination"] as? String ?? ""
                    dismiss()
                    Task { await store.startStorageMigration(destination) }
                }.disabled(!enoughSpace || preview["destination"] as? String == nil)
            }
        }.padding(24).frame(width: 600)
    }
}

struct HFCacheInventoryView: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.dismiss) private var dismiss
    @State private var inventory: [String: Any] = [:]
    @State private var loading = false
    @State private var error: String?
    @State private var candidates: [[String: Any]] = []
    @State private var showingCandidates = false
    @State private var candidatesTruncated = false
    @State private var selected: [String: Any]?
    @State private var confirming = false

    private var repositories: [[String: Any]] { inventory["repositories"] as? [[String: Any]] ?? [] }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Hugging Face 공유 캐시").font(.title2.bold())
            Text("공유 캐시를 다른 앱에서도 사용할 수 있습니다. 정리하기 전에 이 서버와 캐시를 사용하는 다른 앱을 모두 종료하세요.")
                .foregroundStyle(.secondary)
            if loading {
                ProgressView("캐시 사용량을 확인하고 있습니다…")
            } else if let error {
                Text(error).foregroundStyle(.orange).textSelection(.enabled)
            } else {
                let count = inventory["repository_count"] as? Int ?? 0
                let bytes = inventory["size_bytes"] as? Int64 ?? 0
                Text("저장소 \(count)개 · 보고된 파일 크기 \(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file))")
                if let path = inventory["path"] as? String {
                    Text(path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }
                Text("Hub 캐시의 파일 크기입니다. Xet 캐시와 작업 임시 파일은 포함되지 않으며, 실제 확보 가능한 용량과 다를 수 있습니다.")
                    .font(.caption).foregroundStyle(.secondary)
                if let warnings = inventory["warning_count"] as? Int, warnings > 0 {
                    Text("\(warnings)개 항목을 확인하지 못해 사용량에 포함하지 않았습니다.")
                        .font(.caption).foregroundStyle(.orange)
                }
                if inventory["truncated"] as? Bool == true {
                    Text("목록에는 처음 100개 저장소만 표시합니다.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if showingCandidates {
                    Text("등록된 모델의 원본과 이름이 붙은 revision은 제외합니다. 표시된 예상 용량은 실제 확보 공간과 다를 수 있습니다.")
                        .font(.caption).foregroundStyle(.secondary)
                    if candidatesTruncated {
                        Text("처음 100개만 표시합니다. 정리 후 다시 확인하세요.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if candidates.isEmpty {
                        Text("정리할 수 있는 이전 revision이 없습니다.").foregroundStyle(.secondary)
                    } else {
                        List(candidates.indices, id: \.self) { index in
                            let entry = candidates[index]
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(entry["repo"] as? String ?? "")
                                    Text(entry["revision"] as? String ?? "")
                                        .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                                    let bytes = (entry["size_bytes"] as? NSNumber)?.int64Value ?? 0
                                    Text("예상 확보 \(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file))")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                Button("정리…", role: .destructive) {
                                    selected = entry
                                    confirming = true
                                }.disabled(loading)
                            }
                        }.frame(minHeight: 140, maxHeight: 250)
                    }
                } else if repositories.isEmpty {
                    Text("확인된 캐시 저장소가 없습니다.").foregroundStyle(.secondary)
                } else {
                    List(repositories.indices, id: \.self) { index in
                        let repo = repositories[index]
                        VStack(alignment: .leading) {
                            Text(repo["id"] as? String ?? "").textSelection(.enabled)
                            let kind = repo["type"] as? String ?? ""
                            let revisions = repo["revision_count"] as? Int ?? 0
                            let size = repo["size_bytes"] as? Int64 ?? 0
                            Text("\(kind) · revision \(revisions)개 · \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }.frame(minHeight: 140, maxHeight: 250)
                }
            }
            HStack {
                Button("사용량 다시 확인") { Task { await scan() } }.disabled(loading)
                Button("이전 revision 정리…") { Task { await scanCandidates() } }.disabled(loading)
                Spacer()
                Button("닫기") { dismiss() }
            }
        }
        .padding(24).frame(width: 610).frame(minHeight: 300)
        .task { await scan() }
        .confirmationDialog("선택한 캐시를 정리할까요?", isPresented: $confirming) {
            Button("선택한 revision 정리", role: .destructive) { Task { await clean() } }
        } message: {
            Text("\(selected?["repo"] as? String ?? "")의 이전 파일을 삭제합니다. 다른 앱을 먼저 종료하세요. 다시 필요하면 다운로드해야 합니다.")
        }
    }

    private func scan() async {
        loading = true
        defer { loading = false }
        do {
            inventory = try await store.request("/hf-cache") as? [String: Any] ?? [:]
            error = nil
            showingCandidates = false
        } catch {
            inventory = [:]
            self.error = error.localizedDescription
        }
    }

    private func scanCandidates() async {
        loading = true
        defer { loading = false }
        do {
            let result = try await store.request("/hf-cache/cleanup") as? [String: Any] ?? [:]
            candidates = result["candidates"] as? [[String: Any]] ?? []
            candidatesTruncated = result["truncated"] as? Bool == true
            showingCandidates = true
            error = nil
        } catch {
            candidates = []
            self.error = error.localizedDescription
        }
    }

    private func clean() async {
        guard let selected else { return }
        loading = true
        defer { loading = false }
        do {
            _ = try await store.request("/hf-cache/cleanup", method: "POST", body: selected)
            self.selected = nil
            let result = try await store.request("/hf-cache/cleanup") as? [String: Any] ?? [:]
            candidates = result["candidates"] as? [[String: Any]] ?? []
            candidatesTruncated = result["truncated"] as? Bool == true
            inventory = try await store.request("/hf-cache") as? [String: Any] ?? [:]
            error = nil
        } catch {
            self.error = error.localizedDescription
        }
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
