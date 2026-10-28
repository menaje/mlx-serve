import SwiftUI

@main
struct MLXServeApp: App {
    @StateObject private var store = AppStore()

    var body: some Scene {
        WindowGroup("MLX Serve", id: "main") {
            MainView().environmentObject(store)
                .frame(minWidth: 880, minHeight: 580)
        }
        Settings { PreferencesView().environmentObject(store) }
        MenuBarExtra {
            MenuContent().environmentObject(store)
        } label: {
            Image(systemName: store.state == "connected" ? "bolt.horizontal.circle.fill" : "bolt.horizontal.circle")
        }
    }
}

struct MenuContent: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.openWindow) var openWindow

    var body: some View {
        Text("서버: \(readableState(store.state))")
        Text("진행 작업: \(store.activeJobs)")
        Divider()
        Button("MLX Serve 열기") { openWindow(id: "main"); NSApp.activate(ignoringOtherApps: true) }
        SettingsLink { Text("설정…") }
        Divider()
        if ["stopped", "failed"].contains(store.state) {
            Button("서버 시작") { Task { await store.startServer() } }.disabled(!store.agentReady || store.busy)
        } else if store.ownership == "app" {
            Button("서버 중지") { Task { await store.stopServer() } }.disabled(store.busy)
        }
        Divider()
        Button("MLX Serve 종료") { NSApp.terminate(nil) }.keyboardShortcut("q")
    }
}

enum MainPage: String, CaseIterable, Identifiable {
    case overview, models, jobs, diagnostics
    var id: String { rawValue }
    var title: String {
        switch self { case .overview: return "개요"; case .models: return "모델"; case .jobs: return "작업"; case .diagnostics: return "진단" }
    }
    var symbol: String {
        switch self { case .overview: return "square.grid.2x2"; case .models: return "shippingbox"; case .jobs: return "clock.arrow.circlepath"; case .diagnostics: return "stethoscope" }
    }
}

struct MainView: View {
    @EnvironmentObject var store: AppStore
    @State private var selection: MainPage? = .overview

    var body: some View {
        NavigationSplitView {
            List(MainPage.allCases, selection: $selection) { page in
                Label(page.title, systemImage: page.symbol).tag(page)
            }.navigationSplitViewColumnWidth(min: 165, ideal: 185)
        } detail: {
            Group {
                switch selection ?? .overview {
                case .overview: OverviewView()
                case .models: ModelsView()
                case .jobs: JobsView()
                case .diagnostics: DiagnosticsView()
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .alert("알림", isPresented: Binding(
            get: { store.message != nil }, set: { if !$0 { store.message = nil } }
        )) { Button("확인") { store.message = nil } } message: { Text(store.message ?? "") }
    }
}

struct PageHeader: View {
    let title: String
    let subtitle: String
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.largeTitle.bold())
            Text(subtitle).foregroundStyle(.secondary)
        }.padding(.bottom, 8)
    }
}

struct StatusBadge: View {
    let value: String
    var body: some View {
        Text(readableState(value)).font(.caption.weight(.medium))
            .padding(.horizontal, 9).padding(.vertical, 4)
            .background(color.opacity(0.15), in: Capsule()).foregroundStyle(color)
    }
    var color: Color {
        switch value {
        case "connected", "loaded", "succeeded", "complete": return .green
        case "failed", "port_conflict", "interrupted": return .red
        case "degraded", "unknown", "starting", "running", "cancelling": return .orange
        default: return .secondary
        }
    }
}

struct OverviewView: View {
    @EnvironmentObject var store: AppStore
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                PageHeader(title: "개요", subtitle: "서버와 모델의 현재 상태를 확인합니다.")
                GroupBox {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Label("서버", systemImage: "server.rack").font(.headline)
                            Spacer()
                            StatusBadge(value: store.state)
                        }
                        HStack {
                            Text(store.baseURL).font(.system(.body, design: .monospaced))
                            Button("복사") { copyText(store.baseURL) }
                        }
                        Text(ownershipText).font(.callout).foregroundStyle(.secondary)
                        if store.state == "port_conflict" {
                            Text("설정된 포트를 다른 프로세스가 사용 중입니다. 포트를 바꾸거나 점유 중인 서비스를 확인하세요.")
                                .foregroundStyle(.orange)
                        }
                        if store.state == "failed" {
                            Text("앱이 시작한 서버가 예상치 않게 종료되었습니다. 저장 공간과 런타임을 확인한 뒤 다시 시작하세요.")
                                .foregroundStyle(.orange)
                        }
                        if let note = store.automaticStartNote {
                            Text(note).foregroundStyle(.secondary)
                        }
                        if store.state == "degraded" {
                            Text("서버, 워커 또는 모델 저장소의 상태가 불완전합니다. 진단 화면에서 상태를 확인하세요.")
                                .foregroundStyle(.orange)
                        }
                        if let storage = store.modelState["storage"] as? [String: Any],
                           storage["state"] as? String == "unavailable" {
                            Text("모델 저장소에 접근할 수 없습니다. 저장 장치의 연결과 권한을 확인하세요.")
                                .foregroundStyle(.orange)
                        }
                        if store.state == "stopped" && ["0.0.0.0", "::"].contains(store.server["next_bind_host"] as? String ?? "") {
                            Text("현재 설정으로 서버를 시작하면 로컬 네트워크에 추론 API가 공개됩니다. 설정에서 접근 범위를 변경할 수 있습니다.")
                                .foregroundStyle(.orange)
                        }
                        HStack {
                            Button("서버 시작") { Task { await store.startServer() } }
                                .disabled(!["stopped", "failed"].contains(store.state) || !store.agentReady || store.busy)
                            Button("서버 중지") { Task { await store.stopServer() } }
                                .disabled(store.ownership != "app" || store.busy)
                            SettingsLink { Text("설정…") }
                        }
                    }.padding(8)
                }
                if let installation = store.server["installation"] as? [String: Any],
                   let launchd = installation["launchd"] as? [String: Any] {
                    GroupBox("기존 설치 확인") {
                        VStack(alignment: .leading, spacing: 5) {
                            if let cli = installation["homebrew_cli"] as? String
                                ?? installation["cli_in_path"] as? String {
                                Text("기존 CLI: \(cli)").textSelection(.enabled)
                            } else {
                                Text("검사한 경로에서 기존 CLI를 찾지 못했습니다.")
                            }
                            Text("사용자 launchd 서비스: \(launchdState(launchd["user_state"] as? String))")
                            if let files = launchd["files"] as? [String], !files.isEmpty {
                                ForEach(files, id: \.self) { file in
                                    Text("서비스 설정: \(file)").textSelection(.enabled)
                                }
                            }
                            if let config = installation["config_path"] as? String {
                                Text("설정 경로: \(config)").textSelection(.enabled)
                            }
                            Text("서비스 확인은 읽기 전용이며 실행 중인 서버와의 연결 여부는 위에 별도로 표시합니다.")
                                .font(.caption).foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                    }
                }
                HStack(alignment: .top, spacing: 14) {
                    GroupBox("모델") {
                        VStack(alignment: .leading, spacing: 5) {
                            if store.modelStateAvailable {
                                Text("\(store.models.count)개 설치됨").font(.title2.bold())
                                Text("\(store.models.filter { $0["load_state"] as? String == "loaded" }.count)개 메모리에 적재됨")
                                    .foregroundStyle(.secondary)
                            } else {
                                Text("확인 불가").font(.title2.bold())
                                Text("서버 및 워커 상태를 확인하세요.").foregroundStyle(.secondary)
                            }
                        }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                    }
                    GroupBox("작업") {
                        VStack(alignment: .leading, spacing: 5) {
                            Text("\(store.activeJobs)개 진행 중").font(.title2.bold())
                            Text("앱을 종료하면 앱이 시작한 작업과 서버를 종료합니다.")
                                .foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                    }
                }
                if let health = store.server["health"] as? [String: Any],
                   let memory = health["memory"] as? [String: Any],
                   let snapshot = memory["snapshot"] as? [String: Any],
                   let available = snapshot["system_available_bytes"] as? Int64,
                   let total = snapshot["system_total_bytes"] as? Int64,
                   let gatewayRSS = snapshot["process_rss_bytes"] as? Int64 {
                    GroupBox("메모리") {
                        VStack(alignment: .leading, spacing: 5) {
                            Text("시스템 여유 \(ByteCountFormatter.string(fromByteCount: available, countStyle: .memory)) / 전체 \(ByteCountFormatter.string(fromByteCount: total, countStyle: .memory))")
                            Text("게이트웨이 프로세스 RSS \(ByteCountFormatter.string(fromByteCount: gatewayRSS, countStyle: .memory)); 추론 워커는 포함하지 않습니다.")
                                .font(.caption).foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                    }
                }
                if let error = store.modelError { Text(error).foregroundStyle(.orange) }
                Text("CONI 연결 주소는 위의 API 주소입니다. 클라이언트에는 적재할 모델 ID도 함께 지정하세요.")
                    .font(.callout).foregroundStyle(.secondary)
            }.padding(24)
        }
    }
    var ownershipText: String {
        switch store.ownership {
        case "app": return "이 앱이 시작한 서버입니다. 창을 닫아도 메뉴 막대에서 계속 실행됩니다."
        case "external": return "기존 서버에 연결했습니다. 이 앱은 해당 서버를 중지하지 않습니다."
        default:
            return store.state == "stopped" ? "서버가 중지되어 있습니다." : "서버 실행 상태를 확인하고 있습니다."
        }
    }
    private func launchdState(_ state: String?) -> String {
        switch state {
        case "running": return "실행 중"
        case "loaded": return "등록됨"
        case "not_loaded": return "등록되지 않음"
        default: return "확인 불가"
        }
    }
}

struct ModelsView: View {
    @EnvironmentObject var store: AppStore
    @State private var selected: String?
    @State private var adding = false
    var current: [String: Any]? { store.models.first { $0["name"] as? String == selected } }
    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            VStack(alignment: .leading, spacing: 12) {
                PageHeader(title: "모델", subtitle: "파일 준비와 메모리 적재를 별도로 관리합니다.")
                Button { adding = true } label: { Label("모델 추가", systemImage: "plus") }
                    .disabled(!["connected", "degraded"].contains(store.state) || !store.modelStateAvailable)
                List(selection: $selected) {
                    ForEach(store.models.indices, id: \.self) { index in
                        let model = store.models[index]
                        let name = model["name"] as? String ?? ""
                        VStack(alignment: .leading, spacing: 4) {
                            Text(name).fontWeight(.medium)
                            HStack {
                                Text(model["type"] as? String ?? "")
                                Text("·")
                                Text(readableState(model["load_state"] as? String ?? "unknown"))
                            }.font(.caption).foregroundStyle(.secondary)
                        }.tag(name)
                    }
                }.frame(minHeight: 200, maxHeight: .infinity)
                if let error = store.modelError { Text(error).foregroundStyle(.orange).font(.caption) }
            }.padding(20).frame(width: 315).frame(maxHeight: .infinity, alignment: .topLeading)
            Divider()
            Group {
                if let current { ModelDetailView(model: current).id(selected) }
                else { ContentUnavailableView("모델 선택", systemImage: "shippingbox", description: Text("목록에서 모델을 선택하세요.")) }
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .sheet(isPresented: $adding) { AddModelView() }
    }
}

struct AddModelView: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.dismiss) var dismiss
    @State private var repo = "LiquidAI/d1-omni-600M-MLX-fp16"
    @State private var name = ""
    @State private var revision = ""
    @State private var keepOriginal = false
    @State private var loadAfter = true
    private let supported = [
        "LiquidAI/d1-omni-600M", "LiquidAI/d1-omni-600M-MLX-fp16", "vllm-sr/d3-lite",
        "mlx-community/all-MiniLM-L6-v2-bf16", "mlx-community/Qwen3-Reranker-0.6B-4bit"
    ]
    private var submittedRepo: String { repo.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var isD3: Bool { submittedRepo == "vllm-sr/d3-lite" }
    private var canKeepOriginal: Bool { submittedRepo == "LiquidAI/d1-omni-600M" }
    private var profile: String {
        switch submittedRepo {
        case "vllm-sr/d3-lite": return "결정 모델 / BF16"
        case "mlx-community/all-MiniLM-L6-v2-bf16": return "임베딩 / BF16"
        case "mlx-community/Qwen3-Reranker-0.6B-4bit": return "리랭커 / 4비트"
        default: return "결정 모델 / FP16"
        }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            PageHeader(title: "모델 추가", subtitle: "서버의 검증된 모델 준비 경로를 사용합니다.")
            Form {
                TextField("Hugging Face ID", text: $repo)
                TextField("로컬 이름 (선택)", text: $name)
                TextField("Revision (선택)", text: $revision)
                LabeledContent("유형 / 정밀도", value: profile)
                Toggle("원본 보존", isOn: $keepOriginal).disabled(!canKeepOriginal)
                Toggle("준비 후 메모리에 로드", isOn: $loadAfter)
            }
            Text(isD3
                 ? "d3 원본 파일의 무결성과 MLX 추론을 검증합니다. 원본 파일을 직접 사용하므로 별도의 원본 사본을 만들지 않습니다."
                 : "d1 원본·MLX FP16, d3-lite, all-MiniLM 임베딩과 Qwen3 리랭커를 지원합니다. 파일과 추론을 검증하며 이미 호환되는 모델은 다시 변환하지 않습니다.")
                .font(.callout).foregroundStyle(.secondary)
            if !supported.contains(submittedRepo) { Text("이 모델은 아직 지원하지 않습니다.").foregroundStyle(.orange) }
            HStack {
                Spacer()
                Button("취소") { dismiss() }
                Button("준비 시작") {
                    let submitted = submittedRepo
                    Task {
                        await store.startJob(repo: submitted, name: name, revision: revision, keepOriginal: canKeepOriginal && keepOriginal, loadAfter: loadAfter)
                        dismiss()
                    }
                }.keyboardShortcut(.defaultAction).disabled(!supported.contains(submittedRepo) || store.busy)
            }
        }.padding(24).frame(width: 570)
    }
}

struct ModelDetailView: View {
    @EnvironmentObject var store: AppStore
    let model: [String: Any]
    @State private var testText = "I love my cat."
    @State private var question = "Is a cat mentioned?"
    @State private var testType = "bool"
    @State private var choiceOptions = "cat, dog"
    @State private var scoreLevels = "negative, neutral, positive"
    @State private var testResult = ""
    @State private var testing = false
    @State private var confirmDelete = false
    @State private var removalPreview: [String: Any]?
    @State private var reviewingRemoval = false
    var name: String { model["name"] as? String ?? "" }
    var kind: String { model["type"] as? String ?? "" }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack { Text(name).font(.title2.bold()); Spacer(); StatusBadge(value: model["load_state"] as? String ?? "unknown") }
                LabeledContent("유형", value: kind)
                LabeledContent("원본", value: model["source"] as? String ?? "")
                LabeledContent("파일", value: readableState(model["file_state"] as? String ?? "unknown"))
                LabeledContent("요청", value: readableState(model["request_state"] as? String ?? "unknown"))
                if ["decision", "embedding", "reranker"].contains(kind),
                   store.modelState["default_models_supported"] as? Bool == true {
                    LabeledContent("기본 모델", value: model["is_default"] as? Bool == true ? "선택됨" : "선택되지 않음")
                    Button(model["is_default"] as? Bool == true ? "기본 모델 해제" : "기본 모델로 설정") {
                        let operation = model["is_default"] as? Bool == true ? "clear-default" : "set-default"
                        Task { await store.modelCommand(operation, name: name, type: kind) }
                    }.disabled(store.busy)
                    Text("같은 유형의 API 요청에서 모델 ID를 생략하면 선택된 기본 모델을 사용합니다. ID를 지정한 요청은 그 모델을 사용합니다.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let size = model["size_bytes"] as? Int64 { LabeledContent("모델 파일", value: ByteCountFormatter.string(fromByteCount: size, countStyle: .file)) }
                HStack {
                    Button("로드") { Task { await store.modelCommand("load", name: name, type: kind) } }
                        .disabled(store.busy || model["load_state"] as? String != "not_loaded")
                    Button("언로드") { Task { await store.modelCommand("unload", name: name, type: kind) } }
                        .disabled(store.busy || model["load_state"] as? String != "loaded")
                    Button("모델 ID 복사") { copyText(name) }
                }
                Divider()
                Text("간단한 테스트").font(.headline)
                if ["decision", "embedding", "reranker"].contains(kind) {
                    TextField("입력", text: $testText)
                    if kind == "decision" {
                        Picker("질문 유형", selection: $testType) {
                            Text("예 / 아니요").tag("bool"); Text("선택지").tag("choice"); Text("점수").tag("score")
                        }
                        TextField("질문", text: $question)
                        if testType == "choice" {
                            TextField("선택지 (쉼표로 구분)", text: $choiceOptions)
                        } else if testType == "score" {
                            TextField("점수 단계 (낮은 순서부터 쉼표로 구분)", text: $scoreLevels)
                        }
                        Text("예제는 영문으로 제공됩니다. 결과는 모델과 입력 언어에 따라 달라집니다.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Button(testing ? "실행 중…" : "테스트 실행") { Task { await test() } }
                        .disabled(testing || !["connected", "degraded"].contains(store.state))
                    if !testResult.isEmpty { Text(testResult).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }
                } else {
                    Text("이 모델 유형의 테스트는 아직 제공되지 않습니다.").foregroundStyle(.secondary)
                }
                Divider()
                Text("설치 정보").font(.headline)
                Text(pretty(model["metadata"] ?? [:])).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                if let record = model["preparation"] as? [String: Any] {
                    Text("모델 준비 기록").font(.headline)
                    if record["state"] as? String == "recorded" {
                        LabeledContent("라이선스", value: record["license"] as? String ?? "정보 없음")
                        LabeledContent("변환", value: record["converted"] as? Bool == true ? "수행함" : "수행하지 않음")
                        if record["converted"] as? Bool == true,
                           let converter = record["converter"] as? String {
                            let version = record["converter_version"] as? String ?? ""
                            LabeledContent("변환 도구", value: "\(converter) \(version)".trimmingCharacters(in: .whitespaces))
                        }
                        if let options = record["converter_options"] as? [String: Any], !options.isEmpty {
                            LabeledContent("준비 옵션", value: pretty(options))
                        }
                        if let validation = record["validation"] as? [String: Any] {
                            LabeledContent("실행·검증 기록", value: pretty(validation))
                        }
                        Text("설치할 때 기록된 정보입니다. 모델을 새로 로드할 때 파일 무결성을 다시 검사합니다.")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("사용할 수 있는 모델 준비 기록이 없습니다.").foregroundStyle(.secondary)
                    }
                }
                Text("API 주소: \(store.baseURL) · 모델 ID: \(name)").font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button("API 주소와 모델 ID 복사") { copyText("\(store.baseURL)\n\(name)") }
                    if let example = apiExample {
                        Button("API 요청 예제 복사") { copyText(example) }
                    }
                }
                Divider()
                if ["stopped", "failed"].contains(store.state) {
                    Button("모델 제거…", role: .destructive) { Task { await previewRemoval() } }
                        .disabled(store.busy || store.activeJobs > 0 || reviewingRemoval)
                    Text("서버와 이 저장소를 사용하는 다른 앱을 모두 종료한 뒤 제거하세요. 보존한 원본과 공유 캐시는 유지됩니다.")
                        .font(.caption).foregroundStyle(.secondary)
                } else if kind == "decision" {
                    Button("모델 제거…", role: .destructive) {
                        removalPreview = nil
                        confirmDelete = true
                    }.disabled(store.busy)
                    Text("모델 등록과 준비된 모델 파일을 제거합니다. 보존한 원본과 공유 캐시는 유지됩니다.")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("이 모델을 제거하려면 서버를 중지하세요.").font(.caption).foregroundStyle(.secondary)
                }
            }.padding(24).frame(maxWidth: .infinity, alignment: .leading)
        }
        .confirmationDialog("\(name)을 제거할까요?", isPresented: $confirmDelete) {
            Button("모델 제거", role: .destructive) {
                Task {
                    if let preview = removalPreview {
                        await store.action { _ = try await store.request("/stored-models/remove", method: "POST", body: preview) }
                        removalPreview = nil
                    } else {
                        await store.modelCommand("delete", name: name, type: kind)
                    }
                }
            }
        } message: {
            if let preview = removalPreview {
                let size = (preview["size_bytes"] as? NSNumber)?.int64Value ?? 0
                Text("모델 파일 \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))과 등록 정보가 삭제됩니다. 다른 서버나 앱을 모두 종료하세요. 공유 원본은 유지됩니다.")
            } else {
                Text("모델 파일이 삭제됩니다. 다시 사용하려면 재다운로드 또는 재변환해야 합니다. 공유 원본은 유지됩니다.")
            }
        }
    }
    private func previewRemoval() async {
        reviewingRemoval = true
        defer { reviewingRemoval = false }
        do {
            removalPreview = try await store.request("/stored-models/preview", method: "POST", body: ["name": name, "type": kind]) as? [String: Any]
            if removalPreview != nil { confirmDelete = true }
        } catch { store.message = error.localizedDescription }
    }
    var apiExample: String? {
        let path: String
        var payload: [String: Any] = ["model": name]
        switch kind {
        case "decision":
            path = "/v1/decisions"
            payload["state"] = "I love my cat."
            payload["questions"] = ["answer": ["type": "bool", "instructions": "Is a cat mentioned?"]]
        case "embedding":
            path = "/v1/embeddings"
            payload["input"] = "Example text"
        case "reranker":
            path = "/v1/rerank"
            payload["query"] = "Example query"
            payload["documents"] = ["Example document", "Another document"]
        default:
            return nil
        }
        return "POST \(store.baseURL)\(path)\nContent-Type: application/json\n\n\(pretty(payload))"
    }
    func test() async {
        testing = true
        defer { testing = false }
        let start = Date()
        var body: [String: Any] = ["model": name]
        let path: String
        if kind == "decision" {
            var spec: [String: Any] = ["type": testType, "instructions": question]
            if testType != "bool" {
                let text = testType == "choice" ? choiceOptions : scoreLevels
                let options = text.split(separator: ",").map {
                    $0.trimmingCharacters(in: .whitespacesAndNewlines)
                }.filter { !$0.isEmpty }
                let limit = testType == "choice" ? 255 : 10
                guard (2...limit).contains(options.count), Set(options).count == options.count else {
                    testResult = "서로 다른 항목을 2개 이상 쉼표로 구분해 입력하세요. 점수 단계는 최대 10개입니다."
                    return
                }
                spec["criteria"] = testType == "choice"
                    ? Dictionary(uniqueKeysWithValues: options.map { ($0, $0) }) as Any
                    : options as Any
            }
            body["state"] = testText
            body["questions"] = ["answer": spec]
            path = "/v1/decisions"
        } else if kind == "embedding" {
            body["input"] = testText; path = "/v1/embeddings"
        } else {
            body["query"] = testText; body["documents"] = [testText, "다른 문서"]; path = "/v1/rerank"
        }
        do {
            let result = try await store.request("/test", method: "POST", body: ["path": path, "body": body])
            testResult = String(format: "소요 시간: %.2f초\n", Date().timeIntervalSince(start)) + pretty(result)
        } catch { testResult = error.localizedDescription }
    }
}
