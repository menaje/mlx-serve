// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLXServeApp",
    platforms: [.macOS("26.0")],
    products: [.executable(name: "MLXServeApp", targets: ["MLXServeApp"])],
    targets: [.executableTarget(name: "MLXServeApp")]
)
