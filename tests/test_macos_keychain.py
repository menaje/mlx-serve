"""Exercise the app's Keychain operations against a disposable keychain."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swiftc") is None,
    reason="The native Keychain test requires the macOS Swift compiler",
)
def test_keychain_secrets_survive_process_restart_without_touching_login_keychain(tmp_path):
    source = Path(__file__).resolve().parents[1] / "macos/Sources/MLXServeApp/Keychain.swift"
    main = tmp_path / "main.swift"
    main.write_text(
        """
import Foundation
import Security

@main struct Smoke {
    static func main() throws {
        let path = CommandLine.arguments[1]
        let create = CommandLine.arguments[2] == "create"
        let password = "temporary-test-keychain-password"
        _ = SecKeychainSetUserInteractionAllowed(false)
        var keychain: SecKeychain?
        let status = password.withCString { pass in
            if create {
                return SecKeychainCreate(path, UInt32(strlen(pass)), pass, false, nil, &keychain)
            }
            let opened = SecKeychainOpen(path, &keychain)
            if opened != errSecSuccess { return opened }
            return SecKeychainUnlock(keychain, UInt32(strlen(pass)), pass, true)
        }
        precondition(status == errSecSuccess && keychain != nil,
                     "Keychain setup failed: \\(status)")
        let target = keychain!
        let id = "isolated-test-agent"
        let first = String(repeating: "a", count: 64)
        let second = String(repeating: "b", count: 64)
        if create {
            precondition(HFKeychain.read(keychain: target) == nil)
            let initiallyAbsent = try AgentKeychain.read(id, keychain: target)
            precondition(initiallyAbsent == nil)
            try HFKeychain.save("hf_disposable_fixture", keychain: target)
            try HFKeychain.save("hf_updated_fixture", keychain: target)
            try AgentKeychain.save(first, keyID: id, keychain: target)
            try AgentKeychain.save(second, keyID: id, keychain: target)
        } else {
            precondition(HFKeychain.read(keychain: target) == "hf_updated_fixture")
            let saved = try AgentKeychain.read(id, keychain: target)
            precondition(saved == second)
            try HFKeychain.save("", keychain: target)
            try AgentKeychain.remove(id, keychain: target)
            precondition(HFKeychain.read(keychain: target) == nil)
            let removed = try AgentKeychain.read(id, keychain: target)
            precondition(removed == nil)
        }
    }
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "keychain-test"
    subprocess.run(
        ["swiftc", "-parse-as-library", str(source), str(main), "-o", str(executable)],
        check=True, capture_output=True, timeout=60,
    )
    keychain = tmp_path / "disposable.keychain-db"
    for operation in ("create", "reopen"):
        subprocess.run(
            [str(executable), str(keychain), operation],
            check=True, capture_output=True, timeout=20,
        )
