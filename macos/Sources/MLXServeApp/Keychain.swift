import Foundation
import Security

enum HFKeychain {
    private static func query(_ keychain: SecKeychain?, adding: Bool = false) -> [String: Any] {
        var result: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "org.mlx-serve.huggingface",
            kSecAttrAccount as String: "token"
        ]
        if let keychain {
            result[(adding ? kSecUseKeychain : kSecMatchSearchList) as String] =
                adding ? keychain : [keychain]
        }
        return result
    }

    static func read(keychain: SecKeychain? = nil) -> String? {
        var attributes = query(keychain)
        attributes[kSecReturnData as String] = true
        attributes[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        guard SecItemCopyMatching(attributes as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func save(_ value: String, keychain: SecKeychain? = nil) throws {
        if value.isEmpty {
            let status = SecItemDelete(query(keychain) as CFDictionary)
            guard status == errSecSuccess || status == errSecItemNotFound else {
                throw KeychainError(status)
            }
            return
        }
        let bytes = Data(value.utf8)
        let status = SecItemUpdate(query(keychain) as CFDictionary, [kSecValueData as String: bytes] as CFDictionary)
        if status == errSecItemNotFound {
            var attributes = query(keychain, adding: true)
            attributes[kSecValueData as String] = bytes
            let addStatus = SecItemAdd(attributes as CFDictionary, nil)
            guard addStatus == errSecSuccess else { throw KeychainError(addStatus) }
        } else if status != errSecSuccess {
            throw KeychainError(status)
        }
    }

    struct KeychainError: LocalizedError {
        let status: OSStatus
        init(_ status: OSStatus) { self.status = status }
        var errorDescription: String? {
            SecCopyErrorMessageString(status, nil) as String? ?? "Keychain 오류 (\(status))"
        }
    }
}

enum AgentKeychain {
    private static func query(_ keyID: String, keychain: SecKeychain?, adding: Bool = false) -> [String: Any] {
        var result: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: "org.mlx-serve.background-agent",
         kSecAttrAccount as String: keyID]
        if let keychain {
            result[(adding ? kSecUseKeychain : kSecMatchSearchList) as String] =
                adding ? keychain : [keychain]
        }
        return result
    }

    static func read(_ keyID: String, keychain: SecKeychain? = nil) throws -> String? {
        var attributes = query(keyID, keychain: keychain)
        attributes[kSecReturnData as String] = true
        attributes[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(attributes as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess else { throw HFKeychain.KeychainError(status) }
        guard let data = result as? Data, let secret = String(data: data, encoding: .utf8),
              secret.count >= 32 else { throw HFKeychain.KeychainError(errSecDecode) }
        return secret
    }

    static func save(_ secret: String, keyID: String, keychain: SecKeychain? = nil) throws {
        let data = Data(secret.utf8)
        let status = SecItemUpdate(query(keyID, keychain: keychain) as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if status == errSecItemNotFound {
            var attributes = query(keyID, keychain: keychain, adding: true)
            attributes[kSecValueData as String] = data
            let added = SecItemAdd(attributes as CFDictionary, nil)
            guard added == errSecSuccess else { throw HFKeychain.KeychainError(added) }
        } else if status != errSecSuccess {
            throw HFKeychain.KeychainError(status)
        }
    }

    static func remove(_ keyID: String, keychain: SecKeychain? = nil) throws {
        let status = SecItemDelete(query(keyID, keychain: keychain) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw HFKeychain.KeychainError(status)
        }
    }
}
