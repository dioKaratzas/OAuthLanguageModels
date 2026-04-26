import CryptoKit
import Foundation
import Security

// MARK: - PKCE

/// PKCE (Proof Key for Code Exchange — RFC 7636) verifier/challenge pair used
/// by both the Codex and Anthropic OAuth flows. Exposed publicly so callers
/// can drive the flows themselves or reuse this for additional providers.
public struct PKCE: Sendable, Equatable {
    // MARK: Lifecycle

    public init(verifier: String, challenge: String) {
        self.verifier = verifier
        self.challenge = challenge
    }

    // MARK: Public

    public let verifier: String
    public let challenge: String

    /// Generates a fresh PKCE pair using 32 random bytes for the verifier.
    public static func generate() -> PKCE {
        let verifier = oauthRandomBytes(count: 32).base64URLEncodedString
        return PKCE(verifier: verifier, challenge: Self.challenge(for: verifier))
    }

    /// Computes the base64url-encoded SHA-256 challenge for an existing verifier.
    public static func challenge(for verifier: String) -> String {
        let hash = SHA256.hash(data: Data(verifier.utf8))
        return Data(hash).base64URLEncodedString
    }
}

// MARK: - Helpers

func oauthRandomHex(byteCount: Int) -> String {
    oauthRandomBytes(count: byteCount).map { String(format: "%02x", $0) }.joined()
}

func oauthRandomBytes(count: Int) -> Data {
    var bytes = [UInt8](repeating: 0, count: count)
    _ = bytes.withUnsafeMutableBufferPointer { buf in
        SecRandomCopyBytes(kSecRandomDefault, buf.count, buf.baseAddress!)
    }
    return Data(bytes)
}

// MARK: - base64url

public extension Data {
    /// Returns this data as a base64url-encoded string (RFC 4648 §5),
    /// suitable for embedding in OAuth/JWT tokens.
    var base64URLEncodedString: String {
        base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    /// Decodes a base64url-encoded string (RFC 4648 §5).
    init?(base64URLEncoded string: String) {
        var base64 = string
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        let remainder = base64.count % 4
        if remainder != 0 {
            base64 += String(repeating: "=", count: 4 - remainder)
        }
        self.init(base64Encoded: base64)
    }
}
