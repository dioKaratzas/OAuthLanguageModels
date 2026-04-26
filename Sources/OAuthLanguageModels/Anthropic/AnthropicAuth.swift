import Foundation

// MARK: - AnthropicAuth

/// On-the-wire OAuth credentials for Anthropic / Claude. The `expiresAt`
/// timestamp is Unix epoch milliseconds, matching what the Claude Code CLI
/// persists.
public struct AnthropicAuth: Codable, Equatable, Sendable {
    // MARK: Lifecycle

    public init(accessToken: String, refreshToken: String, expiresAt: Double? = nil) {
        self.accessToken = accessToken
        self.refreshToken = refreshToken
        self.expiresAt = expiresAt
    }

    // MARK: Public

    public var accessToken: String
    public var refreshToken: String
    /// Unix epoch milliseconds at which the access token expires.
    public var expiresAt: Double?

    // MARK: Internal

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case refreshToken = "refresh_token"
        case expiresAt = "expires_at"
    }
}

// MARK: - AnthropicAuthError

/// Errors thrown when working with stored Anthropic credentials.
public enum AnthropicAuthError: LocalizedError, Sendable {
    case missingCredentials
    case refreshFailed(String)
    case invalidResponse

    // MARK: Public

    public var errorDescription: String? {
        switch self {
        case .missingCredentials:
            "No Anthropic auth available. Run the OAuth login flow first."
        case let .refreshFailed(message):
            "Anthropic token refresh failed: \(message)"
        case .invalidResponse:
            "Anthropic token endpoint returned an invalid response."
        }
    }
}
