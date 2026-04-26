import Foundation

// MARK: - CodexToken

/// Lightweight token bundle consumed by `CodexLanguageModel`. Decouples the
/// language model from any on-disk auth representation so callers can swap
/// in test fixtures or alternative auth strategies.
public struct CodexToken: Sendable, Equatable {
    // MARK: Lifecycle

    public init(accessToken: String, accountID: String) {
        self.accessToken = accessToken
        self.accountID = accountID
    }

    // MARK: Public

    public var accessToken: String
    public var accountID: String
}

// MARK: - CodexAuth

/// On-the-wire OAuth credentials for ChatGPT / Codex. The CodingKeys match
/// the snake_case keys used by both pi's `auth.json` and Codex's token
/// endpoint, so this type round-trips cleanly with either.
public struct CodexAuth: Codable, Equatable, Sendable {
    // MARK: Lifecycle

    public init(
        accessToken: String,
        refreshToken: String,
        idToken: String? = nil,
        accountID: String? = nil,
        lastRefresh: String? = nil
    ) {
        self.accessToken = accessToken
        self.refreshToken = refreshToken
        self.idToken = idToken
        self.accountID = accountID
        self.lastRefresh = lastRefresh
    }

    // MARK: Public

    public var accessToken: String
    public var refreshToken: String
    public var idToken: String?
    public var accountID: String?
    public var lastRefresh: String?

    // MARK: Internal

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case refreshToken = "refresh_token"
        case idToken = "id_token"
        case accountID = "account_id"
        case lastRefresh = "last_refresh"
    }
}

public extension CodexAuth {
    /// Resolves a `CodexToken`, filling in the ChatGPT account ID from the
    /// access token's JWT if it wasn't already persisted.
    func toToken() throws -> CodexToken {
        let resolved = accountID ?? CodexOAuthFlow.accountID(fromJWT: accessToken)
        guard let resolved, !resolved.isEmpty else {
            throw CodexAuthError.missingAccountID
        }
        return CodexToken(accessToken: accessToken, accountID: resolved)
    }
}

// MARK: - CodexAuthError

/// Errors thrown when working with stored Codex credentials.
public enum CodexAuthError: LocalizedError, Sendable {
    case missingAccountID
    case refreshFailed(String)
    case invalidResponse

    // MARK: Public

    public var errorDescription: String? {
        switch self {
        case .missingAccountID:
            "Could not resolve ChatGPT account ID from Codex auth."
        case let .refreshFailed(message):
            "Codex token refresh failed: \(message)"
        case .invalidResponse:
            "Codex token endpoint returned an invalid response."
        }
    }
}
