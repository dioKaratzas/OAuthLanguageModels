import Foundation

private let anthropicOAuthClientID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
private let anthropicTokenURL = URL(string: "https://platform.claude.com/v1/oauth/token")!
/// Refresh a bit before expiry to avoid racing the server clock.
private let refreshLeewayMilliseconds: Double = 60_000

// MARK: - AnthropicAuthService

/// Refresh + freshness helpers for Anthropic OAuth credentials. Stateless:
/// the caller owns persistence and passes load/save closures into
/// `validAccessToken`, which keeps this package free of opinions about
/// where credentials live on disk.
public enum AnthropicAuthService {
    /// Returns a valid Anthropic OAuth access token, refreshing and
    /// persisting via the supplied closures if necessary.
    public static func validAccessToken(
        load: () throws -> AnthropicAuth?,
        save: (AnthropicAuth) throws -> Void,
        now: () -> Date = Date.init
    ) async throws -> String {
        guard let auth = try load() else {
            throw AnthropicAuthError.missingCredentials
        }

        if isFresh(auth, now: now()) {
            return auth.accessToken
        }

        let refreshed: AnthropicAuth
        do {
            refreshed = try await refresh(using: auth.refreshToken)
        } catch {
            throw AnthropicAuthError.refreshFailed(error.localizedDescription)
        }
        try save(refreshed)
        return refreshed.accessToken
    }

    public static func refresh(using refreshToken: String) async throws -> AnthropicAuth {
        var request = URLRequest(url: anthropicTokenURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let body = AnthropicRefreshRequest(
            grantType: "refresh_token",
            clientId: anthropicOAuthClientID,
            refreshToken: refreshToken
        )
        request.httpBody = try JSONEncoder.snakeCase.encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw AnthropicAuthError.refreshFailed("No HTTP response.")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw AnthropicAuthError.refreshFailed("HTTP \(http.statusCode): \(String(decoding: data, as: UTF8.self))")
        }

        let payload: AnthropicRefreshResponse
        do {
            payload = try JSONDecoder.snakeCase.decode(AnthropicRefreshResponse.self, from: data)
        } catch {
            throw AnthropicAuthError.invalidResponse
        }

        let expiresAt = Date().timeIntervalSince1970 * 1000 + payload.expiresIn * 1000 - refreshLeewayMilliseconds
        return AnthropicAuth(
            accessToken: payload.accessToken,
            refreshToken: payload.refreshToken,
            expiresAt: expiresAt
        )
    }

    public static func isFresh(_ auth: AnthropicAuth, now: Date = Date()) -> Bool {
        guard let expiresAt = auth.expiresAt else {
            return true
        }
        return now.timeIntervalSince1970 * 1000 + refreshLeewayMilliseconds < expiresAt
    }
}

// MARK: - AnthropicRefreshRequest

private struct AnthropicRefreshRequest: Encodable {
    let grantType: String
    let clientId: String
    let refreshToken: String
}

// MARK: - AnthropicRefreshResponse

private struct AnthropicRefreshResponse: Decodable {
    let accessToken: String
    let refreshToken: String
    let expiresIn: Double
}
