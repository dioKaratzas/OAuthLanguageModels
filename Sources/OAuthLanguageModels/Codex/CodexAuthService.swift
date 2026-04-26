import Foundation

private let codexRefreshClientID = "app_EMoamEEZ73f0CkXaXp7hrann"
private let codexRefreshTokenURL = URL(string: "https://auth.openai.com/oauth/token")!
/// Refresh a bit before expiry to avoid racing the clock.
private let codexRefreshLeewaySeconds: TimeInterval = 60

// MARK: - CodexAuthService

/// Refresh + freshness helpers for Codex OAuth credentials. Stateless: the
/// caller owns persistence and passes load/save closures into `validAuth`,
/// which keeps this package free of opinions about where credentials live
/// on disk.
public enum CodexAuthService {
    /// Returns a `CodexAuth` with a valid access token, refreshing and
    /// persisting via the supplied closures if the stored access token is
    /// expired or close to it.
    ///
    /// - Parameters:
    ///   - load: Returns the currently-stored credentials, or `nil` if the
    ///     user hasn't logged in yet.
    ///   - save: Persists a refreshed `CodexAuth`. Called only when a
    ///     refresh actually happened.
    ///   - now: Clock injection point for tests.
    public static func validAuth(
        load: () throws -> CodexAuth?,
        save: (CodexAuth) throws -> Void,
        now: () -> Date = Date.init
    ) async throws -> CodexAuth {
        guard let auth = try load() else {
            throw CodexAuthError.refreshFailed("No Codex auth available. Run the OAuth login flow first.")
        }

        if isFresh(auth.accessToken, now: now()) {
            return auth
        }

        let refreshed: CodexAuth
        do {
            refreshed = try await refresh(using: auth.refreshToken, previousAccountID: auth.accountID)
        } catch {
            throw CodexAuthError.refreshFailed(error.localizedDescription)
        }

        try save(refreshed)
        return refreshed
    }

    /// Performs the actual OAuth refresh request. Mirrors the flow used by
    /// the OpenAI Codex CLI's `refreshAccessToken`.
    public static func refresh(using refreshToken: String, previousAccountID: String? = nil) async throws -> CodexAuth {
        var request = URLRequest(url: codexRefreshTokenURL)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        var components = URLComponents()
        components.queryItems = [
            URLQueryItem(name: "grant_type", value: "refresh_token"),
            URLQueryItem(name: "refresh_token", value: refreshToken),
            URLQueryItem(name: "client_id", value: codexRefreshClientID),
        ]
        request.httpBody = components.percentEncodedQuery?.data(using: .utf8)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw CodexAuthError.refreshFailed("No HTTP response.")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw CodexAuthError.refreshFailed("HTTP \(http.statusCode): \(String(decoding: data, as: UTF8.self))")
        }

        let payload: CodexRefreshResponse
        do {
            payload = try JSONDecoder.snakeCase.decode(CodexRefreshResponse.self, from: data)
        } catch {
            throw CodexAuthError.invalidResponse
        }

        // Providers sometimes omit the refresh token on refresh; fall back to the existing one.
        let newRefresh = payload.refreshToken ?? refreshToken
        let accountID =
            payload.idToken.flatMap(CodexOAuthFlow.accountID(fromJWT:))
                ?? CodexOAuthFlow.accountID(fromJWT: payload.accessToken)
                ?? previousAccountID

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        return CodexAuth(
            accessToken: payload.accessToken,
            refreshToken: newRefresh,
            idToken: payload.idToken,
            accountID: accountID,
            lastRefresh: formatter.string(from: Date())
        )
    }

    /// True if the JWT's `exp` claim is still comfortably in the future.
    /// If we can't parse the token, we conservatively return `false` so we
    /// attempt a refresh.
    public static func isFresh(_ accessToken: String, now: Date = Date()) -> Bool {
        guard let exp = exp(ofJWT: accessToken) else { return false }
        return now.timeIntervalSince1970 + codexRefreshLeewaySeconds < exp
    }

    /// Returns the `exp` claim of the supplied JWT as a Unix timestamp, or
    /// `nil` if it can't be parsed.
    public static func exp(ofJWT token: String) -> TimeInterval? {
        let parts = token.split(separator: ".")
        guard
            parts.count == 3,
            let data = Data(base64URLEncoded: String(parts[1])),
            let payload = try? JSONDecoder.snakeCase.decode(CodexJWTExpPayload.self, from: data)
        else { return nil }
        return payload.exp
    }
}

// MARK: - CodexRefreshResponse

private struct CodexRefreshResponse: Decodable {
    let accessToken: String
    let refreshToken: String?
    let idToken: String?
}

// MARK: - CodexJWTExpPayload

private struct CodexJWTExpPayload: Decodable {
    let exp: TimeInterval?
}
