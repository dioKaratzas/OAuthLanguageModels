import Foundation

// MARK: - Configuration

private let codexOAuthClientID = "app_EMoamEEZ73f0CkXaXp7hrann"
private let codexAuthorizeURL = URL(string: "https://auth.openai.com/oauth/authorize")!
private let codexTokenURL = URL(string: "https://auth.openai.com/oauth/token")!
private let codexCallbackPort: UInt16 = 1455
private let codexCallbackPath = "/auth/callback"
private let codexRedirectURI = "http://localhost:\(codexCallbackPort)\(codexCallbackPath)"
private let codexScope = "openid profile email offline_access"
private let codexCallbackTimeoutSeconds: TimeInterval = 300

// MARK: - CodexOAuthError

public enum CodexOAuthError: LocalizedError, Sendable {
    case tokenExchangeFailed(status: Int, message: String)
    case invalidTokenResponse

    // MARK: Public

    public var errorDescription: String? {
        switch self {
        case let .tokenExchangeFailed(status, message):
            "Codex token exchange failed with status \(status): \(message)"
        case .invalidTokenResponse:
            "Codex token endpoint returned an invalid response."
        }
    }
}

// MARK: - Flow

/// Default value for the `originator` query parameter sent during the
/// Codex authorization request. Embedders that want to attribute requests
/// to their own product should pass an explicit `originator:` argument to
/// `login` / `buildAuthorizationURL`.
public let defaultCodexOriginator = "OAuthLanguageModels"

// MARK: - CodexPendingLogin

/// A login attempt in progress. Spin one up with `CodexOAuthFlow.login()`,
/// direct the user to `authorizationURL` (e.g. open it in a browser, render a
/// button, display a QR code), then `await pending.waitForCallback()` to
/// receive the resulting `CodexAuth`.
public struct CodexPendingLogin: Sendable {
    // MARK: Public

    /// The URL the user must visit to authorize the application.
    public let authorizationURL: URL

    /// Waits for the loopback callback server to receive the authorization
    /// code, exchanges it for tokens, and returns the resulting auth. The
    /// underlying server is stopped before this method returns.
    public func waitForCallback() async throws -> CodexAuth {
        let server = server
        defer { Task.detached { await server.stop() } }
        let code = try await server.waitForCode(timeout: codexCallbackTimeoutSeconds)
        return try await CodexOAuthFlow.exchangeAuthorizationCode(code: code, verifier: verifier)
    }

    /// Stops the loopback callback server without waiting for a callback.
    /// Call this if the user cancels the flow before completing it.
    public func cancel() async {
        await server.stop()
    }

    // MARK: Internal

    let server: OAuthCallbackServer
    let verifier: String
}

// MARK: - CodexOAuthFlow

/// Drives the Codex / ChatGPT OAuth login flow: spins up a loopback callback
/// server, builds the authorization URL, and (once the user has completed
/// the browser flow) exchanges the resulting authorization code for tokens.
/// Callers are responsible for surfacing the authorization URL to the user
/// however they see fit.
public enum CodexOAuthFlow {
    /// Starts a login attempt. Returns a `CodexPendingLogin` containing the
    /// URL the user must visit; `await pending.waitForCallback()` to receive
    /// the resulting `CodexAuth` once they complete the flow.
    public static func login(originator: String = defaultCodexOriginator) async throws -> CodexPendingLogin {
        let pkce = PKCE.generate()
        let state = oauthRandomHex(byteCount: 16)

        let server = try await OAuthCallbackServer.start(
            port: codexCallbackPort,
            path: codexCallbackPath,
            expectedState: state
        )

        let authorizeURL = buildAuthorizationURL(
            challenge: pkce.challenge,
            state: state,
            originator: originator
        )
        return CodexPendingLogin(
            authorizationURL: authorizeURL,
            server: server,
            verifier: pkce.verifier
        )
    }

    public static func buildAuthorizationURL(
        challenge: String,
        state: String,
        originator: String = defaultCodexOriginator
    ) -> URL {
        var components = URLComponents(url: codexAuthorizeURL, resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "response_type", value: "code"),
            URLQueryItem(name: "client_id", value: codexOAuthClientID),
            URLQueryItem(name: "redirect_uri", value: codexRedirectURI),
            URLQueryItem(name: "scope", value: codexScope),
            URLQueryItem(name: "code_challenge", value: challenge),
            URLQueryItem(name: "code_challenge_method", value: "S256"),
            URLQueryItem(name: "state", value: state),
            URLQueryItem(name: "id_token_add_organizations", value: "true"),
            URLQueryItem(name: "codex_cli_simplified_flow", value: "true"),
            URLQueryItem(name: "originator", value: originator),
        ]
        return components.url!
    }

    public static func exchangeAuthorizationCode(code: String, verifier: String) async throws -> CodexAuth {
        var request = URLRequest(url: codexTokenURL)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        var components = URLComponents()
        components.queryItems = [
            URLQueryItem(name: "grant_type", value: "authorization_code"),
            URLQueryItem(name: "client_id", value: codexOAuthClientID),
            URLQueryItem(name: "code", value: code),
            URLQueryItem(name: "code_verifier", value: verifier),
            URLQueryItem(name: "redirect_uri", value: codexRedirectURI),
        ]
        request.httpBody = components.percentEncodedQuery?.data(using: .utf8)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw CodexOAuthError.invalidTokenResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            throw CodexOAuthError.tokenExchangeFailed(
                status: http.statusCode,
                message: String(decoding: data, as: UTF8.self)
            )
        }

        let payload: CodexTokenExchangeResponse
        do {
            payload = try JSONDecoder.snakeCase.decode(CodexTokenExchangeResponse.self, from: data)
        } catch {
            throw CodexOAuthError.invalidTokenResponse
        }

        let accountID = payload.idToken.flatMap(Self.accountID(fromJWT:))
            ?? Self.accountID(fromJWT: payload.accessToken)

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        return CodexAuth(
            accessToken: payload.accessToken,
            refreshToken: payload.refreshToken,
            idToken: payload.idToken,
            accountID: accountID,
            lastRefresh: formatter.string(from: Date())
        )
    }

    /// Extracts the ChatGPT account ID embedded in the auth claim of an
    /// OpenAI-issued JWT. Returns `nil` if the token isn't a JWT or
    /// doesn't contain the claim.
    public static func accountID(fromJWT token: String) -> String? {
        let parts = token.split(separator: ".")
        guard
            parts.count == 3,
            let payloadData = Data(base64URLEncoded: String(parts[1])),
            let payload = try? JSONDecoder().decode(CodexJWTPayload.self, from: payloadData),
            let accountID = payload.authClaim?.chatgptAccountID,
            !accountID.isEmpty
        else { return nil }
        return accountID
    }
}

// MARK: - CodexTokenExchangeResponse

private struct CodexTokenExchangeResponse: Decodable {
    let accessToken: String
    let refreshToken: String
    let idToken: String?
}

// MARK: - CodexJWTPayload

private struct CodexJWTPayload: Decodable {
    struct AuthClaim: Decodable {
        enum CodingKeys: String, CodingKey {
            case chatgptAccountID = "chatgpt_account_id"
        }

        let chatgptAccountID: String?
    }

    enum CodingKeys: String, CodingKey {
        case authClaim = "https://api.openai.com/auth"
    }

    let authClaim: AuthClaim?
}
