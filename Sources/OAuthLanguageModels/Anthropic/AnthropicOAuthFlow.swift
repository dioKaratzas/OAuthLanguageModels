import Foundation

// MARK: - Configuration

/// Claude Code's OAuth client id.
private let anthropicOAuthClientID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
private let anthropicAuthorizeURL = URL(string: "https://claude.ai/oauth/authorize")!
private let anthropicTokenURL = URL(string: "https://platform.claude.com/v1/oauth/token")!
private let anthropicCallbackPort: UInt16 = 53692
private let anthropicCallbackPath = "/callback"
private let anthropicRedirectURI = "http://localhost:\(anthropicCallbackPort)\(anthropicCallbackPath)"
private let anthropicScopes = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
private let anthropicCallbackTimeoutSeconds: TimeInterval = 300
private let anthropicExpiresLeewayMilliseconds: Double = 60_000

// MARK: - AnthropicOAuthError

public enum AnthropicOAuthError: LocalizedError, Sendable {
    case tokenExchangeFailed(status: Int, message: String)
    case invalidTokenResponse

    // MARK: Public

    public var errorDescription: String? {
        switch self {
        case let .tokenExchangeFailed(status, message):
            "Anthropic token exchange failed with status \(status): \(message)"
        case .invalidTokenResponse:
            "Anthropic token endpoint returned an invalid response."
        }
    }
}

// MARK: - AnthropicPendingLogin

/// A login attempt in progress. Spin one up with `AnthropicOAuthFlow.login()`,
/// direct the user to `authorizationURL` (e.g. open it in a browser, render a
/// button, display a QR code), then `await pending.waitForCallback()` to
/// receive the resulting `AnthropicAuth`.
public struct AnthropicPendingLogin: Sendable {
    /// The URL the user must visit to authorize the application.
    public let authorizationURL: URL

    let server: OAuthCallbackServer
    let state: String
    let verifier: String

    /// Waits for the loopback callback server to receive the authorization
    /// code, exchanges it for tokens, and returns the resulting auth. The
    /// underlying server is stopped before this method returns.
    public func waitForCallback() async throws -> AnthropicAuth {
        let server = server
        defer { Task.detached { await server.stop() } }
        let code = try await server.waitForCode(timeout: anthropicCallbackTimeoutSeconds)
        return try await AnthropicOAuthFlow.exchangeAuthorizationCode(
            code: code,
            state: state,
            verifier: verifier
        )
    }

    /// Stops the loopback callback server without waiting for a callback.
    /// Call this if the user cancels the flow before completing it.
    public func cancel() async {
        await server.stop()
    }
}

// MARK: - AnthropicOAuthFlow

/// Drives the Anthropic / Claude OAuth login flow: spins up a loopback
/// callback server, builds the authorization URL, and (once the user has
/// completed the browser flow) exchanges the resulting authorization code
/// for tokens. Callers are responsible for surfacing the authorization URL
/// to the user however they see fit.
public enum AnthropicOAuthFlow {
    /// Starts a login attempt. Returns a `AnthropicPendingLogin` containing
    /// the URL the user must visit; `await pending.waitForCallback()` to
    /// receive the resulting `AnthropicAuth` once they complete the flow.
    public static func login() async throws -> AnthropicPendingLogin {
        // Anthropic's OAuth flow uses the PKCE verifier as the `state` value.
        let pkce = PKCE.generate()
        let state = pkce.verifier

        let server = try await OAuthCallbackServer.start(
            port: anthropicCallbackPort,
            path: anthropicCallbackPath,
            expectedState: state
        )

        let authorizeURL = buildAuthorizationURL(challenge: pkce.challenge, state: state)
        return AnthropicPendingLogin(
            authorizationURL: authorizeURL,
            server: server,
            state: state,
            verifier: pkce.verifier
        )
    }

    public static func buildAuthorizationURL(challenge: String, state: String) -> URL {
        var components = URLComponents(url: anthropicAuthorizeURL, resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "code", value: "true"),
            URLQueryItem(name: "client_id", value: anthropicOAuthClientID),
            URLQueryItem(name: "response_type", value: "code"),
            URLQueryItem(name: "redirect_uri", value: anthropicRedirectURI),
            URLQueryItem(name: "scope", value: anthropicScopes),
            URLQueryItem(name: "code_challenge", value: challenge),
            URLQueryItem(name: "code_challenge_method", value: "S256"),
            URLQueryItem(name: "state", value: state),
        ]
        return components.url!
    }

    public static func exchangeAuthorizationCode(code: String, state: String, verifier: String) async throws -> AnthropicAuth {
        var request = URLRequest(url: anthropicTokenURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let body = AnthropicTokenExchangeRequest(
            grantType: "authorization_code",
            clientId: anthropicOAuthClientID,
            code: code,
            state: state,
            redirectUri: anthropicRedirectURI,
            codeVerifier: verifier
        )
        request.httpBody = try JSONEncoder.snakeCase.encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw AnthropicOAuthError.invalidTokenResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            throw AnthropicOAuthError.tokenExchangeFailed(
                status: http.statusCode,
                message: String(decoding: data, as: UTF8.self)
            )
        }

        let payload: AnthropicTokenExchangeResponse
        do {
            payload = try JSONDecoder.snakeCase.decode(AnthropicTokenExchangeResponse.self, from: data)
        } catch {
            throw AnthropicOAuthError.invalidTokenResponse
        }

        let expiresAt = Date().timeIntervalSince1970 * 1000 + payload.expiresIn * 1000 - anthropicExpiresLeewayMilliseconds
        return AnthropicAuth(
            accessToken: payload.accessToken,
            refreshToken: payload.refreshToken,
            expiresAt: expiresAt
        )
    }
}

// MARK: - AnthropicTokenExchangeRequest

private struct AnthropicTokenExchangeRequest: Encodable {
    let grantType: String
    let clientId: String
    let code: String
    let state: String
    let redirectUri: String
    let codeVerifier: String
}

// MARK: - AnthropicTokenExchangeResponse

private struct AnthropicTokenExchangeResponse: Decodable {
    let accessToken: String
    let refreshToken: String
    let expiresIn: Double
}
