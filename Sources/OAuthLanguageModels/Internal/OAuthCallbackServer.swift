import Foundation
import Network

// MARK: - OAuthCallbackError

/// Errors thrown by the loopback OAuth callback server.
public enum OAuthCallbackError: LocalizedError, Sendable {
    case listenerFailed(String)
    case timeout
    case stateMismatch
    case missingCode
    case serverError(String)

    // MARK: Public

    public var errorDescription: String? {
        switch self {
        case let .listenerFailed(message):
            "Could not start local OAuth callback server: \(message)"
        case .timeout:
            "Timed out waiting for the OAuth callback. Please try again."
        case .stateMismatch:
            "OAuth state mismatch. Please try logging in again."
        case .missingCode:
            "OAuth callback did not include an authorization code."
        case let .serverError(message):
            "OAuth callback server error: \(message)"
        }
    }
}

// MARK: - OAuthCallbackServer

/// Minimal HTTP/1.1 server bound to the loopback interface. Captures a single
/// OAuth redirect request on the supplied port + path, validates `state`, and
/// returns the authorization `code` to the caller.
actor OAuthCallbackServer {
    // MARK: Lifecycle

    private init(listener: NWListener, expectedState: String, path: String) {
        self.listener = listener
        self.expectedState = expectedState
        self.path = path
    }

    // MARK: Internal

    static func start(port: UInt16, path: String, expectedState: String) async throws -> OAuthCallbackServer {
        let params = NWParameters.tcp
        params.requiredInterfaceType = .loopback
        guard let endpointPort = NWEndpoint.Port(rawValue: port) else {
            throw OAuthCallbackError.listenerFailed("Invalid port \(port).")
        }
        let listener: NWListener
        do {
            listener = try NWListener(using: params, on: endpointPort)
        } catch {
            throw OAuthCallbackError.listenerFailed(error.localizedDescription)
        }

        let server = OAuthCallbackServer(listener: listener, expectedState: expectedState, path: path)
        try await server.startAndWait()
        return server
    }

    func waitForCode(timeout seconds: TimeInterval) async throws -> String {
        if let result { return try result.get() }
        return try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask { [weak self] in
                try await withCheckedThrowingContinuation { cont in
                    Task { await self?.register(continuation: cont) }
                }
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw OAuthCallbackError.timeout
            }
            let value = try await group.next()!
            group.cancelAll()
            return value
        }
    }

    func stop() {
        guard !stopped else { return }
        stopped = true
        listener.cancel()
    }

    // MARK: Private

    private let listener: NWListener
    private let expectedState: String
    private let path: String
    private var result: Result<String, Error>?
    private var waiters: [CheckedContinuation<String, Error>] = []
    private var readyWaiters: [CheckedContinuation<Void, Error>] = []
    private var isReady = false
    private var didSettleReady = false
    private var stopped = false

    private func startAndWait() async throws {
        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            Task { await self.stateChanged(state) }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            Task { await self.handle(connection: connection) }
        }
        listener.start(queue: .global(qos: .userInitiated))

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            if isReady { cont.resume(); return }
            readyWaiters.append(cont)
        }
    }

    private func stateChanged(_ state: NWListener.State) {
        switch state {
        case .ready:
            guard !didSettleReady else { return }
            didSettleReady = true
            isReady = true
            for w in readyWaiters {
                w.resume()
            }
            readyWaiters.removeAll()
        case let .failed(error):
            let mapped = OAuthCallbackError.listenerFailed(error.localizedDescription)
            if !didSettleReady {
                didSettleReady = true
                for w in readyWaiters {
                    w.resume(throwing: mapped)
                }
                readyWaiters.removeAll()
            }
            settle(.failure(mapped))
        default:
            break
        }
    }

    private func register(continuation: CheckedContinuation<String, Error>) {
        if let result {
            continuation.resume(with: result)
            return
        }
        waiters.append(continuation)
    }

    private func settle(_ result: Result<String, Error>) {
        guard self.result == nil else { return }
        self.result = result
        for w in waiters {
            w.resume(with: result)
        }
        waiters.removeAll()
    }

    private func handle(connection: NWConnection) async {
        connection.start(queue: .global(qos: .userInitiated))
        defer { connection.cancel() }

        do {
            let head = try await readRequestHead(from: connection)
            let (requestPath, query) = parseRequestTarget(head)

            if requestPath != path {
                try await respond(on: connection, status: 404, body: oauthErrorHTML("Callback route not found."))
                return
            }

            let params = parseQuery(query)

            if let providerError = params["error"], !providerError.isEmpty {
                try await respond(
                    on: connection,
                    status: 400,
                    body: oauthErrorHTML("Provider returned an error: \(providerError)")
                )
                settle(.failure(OAuthCallbackError.serverError(providerError)))
                return
            }

            let returnedState = params["state"] ?? ""
            guard returnedState == expectedState else {
                try await respond(on: connection, status: 400, body: oauthErrorHTML("State mismatch."))
                settle(.failure(OAuthCallbackError.stateMismatch))
                return
            }

            guard let code = params["code"], !code.isEmpty else {
                try await respond(on: connection, status: 400, body: oauthErrorHTML("Missing authorization code."))
                settle(.failure(OAuthCallbackError.missingCode))
                return
            }

            try await respond(on: connection, status: 200, body: oauthSuccessHTML())
            settle(.success(code))
        } catch {
            // Ignore per-connection errors; other callbacks may still arrive.
        }
    }

    private func readRequestHead(from connection: NWConnection) async throws -> String {
        var buffer = Data()
        let terminator = Data("\r\n\r\n".utf8)
        while buffer.count < 16 * 1024 {
            let chunk = try await receive(on: connection)
            if chunk.isEmpty { break }
            buffer.append(chunk)
            if buffer.range(of: terminator) != nil { break }
        }
        return String(decoding: buffer, as: UTF8.self)
    }

    private func receive(on connection: NWConnection) async throws -> Data {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Data, Error>) in
            connection.receive(minimumIncompleteLength: 1, maximumLength: 8 * 1024) { data, _, _, error in
                if let error { cont.resume(throwing: error) } else { cont.resume(returning: data ?? Data()) }
            }
        }
    }

    private func respond(on connection: NWConnection, status: Int, body: String) async throws {
        let statusText = switch status {
        case 200: "OK"
        case 400: "Bad Request"
        case 404: "Not Found"
        default: "OK"
        }
        let bodyData = Data(body.utf8)
        let head = """
        HTTP/1.1 \(status) \(statusText)\r
        Content-Type: text/html; charset=utf-8\r
        Content-Length: \(bodyData.count)\r
        Connection: close\r
        \r

        """
        var response = Data(head.utf8)
        response.append(bodyData)
        try await send(data: response, on: connection)
    }

    private func send(data: Data, on connection: NWConnection) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if let error { cont.resume(throwing: error) } else { cont.resume() }
            })
        }
    }
}

// MARK: - HTTP parsing

private func parseRequestTarget(_ request: String) -> (path: String, query: String) {
    guard let firstLineEnd = request.range(of: "\r\n") else {
        return ("", "")
    }
    let firstLine = request[..<firstLineEnd.lowerBound]
    let parts = firstLine.split(separator: " ")
    guard parts.count >= 2 else { return ("", "") }
    let target = String(parts[1])
    guard let q = target.firstIndex(of: "?") else {
        return (target, "")
    }
    return (String(target[..<q]), String(target[target.index(after: q)...]))
}

private func parseQuery(_ query: String) -> [String: String] {
    var params: [String: String] = [:]
    for pair in query.split(separator: "&") {
        let kv = pair.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
        let rawKey = String(kv[0])
        let rawValue = kv.count > 1 ? String(kv[1]) : ""
        let key = rawKey.removingPercentEncoding ?? rawKey
        let value = rawValue.removingPercentEncoding ?? rawValue
        params[key] = value
    }
    return params
}

// MARK: - HTML helpers

func oauthSuccessHTML() -> String {
    """
    <!doctype html><html><head><meta charset="utf-8"><title>OAuth login</title>
    <style>body{font-family:system-ui,-apple-system;max-width:520px;margin:80px auto;padding:0 16px;color:#111;text-align:center}h1{font-weight:600}</style>
    </head><body>
    <h1>You're logged in</h1>
    <p>You can close this window and return to your terminal.</p>
    </body></html>
    """
}

func oauthErrorHTML(_ message: String) -> String {
    """
    <!doctype html><html><head><meta charset="utf-8"><title>OAuth login</title>
    <style>body{font-family:system-ui,-apple-system;max-width:520px;margin:80px auto;padding:0 16px;color:#111;text-align:center}h1{font-weight:600;color:#b00020}</style>
    </head><body>
    <h1>Login failed</h1>
    <p>\(message)</p>
    </body></html>
    """
}
