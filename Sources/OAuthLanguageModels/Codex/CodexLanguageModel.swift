import AnyLanguageModel
import Foundation

public let defaultCodexResponsesBaseURL = URL(string: "https://chatgpt.com/backend-api/")!

// MARK: - CodexLanguageModel

/// Talks to ChatGPT / Codex over the undocumented
/// `/backend-api/codex/responses` SSE endpoint, authenticated with a ChatGPT
/// account access token (typically obtained via `CodexOAuthFlow`).
///
/// This single type conforms to **both** language-model protocols:
///
/// - `AnyLanguageModel.LanguageModel` on all supported OS versions
///   (see `CodexLanguageModel+AnyLanguageModel.swift`).
/// - `FoundationModels.LanguageModel` on iOS/macOS/visionOS/watchOS 27+
///   (see `CodexLanguageModel+FoundationModels.swift`).
///
/// ```swift
/// let model = CodexLanguageModel(
///     tokenProvider: { try await myAuth.validToken() },
///     model: "gpt-5"
/// )
/// let session = LanguageModelSession(model: model)
/// ```
public struct CodexLanguageModel: Sendable {
    // MARK: Lifecycle

    public init(
        tokenProvider: @escaping @Sendable () async throws -> CodexToken,
        model: String,
        baseURL: URL = defaultCodexResponsesBaseURL,
        sessionID: String = UUID().uuidString.lowercased(),
        originator: String = "OAuthLanguageModels",
        maxToolRounds: Int = defaultMaxToolRounds,
        onEvent: (@Sendable (GenerationEvent) -> Void)? = nil
    ) {
        self.tokenProvider = tokenProvider
        self.model = model
        self.baseURL = baseURL
        self.sessionID = sessionID
        self.originator = originator
        self.maxToolRounds = maxToolRounds
        self.onEvent = onEvent
        state = CodexSessionState()
    }

    // MARK: Public

    public let tokenProvider: @Sendable () async throws -> CodexToken
    public let model: String
    public let baseURL: URL
    public let sessionID: String
    /// Identifies the client to OpenAI. Sent as the `originator` HTTP header
    /// and audited server-side.
    public let originator: String

    /// How many times a single exchange may hand tool results back before the package
    /// gives up on it.
    ///
    /// A model that answers every tool result with another tool call would otherwise run
    /// until the caller cancelled, spending a turn's tokens each time round.
    public let maxToolRounds: Int

    /// Called with everything a turn produces besides the answer text: the model's
    /// reasoning as it is written, each tool it asks for once the arguments are whole,
    /// and a ``TurnReport`` once per request.
    ///
    /// Fires on the streaming and the non-streaming path alike, and once per round trip
    /// of a tool-using exchange. Called from whichever task is draining the response, so
    /// the closure should be cheap and must not assume a particular actor.
    public let onEvent: (@Sendable (GenerationEvent) -> Void)?

    // MARK: Internal

    /// Top-level request keys callers may not override via `extraBody`.
    static let reservedBodyKeys: Set<String> = [
        "model", "input", "instructions", "tools",
        "prompt_cache_key", "store", "stream", "include"
    ]

    /// Maps `call_id` -> `item_id` so function calls can be replayed with
    /// their original item identifiers across turns in a session.
    let state: CodexSessionState

    /// The turn as it is written: text and reasoning a fragment at a time, each function
    /// call once its arguments are whole, and a terminal part carrying what the turn cost
    /// and why it stopped.
    ///
    /// Not retried. A retry would replay an answer the caller has already been given half
    /// of; a stream that fails mid-flight is the caller's to restart.
    func sendStream(
        inputs: [CodexInputItem],
        instructions: String?,
        tools: [OpenResponsesTool]?,
        parameters: CodexRequestParameters
    ) async throws -> AsyncThrowingStream<CodexStreamPart, any Error> {
        let bytes: URLSession.AsyncBytes
        do {
            bytes = try await openStream(
                inputs: inputs,
                instructions: instructions,
                tools: tools,
                parameters: parameters
            )
        } catch let retryable as RetryableServerError {
            throw CodexLanguageModelError.requestFailed(statusCode: retryable.statusCode, message: retryable.message)
        }

        let onEvent = onEvent
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var parser = CodexStreamParser()
                    for try await line in bytes.lines {
                        guard let payload = Self.payload(in: line) else { continue }
                        for part in try parser.consume(payload: payload) {
                            Self.announce(part, to: onEvent)
                            continuation.yield(part)
                        }
                    }
                    for part in parser.finish() {
                        Self.announce(part, to: onEvent)
                        continuation.yield(part)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// The whole turn, once it is whole. Retried on transient failures, which is safe
    /// only because nothing has been shown to the caller yet.
    func send(
        inputs: [CodexInputItem],
        instructions: String?,
        tools: [OpenResponsesTool]?,
        parameters: CodexRequestParameters
    ) async throws -> CodexStreamingResponse {
        do {
            return try await withNetworkRetry {
                let bytes = try await openStream(
                    inputs: inputs,
                    instructions: instructions,
                    tools: tools,
                    parameters: parameters
                )
                var parser = CodexStreamParser()
                var response: CodexStreamingResponse?
                for try await line in bytes.lines {
                    guard let payload = Self.payload(in: line) else { continue }
                    for part in try parser.consume(payload: payload) {
                        Self.announce(part, to: onEvent)
                        if case let .finished(finished) = part { response = finished }
                    }
                }
                for part in parser.finish() {
                    Self.announce(part, to: onEvent)
                    if case let .finished(finished) = part { response = finished }
                }
                guard let response else { throw CodexLanguageModelError.invalidResponse }
                return response
            }
        } catch let retryable as RetryableServerError {
            throw CodexLanguageModelError.requestFailed(statusCode: retryable.statusCode, message: retryable.message)
        }
    }

    static func makeRequestBody(
        model: String,
        instructions: String,
        inputs: [JSONValue],
        tools: [OpenResponsesTool]?,
        promptCacheKey: String,
        parameters: CodexRequestParameters
    ) -> JSONValue {
        let verbosity = parameters.verbosity ?? "medium"
        let parallel = parameters.parallelToolCalls ?? true

        var textObject: [String: JSONValue] = ["verbosity": .string(verbosity)]
        if let format = parameters.responseFormat {
            textObject["format"] = format
        }

        var body: [String: JSONValue] = [
            "model": .string(model),
            "instructions": .string(instructions),
            "input": .array(inputs),
            "prompt_cache_key": .string(promptCacheKey),
            "store": .bool(false),
            "stream": .bool(true),
            "text": .object(textObject),
            "include": .array([.string("reasoning.encrypted_content")]),
            "parallel_tool_calls": .bool(parallel)
        ]

        if let tools, !tools.isEmpty {
            body["tools"] = .array(tools.map(\.jsonValue))
        }
        if let temperature = parameters.temperature {
            body["temperature"] = .double(temperature)
        }
        if let topP = parameters.topP {
            body["top_p"] = .double(topP)
        }
        if let maxOutput = parameters.maxOutputTokens {
            body["max_output_tokens"] = .int(maxOutput)
        }
        if let maxToolCalls = parameters.maxToolCalls {
            body["max_tool_calls"] = .int(maxToolCalls)
        }

        var reasoningObject: [String: JSONValue] = [:]
        if let effort = parameters.reasoningEffort {
            reasoningObject["effort"] = .string(effort)
        }
        if let summary = parameters.reasoningSummary {
            reasoningObject["summary"] = .string(summary)
        }
        if !reasoningObject.isEmpty {
            body["reasoning"] = .object(reasoningObject)
        }

        body["tool_choice"] = parameters.toolChoice ?? .string("auto")

        for (key, value) in parameters.extraBody ?? [:] {
            guard !reservedBodyKeys.contains(key) else {
                logDropped("body key", name: key, from: "extraBody")
                continue
            }
            body[key] = body[key].map { $0.merging(value) } ?? value
        }

        return .object(body)
    }

    // MARK: Private

    private static var userAgent: String {
        let processInfo = ProcessInfo.processInfo
        let version = processInfo.operatingSystemVersion
        let osName: String
        #if os(macOS)
        osName = "macOS"
        #elseif os(Linux)
        osName = "Linux"
        #else
        osName = "Unknown"
        #endif
        return "OAuthLanguageModels (\(osName) \(version.majorVersion).\(version.minorVersion).\(version.patchVersion))"
    }

    private func resolvedInstructions(_ instructions: String?) -> String {
        guard let instructions, !instructions.isEmpty else {
            return "You are a helpful assistant."
        }
        return instructions
    }

    /// The responses endpoint under whatever the caller gave as a base.
    ///
    /// Built by appending path components rather than by pasting strings together and
    /// parsing them again: appending cannot fail, so a base URL the caller chose can
    /// never take the process down.
    private func resolveCodexURL(from baseURL: URL) -> URL {
        let path = baseURL.path.replacingOccurrences(of: #"/+$"#, with: "", options: .regularExpression)
        if path.hasSuffix("/codex/responses") { return baseURL }
        if path.hasSuffix("/codex") { return baseURL.appendingPathComponent("responses") }
        return baseURL.appendingPathComponent("codex").appendingPathComponent("responses")
    }

    /// Sends the request and hands back the response body, having already turned a
    /// non-2xx into the error its body describes. Retryable statuses come back as
    /// ``RetryableServerError`` so the non-streaming path can act on them.
    private func openStream(
        inputs: [CodexInputItem],
        instructions: String?,
        tools: [OpenResponsesTool]?,
        parameters: CodexRequestParameters
    ) async throws -> URLSession.AsyncBytes {
        let token = try await tokenProvider()

        var request = URLRequest(url: resolveCodexURL(from: baseURL))
        request.httpMethod = "POST"
        request.timeoutInterval = defaultLLMRequestTimeout
        request.setValue("Bearer \(token.accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue(token.accountID, forHTTPHeaderField: "chatgpt-account-id")
        request.setValue(originator, forHTTPHeaderField: "originator")
        request.setValue("responses=experimental", forHTTPHeaderField: "OpenAI-Beta")
        request.setValue("text/event-stream", forHTTPHeaderField: "accept")
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue(sessionID, forHTTPHeaderField: "session_id")
        request.setValue(sessionID, forHTTPHeaderField: "x-client-request-id")
        request.setValue(Self.userAgent, forHTTPHeaderField: "User-Agent")
        request.httpBody = try JSONEncoder.deterministic.encode(
            Self.makeRequestBody(
                model: model,
                instructions: resolvedInstructions(instructions),
                inputs: inputs.map(\.json),
                tools: tools,
                promptCacheKey: sessionID,
                parameters: parameters
            )
        )

        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CodexLanguageModelError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let message = String(decoding: try await Self.collect(bytes), as: UTF8.self)
            if isRetryableHTTPStatus(httpResponse.statusCode) {
                throw RetryableServerError(statusCode: httpResponse.statusCode, message: message)
            }
            throw CodexLanguageModelError.requestFailed(statusCode: httpResponse.statusCode, message: message)
        }
        return bytes
    }

    /// The JSON an SSE line carries, or nothing for the `event:` names and the blank
    /// separators between events.
    private static func payload(in line: String) -> String? {
        guard line.hasPrefix("data:") else { return nil }
        return String(line.dropFirst("data:".count)).trimmingCharacters(in: .whitespaces)
    }

    private static func announce(
        _ part: CodexStreamPart,
        to onEvent: (@Sendable (GenerationEvent) -> Void)?
    ) {
        guard let onEvent else { return }
        switch part {
        case let .reasoning(delta):
            onEvent(.reasoning(delta))
        case let .toolCall(call):
            // The parser hands a call over once, when its item closes, so this is one
            // event per call however many fragments its arguments arrived in — and the
            // sweep of the final output array does not repeat one already seen.
            onEvent(.toolCall(name: call.name, arguments: call.argumentsJSON))
        case let .finished(response):
            onEvent(.turnFinished(response.report))
        case .text:
            break
        }
    }

    private static func collect(_ bytes: URLSession.AsyncBytes) async throws -> Data {
        var data = Data()
        for try await byte in bytes {
            data.append(byte)
        }
        return data
    }

}

// MARK: - CodexRequestParameters

/// Framework-agnostic per-request knobs. Each adapter maps its own options
/// into this shape.
struct CodexRequestParameters {
    var temperature: Double?
    var topP: Double?
    var maxOutputTokens: Int?
    var maxToolCalls: Int?
    var verbosity: String?
    var parallelToolCalls: Bool?
    var reasoningEffort: String?
    var reasoningSummary: String?
    /// Pre-rendered `tool_choice` value (defaults to `"auto"` when nil).
    var toolChoice: JSONValue?
    /// The JSON Schema the answer has to match, as `text.format`. Nil for a plain-text
    /// turn.
    var responseFormat: JSONValue?
    var extraBody: [String: JSONValue]?
}

// MARK: - CodexSessionState

actor CodexSessionState {
    // MARK: Internal

    func remember(_ calls: [CodexToolCall]) {
        for call in calls {
            if let itemID = call.itemID {
                itemIDsByCallID[call.id] = itemID
            }
        }
    }

    func itemID(for callID: String) -> String? {
        itemIDsByCallID[callID]
    }

    // MARK: Private

    private var itemIDsByCallID: [String: String] = [:]
}

// MARK: - CodexToolCall

/// Framework-agnostic representation of a model-issued tool call.
struct CodexToolCall {
    let id: String
    let itemID: String?
    let name: String
    /// Raw JSON-object arguments string.
    let argumentsJSON: String
}

// MARK: - CodexInputItem

/// Opaque wrapper around a single Responses-API `input` item so adapters can
/// build request inputs without depending on the JSON representation.
struct CodexInputItem {
    let json: JSONValue
}

extension CodexInputItem {
    static func userMessage(textSegments: [String], imageURLs: [String]) -> CodexInputItem {
        var content: [JSONValue] = textSegments.map {
            .object(["type": .string("input_text"), "text": .string($0)])
        }
        content.append(contentsOf: imageURLs.map {
            .object(["type": .string("input_image"), "image_url": .string($0)])
        })
        return CodexInputItem(json: .object([
            "type": .string("message"),
            "role": .string("user"),
            "content": .array(content)
        ]))
    }

    static func assistantMessage(textSegments: [String]) -> CodexInputItem {
        let content: [JSONValue] = textSegments.map {
            .object(["type": .string("output_text"), "text": .string($0)])
        }
        return CodexInputItem(json: .object([
            "type": .string("message"),
            "role": .string("assistant"),
            "content": .array(content)
        ]))
    }

    static func functionCall(itemID: String, callID: String, name: String, argumentsJSON: String) -> CodexInputItem {
        CodexInputItem(json: .object([
            "id": .string(itemID),
            "type": .string("function_call"),
            "call_id": .string(callID),
            "name": .string(name),
            "arguments": .string(argumentsJSON)
        ]))
    }

    static func functionCallOutput(callID: String, output: String) -> CodexInputItem {
        CodexInputItem(json: .object([
            "type": .string("function_call_output"),
            "call_id": .string(callID),
            "output": .string(output)
        ]))
    }

    static func functionCalls(for calls: [CodexToolCall]) -> [CodexInputItem] {
        calls.map {
            .functionCall(itemID: $0.itemID ?? $0.id, callID: $0.id, name: $0.name, argumentsJSON: $0.argumentsJSON)
        }
    }
}

// MARK: - OpenResponsesTool

struct OpenResponsesTool {
    let type: String = "function"
    let name: String
    let description: String
    let parameters: JSONValue?

    var jsonValue: JSONValue {
        var object: [String: JSONValue] = [
            "type": .string(type),
            "name": .string(name),
            "description": .string(description)
        ]
        if let parameters {
            object["parameters"] = parameters
        }
        return .object(object)
    }
}

/// Builds an `OpenResponsesTool` from any encodable schema.
///
/// Throws where the schema will not encode. Sending the tool without its parameters
/// instead would offer the model a tool it cannot call correctly, and nothing would say
/// why the calls come back wrong.
func makeOpenResponsesTool(name: String, description: String, schema: some Encodable) throws -> OpenResponsesTool {
    let parameters = try providerToolSchemaJSONValue(forEncodableSchema: schema)
    return OpenResponsesTool(name: name, description: description, parameters: parameters)
}

// MARK: - CodexStreamingResponse

struct CodexStreamingResponse: Sendable {
    /// Final assistant text (accumulated deltas or extracted output).
    let text: String?
    /// Whether the response carried an `output` array at all.
    let hasOutput: Bool
    let toolCalls: [CodexToolCall]
    /// Encrypted `reasoning` items to replay in subsequent requests.
    let reasoningItems: [CodexInputItem]
    let report: TurnReport
}

// MARK: - CodexLanguageModelError

/// Errors thrown by `CodexLanguageModel` while talking to the Codex backend.
public enum CodexLanguageModelError: LocalizedError, Sendable {
    case invalidResponse
    case requestFailed(statusCode: Int, message: String)
    case unsupportedContentType
    case noResponseGenerated
    case toolLoopLimitExceeded(rounds: Int)

    // MARK: Public

    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            "Codex returned an invalid response."
        case let .requestFailed(statusCode, message):
            "Codex request failed with status \(statusCode): \(message)"
        case .unsupportedContentType:
            "CodexLanguageModel only supports text responses."
        case .noResponseGenerated:
            "Codex did not produce any text or tool calls."
        case let .toolLoopLimitExceeded(rounds):
            "Codex kept calling tools after \(rounds) rounds without answering."
        }
    }
}
