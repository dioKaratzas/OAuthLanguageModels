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
        originator: String = "OAuthLanguageModels"
    ) {
        self.tokenProvider = tokenProvider
        self.model = model
        self.baseURL = baseURL
        self.sessionID = sessionID
        self.originator = originator
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

    // MARK: Internal

    /// Top-level request keys callers may not override via `extraBody`.
    static let reservedBodyKeys: Set<String> = [
        "model", "input", "instructions", "tools",
        "prompt_cache_key", "store", "stream", "include"
    ]

    /// Maps `call_id` -> `item_id` so function calls can be replayed with
    /// their original item identifiers across turns in a session.
    let state: CodexSessionState

    /// Streams a request's text deltas as they arrive, for a plain-text turn with no
    /// tools to reassemble. Each element is the piece that just landed, not the answer
    /// so far.
    func sendStream(
        inputs: [CodexInputItem],
        instructions: String?,
        tools: [OpenResponsesTool]?,
        parameters: CodexRequestParameters
    ) -> AsyncThrowingStream<String, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    _ = try await send(
                        inputs: inputs,
                        instructions: instructions,
                        tools: tools,
                        parameters: parameters,
                        onDelta: { continuation.yield($0) }
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func send(
        inputs: [CodexInputItem],
        instructions: String?,
        tools: [OpenResponsesTool]?,
        parameters: CodexRequestParameters,
        onDelta: (@Sendable (String) -> Void)? = nil
    ) async throws -> CodexStreamingResponse {
        let token = try await tokenProvider()
        let url = resolveCodexURL(from: baseURL)

        var request = URLRequest(url: url)
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

        let requestBody = try JSONEncoder.deterministic.encode(
            Self.makeRequestBody(
                model: model,
                instructions: resolvedInstructions(instructions),
                inputs: inputs.map(\.json),
                tools: tools,
                promptCacheKey: sessionID,
                parameters: parameters
            )
        )
        request.httpBody = requestBody

        do {
            return try await withNetworkRetry {
                let (bytes, response) = try await URLSession.shared.bytes(for: request)
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw CodexLanguageModelError.invalidResponse
                }

                guard (200..<300).contains(httpResponse.statusCode) else {
                    let data = try await collect(bytes)
                    let message = String(decoding: data, as: UTF8.self)
                    if isRetryableHTTPStatus(httpResponse.statusCode) {
                        throw RetryableServerError(statusCode: httpResponse.statusCode, message: message)
                    }
                    throw CodexLanguageModelError.requestFailed(statusCode: httpResponse.statusCode, message: message)
                }

                return try await parseSSE(bytes: bytes, onDelta: onDelta)
            }
        } catch let retryable as RetryableServerError {
            throw CodexLanguageModelError.requestFailed(statusCode: retryable.statusCode, message: retryable.message)
        }
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

    private static func makeRequestBody(
        model: String,
        instructions: String,
        inputs: [JSONValue],
        tools: [OpenResponsesTool]?,
        promptCacheKey: String,
        parameters: CodexRequestParameters
    ) -> JSONValue {
        let verbosity = parameters.verbosity ?? "medium"
        let parallel = parameters.parallelToolCalls ?? true

        var body: [String: JSONValue] = [
            "model": .string(model),
            "instructions": .string(instructions),
            "input": .array(inputs),
            "prompt_cache_key": .string(promptCacheKey),
            "store": .bool(false),
            "stream": .bool(true),
            "text": .object(["verbosity": .string(verbosity)]),
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
            body[key] = value
        }

        return .object(body)
    }

    // MARK: SSE parsing helpers

    static func processEvent(
        _ payload: String,
        accumulatedText: inout String,
        latestOutput: inout [JSONValue]?,
        latestOutputText: inout String?,
        toolCallsByID: inout [String: CodexToolCall],
        onDelta: (@Sendable (String) -> Void)? = nil
    ) throws {
        guard payload != "[DONE]", !payload.isEmpty else { return }
        guard let data = payload.data(using: .utf8) else { return }

        let value: JSONValue
        do {
            value = try JSONDecoder().decode(JSONValue.self, from: data)
        } catch {
            return
        }

        if case let .object(object) = value {
            if let type = object["type"].flatMap({ if case let .string(string) = $0 { string } else { nil } }),
               type == "response.output_text.delta",
               let delta = object["delta"].flatMap({ if case let .string(string) = $0 { string } else { nil } }) {
                accumulatedText += delta
                onDelta?(delta)
            }

            if let outputText = object["output_text"].flatMap({ if case let .string(string) = $0 { string } else { nil } }) {
                latestOutputText = outputText
            }

            if let response = object["response"], case let .object(responseObject) = response {
                if case let .array(output)? = responseObject["output"] {
                    latestOutput = output
                }
                if let outputText = responseObject["output_text"].flatMap({ if case let .string(string) = $0 { string } else { nil } }) {
                    latestOutputText = outputText
                }
            } else if case let .array(output)? = object["output"] {
                latestOutput = output
            }
        }

        var collected: [CodexToolCall] = []
        collectToolCalls(from: value, into: &collected)
        for call in collected {
            toolCallsByID[call.id] = call
        }
    }

    private static func collectToolCalls(from value: JSONValue, into result: inout [CodexToolCall]) {
        switch value {
        case let .object(object):
            let type = object["type"].flatMap {
                if case let .string(string) = $0 { string } else { nil }
            }
            if let type, ["function_call", "tool_call", "tool_use"].contains(type),
               let call = parseToolCall(from: object) {
                result.append(call)
            }
            if let item = object["item"] {
                collectToolCalls(from: item, into: &result)
            }
            if let toolCall = object["tool_call"] {
                collectToolCalls(from: toolCall, into: &result)
            }
            if let content = object["content"] {
                collectToolCalls(from: content, into: &result)
            }
            for (key, value) in object where key != "content" && key != "item" && key != "tool_call" {
                collectToolCalls(from: value, into: &result)
            }
        case let .array(array):
            for item in array {
                collectToolCalls(from: item, into: &result)
            }
        default:
            break
        }
    }

    private static func parseToolCall(from object: [String: JSONValue]) -> CodexToolCall? {
        let itemID = object["id"].flatMap {
            if case let .string(string) = $0 { string } else { nil }
        }
        let callID = object["call_id"].flatMap {
            if case let .string(string) = $0 { string } else { nil }
        } ?? itemID
        let name = object["name"].flatMap {
            if case let .string(string) = $0 { string } else { nil }
        }
        guard let callID, let name, !callID.isEmpty, !name.isEmpty else { return nil }

        let argumentsJSON: String = if let arguments = object["arguments"] {
            switch arguments {
            case let .string(string):
                string
            case let .object(object):
                (try? jsonObjectString(from: object)) ?? "{}"
            default:
                "{}"
            }
        } else {
            "{}"
        }

        return CodexToolCall(id: callID, itemID: itemID, name: name, argumentsJSON: argumentsJSON)
    }

    private static func extractText(from output: [JSONValue]?) -> String? {
        guard let output else { return nil }
        var parts: [String] = []
        for item in output {
            guard case let .object(object) = item,
                  object["type"].flatMap({ if case let .string(string) = $0 { string } else { nil } }) == "message",
                  case let .array(content)? = object["content"] else {
                continue
            }
            for block in content {
                guard case let .object(object) = block,
                      object["type"].flatMap({ if case let .string(string) = $0 { string } else { nil } }) == "output_text",
                      case let .string(text)? = object["text"] else {
                    continue
                }
                parts.append(text)
            }
        }
        return parts.isEmpty ? nil : parts.joined()
    }

    /// Pull `reasoning` items out of the final response output array, in
    /// their original order, to replay across turns when `store: false`.
    private static func extractReasoningItems(from output: [JSONValue]?) -> [JSONValue] {
        guard let output else { return [] }
        return output.compactMap { item in
            guard case let .object(object) = item,
                  case let .string(type)? = object["type"],
                  type == "reasoning" else {
                return nil
            }
            return item
        }
    }

    private func resolvedInstructions(_ instructions: String?) -> String {
        guard let instructions, !instructions.isEmpty else {
            return "You are a helpful assistant."
        }
        return instructions
    }

    private func resolveCodexURL(from baseURL: URL) -> URL {
        let trimmed = baseURL.absoluteString.replacingOccurrences(of: #"/+$"#, with: "", options: .regularExpression)
        if trimmed.hasSuffix("/codex/responses") {
            return URL(string: trimmed)!
        }
        if trimmed.hasSuffix("/codex") {
            return URL(string: trimmed + "/responses")!
        }
        return URL(string: trimmed + "/codex/responses")!
    }

    private func parseSSE(
        bytes: URLSession.AsyncBytes,
        onDelta: (@Sendable (String) -> Void)? = nil
    ) async throws -> CodexStreamingResponse {
        var accumulatedText = ""
        var latestOutput: [JSONValue]?
        var latestOutputText: String?
        var toolCallsByID: [String: CodexToolCall] = [:]

        for try await line in bytes.lines {
            guard line.hasPrefix("data:") else { continue }
            let payload = String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces)
            try Self.processEvent(
                payload,
                accumulatedText: &accumulatedText,
                latestOutput: &latestOutput,
                latestOutputText: &latestOutputText,
                toolCallsByID: &toolCallsByID,
                onDelta: onDelta
            )
        }

        let toolCalls = Array(toolCallsByID.values).sorted { $0.id < $1.id }
        let outputText = accumulatedText.isEmpty ? latestOutputText : accumulatedText
        let text = outputText ?? Self.extractText(from: latestOutput)
        let reasoningItems = Self.extractReasoningItems(from: latestOutput).map { CodexInputItem(json: $0) }
        return CodexStreamingResponse(
            text: text,
            hasOutput: latestOutput != nil,
            toolCalls: toolCalls,
            reasoningItems: reasoningItems
        )
    }

    private func collect(_ bytes: URLSession.AsyncBytes) async throws -> Data {
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
func makeOpenResponsesTool(name: String, description: String, schema: some Encodable) -> OpenResponsesTool {
    let parameters = try? providerToolSchemaJSONValue(forEncodableSchema: schema)
    return OpenResponsesTool(name: name, description: description, parameters: parameters)
}

// MARK: - CodexStreamingResponse

struct CodexStreamingResponse {
    /// Final assistant text (accumulated deltas or extracted output).
    let text: String?
    /// Whether the response carried an `output` array at all.
    let hasOutput: Bool
    let toolCalls: [CodexToolCall]
    /// Encrypted `reasoning` items to replay in subsequent requests.
    let reasoningItems: [CodexInputItem]
}

// MARK: - CodexLanguageModelError

/// Errors thrown by `CodexLanguageModel` while talking to the Codex backend.
public enum CodexLanguageModelError: LocalizedError, Sendable {
    case invalidResponse
    case requestFailed(statusCode: Int, message: String)
    case unsupportedContentType
    case noResponseGenerated

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
        }
    }
}
