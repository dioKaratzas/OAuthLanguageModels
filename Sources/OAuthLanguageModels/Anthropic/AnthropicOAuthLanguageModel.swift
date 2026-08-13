import AnyLanguageModel
import Foundation

public let defaultAnthropicBaseURL = URL(string: "https://api.anthropic.com/")!
private let claudeCodeVersion = "2.1.75"
let claudeCodeSystemPreamble = "You are Claude Code, Anthropic's official CLI for Claude."

// MARK: - AnthropicOAuthLanguageModel

/// Talks to Anthropic's Messages API using a Claude Code OAuth access token
/// (typically obtained via `AnthropicOAuthFlow`).
///
/// The OAuth path requires the request to identify itself as Claude Code,
/// so this model always prefixes the system prompt with the Claude Code
/// preamble and sends the corresponding `user-agent`/`x-app` headers.
///
/// This single type conforms to **both** language-model protocols:
///
/// - `AnyLanguageModel.LanguageModel` on all supported OS versions
///   (see `AnthropicOAuthLanguageModel+AnyLanguageModel.swift`).
/// - `FoundationModels.LanguageModel` on iOS/macOS/visionOS/watchOS 27+
///   (see `AnthropicOAuthLanguageModel+FoundationModels.swift`).
///
/// Drop it into either framework's `LanguageModelSession`:
///
/// ```swift
/// let model = AnthropicOAuthLanguageModel(
///     tokenProvider: { try await myAuth.validAccessToken() },
///     model: "claude-opus-4"
/// )
/// let session = LanguageModelSession(model: model)
/// ```
public struct AnthropicOAuthLanguageModel: Sendable {
    // MARK: Lifecycle

    public init(
        tokenProvider: @escaping @Sendable () async throws -> String,
        model: String,
        baseURL: URL = defaultAnthropicBaseURL,
        maxTokens: Int = 4096,
        longCacheRetention: Bool = false,
        extraBetas: [String] = [],
        extraHeaders: [String: String] = [:]
    ) {
        self.tokenProvider = tokenProvider
        self.model = model
        self.baseURL = baseURL
        self.maxTokens = maxTokens
        self.longCacheRetention = longCacheRetention
        self.extraBetas = extraBetas
        self.extraHeaders = extraHeaders
    }

    // MARK: Public

    public let tokenProvider: @Sendable () async throws -> String
    public let model: String
    public let baseURL: URL
    public let maxTokens: Int
    public let longCacheRetention: Bool

    /// Beta names appended to the `anthropic-beta` header, after the ones the
    /// OAuth/Claude Code request shape requires.
    ///
    /// Features that are opted into by header rather than by body key need this:
    /// fast mode, for one, is `speed: "fast"` in `extraBody` *and*
    /// `fast-mode-2026-02-01` here — either half alone is refused.
    public let extraBetas: [String]

    /// Extra header fields to set on every request.
    ///
    /// Fields the OAuth request shape depends on are dropped, the same way
    /// ``reservedBodyKeys`` protects the body: they are what the API honours a
    /// subscription token for, and overriding them only produces a 401.
    public let extraHeaders: [String: String]

    // MARK: Internal

    /// Top-level body keys that callers may not override via `extraBody`.
    /// These are required for the OAuth/Claude Code request shape.
    static let reservedBodyKeys: Set<String> = [
        "model", "system", "messages", "tools"
    ]

    /// Header fields that callers may not override via `extraHeaders`, lowercased.
    static let reservedHeaderFields: Set<String> = [
        "authorization", "anthropic-version", "anthropic-beta", "user-agent",
        "x-app", "accept", "content-type"
    ]

    /// The betas the OAuth path requires, plus whatever the caller added.
    var betaHeaderValue: String {
        (["claude-code-20250219", "oauth-2025-04-20"] + extraBetas).joined(separator: ",")
    }

    var cacheControl: AnthropicRequest.CacheControl {
        longCacheRetention ? .ephemeralLong : .ephemeral
    }

    /// Encode the request body and merge any caller-supplied `extraBody`
    /// keys, dropping reserved keys to preserve the OAuth request shape.
    static func encodeBody(
        _ body: AnthropicRequest,
        mergingExtraBody extra: [String: JSONValue]?
    ) throws -> Data {
        let baseData = try JSONEncoder.snakeCase.encode(body)
        guard let extra, !extra.isEmpty else { return baseData }

        let value = try JSONDecoder().decode(JSONValue.self, from: baseData)
        guard case var .object(object) = value else { return baseData }
        for (key, value) in extra where !reservedBodyKeys.contains(key) {
            object[key] = value
        }
        return try JSONEncoder.deterministic.encode(JSONValue.object(object))
    }

    func makeRequest(
        streaming: Bool,
        messages: [AnthropicRequest.Message],
        instructions: String?,
        tools: [AnthropicTool]?,
        parameters: AnthropicRequestParameters
    ) async throws -> URLRequest {
        let accessToken = try await tokenProvider()

        var url = baseURL
        if !url.path.hasSuffix("/") {
            url = url.appendingPathComponent("")
        }
        url = url.appendingPathComponent("v1/messages")

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = defaultLLMRequestTimeout
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        request.setValue(betaHeaderValue, forHTTPHeaderField: "anthropic-beta")
        request.setValue("true", forHTTPHeaderField: "anthropic-dangerous-direct-browser-access")
        request.setValue("claude-cli/\(claudeCodeVersion)", forHTTPHeaderField: "user-agent")
        request.setValue("cli", forHTTPHeaderField: "x-app")
        request.setValue(streaming ? "text/event-stream" : "application/json", forHTTPHeaderField: "accept")
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        for (field, value) in extraHeaders where !Self.reservedHeaderFields.contains(field.lowercased()) {
            request.setValue(value, forHTTPHeaderField: field)
        }

        var system = [AnthropicRequest.TextBlock(text: claudeCodeSystemPreamble)]
        if let instructions, !instructions.isEmpty {
            system.append(AnthropicRequest.TextBlock(text: instructions))
        }
        if !system.isEmpty {
            system[system.count - 1].cacheControl = cacheControl
        }

        var cachedMessages = messages
        if !cachedMessages.isEmpty {
            cachedMessages[cachedMessages.count - 1].markLastBlockCached(with: cacheControl)
        }

        let body = AnthropicRequest(
            model: model,
            maxTokens: parameters.maxTokens ?? maxTokens,
            system: system,
            messages: cachedMessages,
            tools: tools,
            stream: streaming,
            temperature: parameters.temperature,
            topP: parameters.topP,
            topK: parameters.topK,
            stopSequences: parameters.stopSequences,
            toolChoice: parameters.toolChoice,
            thinking: parameters.thinkingBudgetTokens.map { .init(budgetTokens: $0) }
        )
        request.httpBody = try Self.encodeBody(body, mergingExtraBody: parameters.extraBody)
        return request
    }

    func send(
        messages: [AnthropicRequest.Message],
        instructions: String?,
        tools: [AnthropicTool]?,
        parameters: AnthropicRequestParameters
    ) async throws -> AnthropicResponse {
        let request = try await makeRequest(
            streaming: false,
            messages: messages,
            instructions: instructions,
            tools: tools,
            parameters: parameters
        )

        do {
            return try await withNetworkRetry {
                let (data, response) = try await URLSession.shared.data(for: request)
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw AnthropicOAuthLanguageModelError.invalidResponse
                }

                guard (200..<300).contains(httpResponse.statusCode) else {
                    let message = String(decoding: data, as: UTF8.self)
                    if isRetryableHTTPStatus(httpResponse.statusCode) {
                        throw RetryableServerError(statusCode: httpResponse.statusCode, message: message)
                    }
                    throw AnthropicOAuthLanguageModelError.requestFailed(statusCode: httpResponse.statusCode, message: message)
                }

                do {
                    return try JSONDecoder.snakeCase.decode(AnthropicResponse.self, from: data)
                } catch {
                    throw AnthropicOAuthLanguageModelError.invalidResponse
                }
            }
        } catch let retryable as RetryableServerError {
            throw AnthropicOAuthLanguageModelError.requestFailed(statusCode: retryable.statusCode, message: retryable.message)
        }
    }

    /// The answer as it is written, one text delta at a time.
    ///
    /// Text only: a stream carrying tool calls has to reassemble each call from its
    /// `input_json_delta` fragments before anything can be run, which is the
    /// non-streaming path's job. Callers with tools in play use ``send(messages:instructions:tools:parameters:)``.
    ///
    /// Not retried. A retry would replay an answer the caller has already been given
    /// half of; a stream that fails mid-flight is the caller's to restart.
    func sendStream(
        messages: [AnthropicRequest.Message],
        instructions: String?,
        parameters: AnthropicRequestParameters
    ) async throws -> AsyncThrowingStream<String, any Error> {
        let request = try await makeRequest(
            streaming: true,
            messages: messages,
            instructions: instructions,
            tools: nil,
            parameters: parameters
        )
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        // Before the stream rather than inside it: a refusal is the whole answer, and
        // the caller should see it thrown rather than delivered as an empty stream.
        try await Self.checkStreamResponse(response, bytes: bytes)

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await line in bytes.lines {
                        if let delta = try Self.textDelta(in: line) {
                            continuation.yield(delta)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// The refusal arrives as the body of a non-2xx, which has to be read off the
    /// byte stream before it can be reported.
    private static func checkStreamResponse(
        _ response: URLResponse,
        bytes: URLSession.AsyncBytes
    ) async throws {
        guard let http = response as? HTTPURLResponse else {
            throw AnthropicOAuthLanguageModelError.invalidResponse
        }
        guard !(200..<300).contains(http.statusCode) else { return }

        var message = ""
        for try await line in bytes.lines where message.count < 4096 {
            message += line
        }
        throw AnthropicOAuthLanguageModelError.requestFailed(statusCode: http.statusCode, message: message)
    }

    /// The text an event carries, or nothing for the events that carry none — pings,
    /// block boundaries, the usage totals at the end.
    static func textDelta(in line: String) throws -> String? {
        guard line.hasPrefix("data:") else { return nil }
        let json = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
        guard let event = try? JSONDecoder.snakeCase.decode(AnthropicStreamEvent.self, from: Data(json.utf8)) else {
            return nil
        }
        switch event.type {
        case "content_block_delta":
            guard event.delta?.type == "text_delta" else { return nil }
            return event.delta?.text
        case "error":
            // Mid-stream failures arrive as an event, with a 200 already on the wire.
            throw AnthropicOAuthLanguageModelError.requestFailed(
                statusCode: 200,
                message: event.error?.message ?? "Anthropic ended the stream with an error."
            )
        default:
            return nil
        }
    }
}

// MARK: - AnthropicStreamEvent

struct AnthropicStreamEvent: Decodable {
    struct Delta: Decodable {
        let type: String?
        let text: String?
    }

    struct StreamError: Decodable {
        let type: String?
        let message: String?
    }

    let type: String
    var delta: Delta?
    var error: StreamError?
}

// MARK: - AnthropicRequestParameters

/// Framework-agnostic per-request knobs. Each adapter (AnyLanguageModel /
/// FoundationModels) maps its own options into this shape.
struct AnthropicRequestParameters {
    var temperature: Double?
    var maxTokens: Int?
    var topP: Double?
    var topK: Int?
    var stopSequences: [String]?
    var toolChoice: AnthropicRequest.ToolChoice?
    var thinkingBudgetTokens: Int?
    var extraBody: [String: JSONValue]?
}

// MARK: - AnthropicRequest

struct AnthropicRequest: Encodable {
    struct CacheControl: Codable, Equatable {
        static let ephemeral = CacheControl(type: "ephemeral")
        static let ephemeralLong = CacheControl(type: "ephemeral", ttl: "1h")

        let type: String
        var ttl: String?
    }

    struct TextBlock: Encodable {
        // MARK: Lifecycle

        init(type: String = "text", text: String, cacheControl: CacheControl? = nil) {
            self.type = type
            self.text = text
            self.cacheControl = cacheControl
        }

        // MARK: Internal

        let type: String
        let text: String
        var cacheControl: CacheControl?
    }

    struct Message: Encodable {
        let role: String
        var content: [AnthropicResponse.ContentBlock]

        /// Puts the cache breakpoint on the last block that can hold one.
        ///
        /// Every block type but thinking accepts `cache_control`; a thinking block is
        /// refused one and is cached implicitly with the turn around it, so the walk
        /// steps back past it rather than leaving the whole conversation behind this
        /// message to be re-read at full price on the next turn.
        mutating func markLastBlockCached(with cacheControl: CacheControl) {
            for index in content.indices.reversed() {
                switch content[index] {
                case var .text(text):
                    text.cacheControl = cacheControl
                    content[index] = .text(text)
                case var .image(image):
                    image.cacheControl = cacheControl
                    content[index] = .image(image)
                case var .toolUse(toolUse):
                    toolUse.cacheControl = cacheControl
                    content[index] = .toolUse(toolUse)
                case var .toolResult(toolResult):
                    toolResult.cacheControl = cacheControl
                    content[index] = .toolResult(toolResult)
                case .thinking, .redactedThinking:
                    continue
                }
                return
            }
        }
    }

    enum ToolChoice: Encodable {
        case auto
        case any
        case tool(name: String)
        case disabled

        // MARK: Internal

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .auto: try container.encode("auto", forKey: .type)
            case .any: try container.encode("any", forKey: .type)
            case let .tool(name):
                try container.encode("tool", forKey: .type)
                try container.encode(name, forKey: .name)
            case .disabled: try container.encode("none", forKey: .type)
            }
        }

        // MARK: Private

        private enum CodingKeys: String, CodingKey { case type, name }
    }

    struct Thinking: Encodable {
        // MARK: Lifecycle

        init(budgetTokens: Int) {
            type = "enabled"
            self.budgetTokens = budgetTokens
        }

        // MARK: Internal

        let type: String
        let budgetTokens: Int
    }

    let model: String
    let maxTokens: Int
    var system: [TextBlock]
    let messages: [Message]
    let tools: [AnthropicTool]?
    var stream = false
    var temperature: Double?
    var topP: Double?
    var topK: Int?
    var stopSequences: [String]?
    var toolChoice: ToolChoice?
    var thinking: Thinking?
}

// MARK: - AnthropicTool

struct AnthropicTool: Codable {
    enum CodingKeys: String, CodingKey {
        case name
        case description
        case inputSchema = "input_schema"
    }

    let name: String
    let description: String
    let inputSchema: JSONValue
}

// MARK: - AnthropicResponse

struct AnthropicResponse: Decodable {
    enum ContentBlock: Decodable, Encodable {
        case text(Text)
        case image(Image)
        case toolUse(ToolUse)
        case toolResult(ToolResult)
        case thinking(Thinking)
        case redactedThinking(RedactedThinking)

        // MARK: Lifecycle

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            switch try container.decode(ContentType.self, forKey: .type) {
            case .text:
                self = try .text(Text(from: decoder))
            case .image:
                self = try .image(Image(from: decoder))
            case .toolUse:
                self = try .toolUse(ToolUse(from: decoder))
            case .toolResult:
                self = try .toolResult(ToolResult(from: decoder))
            case .thinking:
                self = try .thinking(Thinking(from: decoder))
            case .redactedThinking:
                self = try .redactedThinking(RedactedThinking(from: decoder))
            }
        }

        // MARK: Internal

        enum CodingKeys: String, CodingKey { case type }
        enum ContentType: String, Codable {
            case text, image
            case toolUse = "tool_use"
            case toolResult = "tool_result"
            case thinking
            case redactedThinking = "redacted_thinking"
        }

        func encode(to encoder: any Encoder) throws {
            switch self {
            case let .text(value): try value.encode(to: encoder)
            case let .image(value): try value.encode(to: encoder)
            case let .toolUse(value): try value.encode(to: encoder)
            case let .toolResult(value): try value.encode(to: encoder)
            case let .thinking(value): try value.encode(to: encoder)
            case let .redactedThinking(value): try value.encode(to: encoder)
            }
        }
    }

    struct Thinking: Codable {
        // MARK: Lifecycle

        init(thinking: String, signature: String?) {
            type = "thinking"
            self.thinking = thinking
            self.signature = signature
        }

        // MARK: Internal

        let type: String
        let thinking: String
        let signature: String?
    }

    struct RedactedThinking: Codable {
        // MARK: Lifecycle

        init(data: String) {
            type = "redacted_thinking"
            self.data = data
        }

        // MARK: Internal

        let type: String
        let data: String
    }

    struct Text: Codable {
        // MARK: Lifecycle

        init(text: String, cacheControl: AnthropicRequest.CacheControl? = nil) {
            type = "text"
            self.text = text
            self.cacheControl = cacheControl
        }

        // MARK: Internal

        let type: String
        let text: String
        var cacheControl: AnthropicRequest.CacheControl?
    }

    struct Image: Codable {
        // MARK: Lifecycle

        init(base64Data: String, mimeType: String) {
            type = "image"
            source = .init(type: "base64", mediaType: mimeType, data: base64Data, url: nil)
        }

        init(url: String) {
            type = "image"
            source = .init(type: "url", mediaType: nil, data: nil, url: url)
        }

        // MARK: Internal

        struct Source: Codable {
            enum CodingKeys: String, CodingKey {
                case type
                case mediaType = "media_type"
                case data
                case url
            }

            let type: String
            let mediaType: String?
            let data: String?
            let url: String?
        }

        let type: String
        let source: Source
        var cacheControl: AnthropicRequest.CacheControl?
    }

    struct ToolUse: Codable {
        // MARK: Lifecycle

        init(id: String, name: String, input: [String: JSONValue]?) {
            type = "tool_use"
            self.id = id
            self.name = name
            self.input = input
        }

        // MARK: Internal

        let type: String
        let id: String
        let name: String
        let input: [String: JSONValue]?
        var cacheControl: AnthropicRequest.CacheControl?
    }

    struct ToolResult: Codable {
        // MARK: Lifecycle

        init(toolUseID: String, content: [ContentBlock]) {
            type = "tool_result"
            self.toolUseID = toolUseID
            self.content = content
        }

        // MARK: Internal

        enum CodingKeys: String, CodingKey {
            case type
            case toolUseID = "tool_use_id"
            case content
            case cacheControl = "cache_control"
        }

        let type: String
        let toolUseID: String
        let content: [ContentBlock]
        var cacheControl: AnthropicRequest.CacheControl?
    }

    let content: [ContentBlock]
}

// MARK: - JSONValue tool helpers

extension AnthropicResponse.ToolUse {
    /// Tool-call arguments serialized as a JSON object string.
    var argumentsJSONString: String {
        (try? jsonObjectString(from: input ?? [:])) ?? "{}"
    }
}

/// Builds an `AnthropicTool` from any encodable schema.
func makeAnthropicTool(name: String, description: String, schema: some Encodable) throws -> AnthropicTool {
    let inputSchema = try providerToolSchemaJSONValue(forEncodableSchema: schema)
    return AnthropicTool(name: name, description: description, inputSchema: inputSchema)
}

/// Builds a `tool_use` content block from a JSON-object arguments string.
func makeAnthropicToolUseBlock(id: String, name: String, argumentsJSONString: String) -> AnthropicResponse.ContentBlock {
    let input: [String: JSONValue]? = if let data = argumentsJSONString.data(using: .utf8),
                                         case let .object(object)? = try? JSONDecoder().decode(JSONValue.self, from: data) {
        object
    } else {
        nil
    }
    return .toolUse(.init(id: id, name: name, input: input))
}

/// Serializes a `[String: JSONValue]` to a deterministic JSON object string.
func jsonObjectString(from object: [String: JSONValue]) throws -> String {
    let data = try JSONEncoder.deterministic.encode(JSONValue.object(object))
    return String(data: data, encoding: .utf8) ?? "{}"
}

// MARK: - AnthropicOAuthLanguageModelError

/// Errors thrown by `AnthropicOAuthLanguageModel` while talking to the
/// Anthropic Messages API.
public enum AnthropicOAuthLanguageModelError: LocalizedError, Sendable {
    case invalidResponse
    case requestFailed(statusCode: Int, message: String)
    case unsupportedContentType

    // MARK: Public

    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            "Anthropic returned an invalid response."
        case let .requestFailed(statusCode, message):
            "Anthropic request failed with status \(statusCode): \(message)"
        case .unsupportedContentType:
            "AnthropicOAuthLanguageModel only supports text responses."
        }
    }
}
