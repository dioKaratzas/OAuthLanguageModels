import AnyLanguageModel
import Foundation

public let defaultAnthropicBaseURL = URL(string: "https://api.anthropic.com/")!
private let claudeCodeVersion = "2.1.75"
private let claudeCodeSystemPreamble = "You are Claude Code, Anthropic's official CLI for Claude."

// MARK: - AnthropicOAuthLanguageModel

/// `LanguageModel` implementation that talks to Anthropic's Messages API
/// using a Claude Code OAuth access token (typically obtained via
/// `AnthropicOAuthFlow`).
///
/// The OAuth path requires the request to identify itself as Claude Code,
/// so this model always prefixes the system prompt with the Claude Code
/// preamble and sends the corresponding `user-agent`/`x-app` headers.
///
/// Drop into any `LanguageModelSession` from AnyLanguageModel:
///
/// ```swift
/// let model = AnthropicOAuthLanguageModel(
///     tokenProvider: { try await myAuth.validAccessToken() },
///     model: "claude-opus-4"
/// )
/// let session = LanguageModelSession(model: model)
/// ```
public struct AnthropicOAuthLanguageModel: LanguageModel {
    // MARK: Lifecycle

    public init(
        tokenProvider: @escaping @Sendable () async throws -> String,
        model: String,
        baseURL: URL = defaultAnthropicBaseURL,
        maxTokens: Int = 4096,
        longCacheRetention: Bool = false
    ) {
        self.tokenProvider = tokenProvider
        self.model = model
        self.baseURL = baseURL
        self.maxTokens = maxTokens
        self.longCacheRetention = longCacheRetention
    }

    // MARK: Public

    public typealias UnavailableReason = Never

    public let tokenProvider: @Sendable () async throws -> String
    public let model: String
    public let baseURL: URL
    public let maxTokens: Int
    public let longCacheRetention: Bool

    public func respond<Content: Generable>(
        within session: LanguageModelSession,
        to _: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt _: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        guard type == String.self else {
            throw AnthropicOAuthLanguageModelError.unsupportedContentType
        }

        let custom = options[custom: Self.self] ?? .init()
        var messages = try Self.buildMessages(from: session.transcript)
        let tools = try session.tools.map(Self.convertToolToAnthropicFormat)
        var entries: [Transcript.Entry] = []

        while true {
            let payload = try await send(
                messages: messages,
                instructions: session.instructions?.description,
                tools: tools.isEmpty ? nil : tools,
                options: options,
                custom: custom
            )

            let toolCalls = try payload.content.compactMap { block -> ProviderToolCall? in
                guard case let .toolUse(use) = block else { return nil }
                return try ProviderToolCall(
                    id: use.id,
                    itemID: use.id,
                    name: use.name,
                    arguments: Self.toGeneratedContent(use.input)
                )
            }

            if !toolCalls.isEmpty {
                let resolution = try await resolveToolCalls(toolCalls, session: session)
                switch resolution {
                case let .stop(calls):
                    if !calls.isEmpty {
                        entries.append(.toolCalls(Transcript.ToolCalls(calls)))
                    }
                    let empty = try emptyResponseContent(for: type)
                    return LanguageModelSession.Response(
                        content: empty.content,
                        rawContent: empty.rawContent,
                        transcriptEntries: ArraySlice(entries)
                    )
                case let .invocations(invocations):
                    if !invocations.isEmpty {
                        entries.append(.toolCalls(Transcript.ToolCalls(invocations.map(\.call))))
                        messages.append(.init(role: "assistant", content: payload.content))
                        for invocation in invocations {
                            entries.append(.toolOutput(invocation.output))
                            messages.append(
                                .init(
                                    role: "user",
                                    content: [
                                        .toolResult(
                                            .init(
                                                toolUseID: invocation.call.id,
                                                content: Self.convertSegmentsToAnthropicContent(invocation.output.segments)
                                            )
                                        )
                                    ]
                                )
                            )
                        }
                        continue
                    }
                }
            }

            let text = payload.content.compactMap { block -> String? in
                if case let .text(text) = block { return text.text }
                return nil
            }.joined()

            return LanguageModelSession.Response(
                content: text as! Content,
                rawContent: GeneratedContent(text),
                transcriptEntries: ArraySlice(entries)
            )
        }
    }

    public func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init { continuation in
            let task = Task {
                do {
                    let response = try await respond(
                        within: session,
                        to: prompt,
                        generating: type,
                        includeSchemaInPrompt: includeSchemaInPrompt,
                        options: options
                    )
                    continuation.yield(.init(content: response.content.asPartiallyGenerated(), rawContent: response.rawContent))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return LanguageModelSession.ResponseStream(stream: stream)
    }

    // MARK: Private

    /// Top-level body keys that callers may not override via
    /// `CustomGenerationOptions.extraBody`. These are required for the
    /// OAuth/Claude Code request shape to be valid.
    private static let reservedBodyKeys: Set<String> = [
        "model", "system", "messages", "tools"
    ]

    private var cacheControl: AnthropicRequest.CacheControl {
        longCacheRetention ? .ephemeralLong : .ephemeral
    }

    private static func buildMessages(from transcript: Transcript) throws -> [AnthropicRequest.Message] {
        var messages: [AnthropicRequest.Message] = []
        for entry in transcript {
            switch entry {
            case .instructions:
                break
            case let .prompt(prompt):
                messages.append(.init(role: "user", content: convertSegmentsToAnthropicContent(prompt.segments)))
            case let .response(response):
                messages.append(.init(role: "assistant", content: convertSegmentsToAnthropicContent(response.segments)))
            case let .toolCalls(toolCalls):
                let blocks = try toolCalls.map { call in
                    try AnthropicResponse.ContentBlock.toolUse(
                        .init(id: call.id, name: call.toolName, input: fromGeneratedContent(call.arguments))
                    )
                }
                messages.append(.init(role: "assistant", content: blocks))
            case let .toolOutput(toolOutput):
                messages.append(
                    .init(
                        role: "user",
                        content: [
                            .toolResult(
                                .init(toolUseID: toolOutput.id, content: convertSegmentsToAnthropicContent(toolOutput.segments))
                            )
                        ]
                    )
                )
            }
        }
        return messages
    }

    private static func convertSegmentsToAnthropicContent(_ segments: [Transcript.Segment]) -> [AnthropicResponse.ContentBlock] {
        segments.compactMap { segment in
            switch segment {
            case let .text(text):
                .text(.init(text: text.content))
            case let .structure(structured):
                switch structured.content.kind {
                case let .string(string):
                    .text(.init(text: string))
                default:
                    .text(.init(text: structured.content.jsonString))
                }
            case let .image(image):
                switch image.source {
                case let .data(data, mimeType):
                    .image(.init(base64Data: data.base64EncodedString(), mimeType: mimeType))
                case let .url(url):
                    .image(.init(url: url.absoluteString))
                }
            }
        }
    }

    private static func convertToolToAnthropicFormat(_ tool: any Tool) throws -> AnthropicTool {
        let inputSchema = try providerToolSchemaJSONValue(for: tool.parameters)
        return AnthropicTool(name: tool.name, description: tool.description, inputSchema: inputSchema)
    }

    private static func toGeneratedContent(_ value: [String: JSONValue]?) throws -> GeneratedContent {
        guard let value else { return GeneratedContent(properties: [:]) }
        let data = try JSONEncoder().encode(JSONValue.object(value))
        let json = String(data: data, encoding: .utf8) ?? "{}"
        return try GeneratedContent(json: json)
    }

    private static func fromGeneratedContent(_ content: GeneratedContent) throws -> [String: JSONValue] {
        let data = try JSONEncoder().encode(content)
        let value = try JSONDecoder().decode(JSONValue.self, from: data)
        guard case let .object(object) = value else { return [:] }
        return object
    }

    /// Encode the request body and merge any caller-supplied `extraBody`
    /// keys, dropping reserved keys to preserve the OAuth request shape.
    /// Re-encodes through the deterministic encoder so the final bytes
    /// have stable key ordering (cache-prefix safety).
    private static func encodeBody(
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

    private func send(
        messages: [AnthropicRequest.Message],
        instructions: String?,
        tools: [AnthropicTool]?,
        options: GenerationOptions,
        custom: CustomGenerationOptions
    ) async throws -> AnthropicResponse {
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
        request.setValue("claude-code-20250219,oauth-2025-04-20", forHTTPHeaderField: "anthropic-beta")
        request.setValue("true", forHTTPHeaderField: "anthropic-dangerous-direct-browser-access")
        request.setValue("claude-cli/\(claudeCodeVersion)", forHTTPHeaderField: "user-agent")
        request.setValue("cli", forHTTPHeaderField: "x-app")
        request.setValue("application/json", forHTTPHeaderField: "accept")
        request.setValue("application/json", forHTTPHeaderField: "content-type")

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

        // Anthropic requires temperature == 1 when extended thinking is on.
        let temperature: Double? = custom.thinking != nil ? 1 : options.temperature

        let body = AnthropicRequest(
            model: model,
            maxTokens: options.maximumResponseTokens ?? maxTokens,
            system: system,
            messages: cachedMessages,
            tools: tools,
            temperature: temperature,
            topP: custom.topP,
            topK: custom.topK,
            stopSequences: custom.stopSequences,
            toolChoice: custom.toolChoice.map(AnthropicRequest.ToolChoice.init(from:)),
            thinking: custom.thinking.map { .init(budgetTokens: $0.budgetTokens) }
        )
        request.httpBody = try Self.encodeBody(body, mergingExtraBody: custom.extraBody)

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
}

// MARK: - AnthropicRequest

private struct AnthropicRequest: Encodable {
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

        mutating func markLastBlockCached(with cacheControl: CacheControl) {
            for index in content.indices.reversed() {
                switch content[index] {
                case var .text(text):
                    text.cacheControl = cacheControl
                    content[index] = .text(text)
                    return
                case .image, .toolUse, .toolResult, .thinking, .redactedThinking:
                    continue
                }
            }
        }
    }

    enum ToolChoice: Encodable {
        case auto
        case any
        case tool(name: String)
        case disabled

        // MARK: Lifecycle

        init(from choice: AnthropicOAuthLanguageModel.CustomGenerationOptions.ToolChoice) {
            switch choice {
            case .auto: self = .auto
            case .any: self = .any
            case let .tool(name): self = .tool(name: name)
            case .disabled: self = .disabled
            }
        }

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
    let stream: Bool = false
    var temperature: Double?
    var topP: Double?
    var topK: Int?
    var stopSequences: [String]?
    var toolChoice: ToolChoice?
    var thinking: Thinking?
}

// MARK: - AnthropicTool

private struct AnthropicTool: Codable {
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

private struct AnthropicResponse: Decodable {
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

    /// A thinking block emitted by Claude during extended thinking. The
    /// `signature` must be preserved verbatim and replayed in subsequent
    /// requests within the same tool-use turn, otherwise Anthropic
    /// rejects the request.
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

    /// A thinking block whose contents have been redacted by Anthropic's
    /// safety systems. Opaque to clients but must still be replayed.
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
        }

        let type: String
        let toolUseID: String
        let content: [ContentBlock]
    }

    let content: [ContentBlock]
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
