import AnyLanguageModel
import Foundation

// MARK: - AnyLanguageModel.LanguageModel conformance

extension AnthropicOAuthLanguageModel: AnyLanguageModel.LanguageModel {
    public typealias UnavailableReason = Never

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
        let tools = try session.tools.map(Self.convertTool)
        var entries: [Transcript.Entry] = []
        var rounds = 0

        while true {
            let payload = try await send(
                messages: messages,
                instructions: session.instructions?.description,
                tools: tools.isEmpty ? nil : tools,
                parameters: parameters(options: options, custom: custom)
            )

            let toolCalls = payload.content.compactMap { block -> ProviderToolCall? in
                guard case let .toolUse(use) = block else { return nil }
                return Self.providerCall(for: use)
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
                                                content: Self.convertSegments(invocation.output.segments)
                                            )
                                        )
                                    ]
                                )
                            )
                        }
                        rounds += 1
                        guard rounds < maxToolRounds else {
                            throw AnthropicOAuthLanguageModelError.toolLoopLimitExceeded(rounds: rounds)
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

    /// The answer as it is written, running the tool loop as it goes.
    ///
    /// Each snapshot is the answer so far, and it keeps growing across tool rounds: a
    /// caller appending the tail of every snapshot gets one continuous answer whether or
    /// not tools were called in the middle of it.
    ///
    /// The tool calls and their outputs are not reported as transcript entries — a
    /// `ResponseStream` has nowhere to put them — so a session that streams a tool-using
    /// turn does not have that turn's calls in its transcript afterwards. Use
    /// ``respond(within:to:generating:includeSchemaInPrompt:options:)`` where the
    /// transcript has to be complete.
    ///
    /// A `.stop` decision from a `ToolExecutionDelegate` ends the stream with what has
    /// already been written: a snapshot the caller has been shown cannot be withdrawn, so
    /// the answer stands as far as it got rather than being replaced by an empty one the
    /// way ``respond(within:to:generating:includeSchemaInPrompt:options:)`` does.
    public func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        // A structured type is only decodable once it is whole, so it cannot be handed
        // over a piece at a time. Plain text can, tools or no tools.
        guard type == String.self else {
            return wholeResponseAsStream(
                within: session,
                to: prompt,
                generating: type,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init { continuation in
            let task = Task {
                do {
                    let custom = options[custom: Self.self] ?? .init()
                    let tools = try session.tools.map(Self.convertTool)
                    var messages = try Self.buildMessages(from: session.transcript)
                    var text = ""
                    var rounds = 0

                    while true {
                        var toolCalls: [ProviderToolCall] = []
                        var turn: [AnthropicResponse.ContentBlock] = []

                        let parts = try await sendStream(
                            messages: messages,
                            instructions: session.instructions?.description,
                            tools: tools.isEmpty ? nil : tools,
                            parameters: parameters(options: options, custom: custom)
                        )
                        for try await part in parts {
                            switch part {
                            case let .text(delta):
                                text += delta
                                // Snapshots are cumulative across the whole exchange:
                                // each one is the answer so far, tool rounds included.
                                let content = text as! Content
                                continuation.yield(
                                    .init(content: content.asPartiallyGenerated(), rawContent: GeneratedContent(text))
                                )
                            case .thinking:
                                // Reasoning reaches the caller through the model's
                                // `onEvent`, never as part of the answer.
                                break
                            case let .toolUse(use):
                                if let call = Self.providerCall(for: use) {
                                    toolCalls.append(call)
                                }
                            case let .finished(content, _):
                                turn = content
                            }
                        }

                        guard !toolCalls.isEmpty else { break }
                        guard case let .invocations(invocations) = try await resolveToolCalls(toolCalls, session: session),
                              !invocations.isEmpty else {
                            break
                        }

                        messages.append(.init(role: "assistant", content: turn))
                        for invocation in invocations {
                            messages.append(
                                .init(
                                    role: "user",
                                    content: [
                                        .toolResult(
                                            .init(
                                                toolUseID: invocation.call.id,
                                                content: Self.convertSegments(invocation.output.segments)
                                            )
                                        )
                                    ]
                                )
                            )
                        }
                        rounds += 1
                        guard rounds < maxToolRounds else {
                            throw AnthropicOAuthLanguageModelError.toolLoopLimitExceeded(rounds: rounds)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return LanguageModelSession.ResponseStream(stream: stream)
    }

    private func wholeResponseAsStream<Content: Generable>(
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

    private func parameters(
        options: GenerationOptions,
        custom: CustomGenerationOptions
    ) -> AnthropicRequestParameters {
        AnthropicRequestParameters(
            // Anthropic requires temperature == 1 when extended thinking is on.
            temperature: custom.thinking != nil ? 1 : options.temperature,
            maxTokens: options.maximumResponseTokens,
            topP: custom.topP,
            topK: custom.topK,
            stopSequences: custom.stopSequences,
            toolChoice: custom.toolChoice.map(Self.toolChoice(from:)),
            thinkingBudgetTokens: custom.thinking?.budgetTokens,
            extraBody: custom.extraBody
        )
    }

    private static func toolChoice(
        from choice: CustomGenerationOptions.ToolChoice
    ) -> AnthropicRequest.ToolChoice {
        switch choice {
        case .auto: .auto
        case .any: .any
        case let .tool(name): .tool(name: name)
        case .disabled: .disabled
        }
    }

    private static func buildMessages(from transcript: Transcript) throws -> [AnthropicRequest.Message] {
        var messages: [AnthropicRequest.Message] = []
        for entry in transcript {
            switch entry {
            case .instructions:
                break
            case let .prompt(prompt):
                messages.append(.init(role: "user", content: convertSegments(prompt.segments)))
            case let .response(response):
                messages.append(.init(role: "assistant", content: convertSegments(response.segments)))
            case let .toolCalls(toolCalls):
                let blocks = try toolCalls.map { call in
                    try makeAnthropicToolUseBlock(
                        id: call.id,
                        name: call.toolName,
                        argumentsJSONString: jsonObjectString(from: jsonObject(from: call.arguments))
                    )
                }
                messages.append(.init(role: "assistant", content: blocks))
            case let .toolOutput(toolOutput):
                messages.append(
                    .init(
                        role: "user",
                        content: [
                            .toolResult(
                                .init(toolUseID: toolOutput.id, content: convertSegments(toolOutput.segments))
                            )
                        ]
                    )
                )
            }
        }
        return messages
    }

    private static func convertSegments(_ segments: [Transcript.Segment]) -> [AnthropicResponse.ContentBlock] {
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

    private static func providerCall(for use: AnthropicResponse.ToolUse) -> ProviderToolCall? {
        guard let arguments = try? GeneratedContent(json: use.argumentsJSONString) else { return nil }
        return ProviderToolCall(id: use.id, itemID: use.id, name: use.name, arguments: arguments)
    }

    private static func convertTool(_ tool: any Tool) throws -> AnthropicTool {
        try makeAnthropicTool(name: tool.name, description: tool.description, schema: tool.parameters)
    }

    private static func jsonObject(from content: GeneratedContent) throws -> [String: JSONValue] {
        let data = try JSONEncoder().encode(content)
        let value = try JSONDecoder().decode(JSONValue.self, from: data)
        guard case let .object(object) = value else { return [:] }
        return object
    }
}
