import AnyLanguageModel
import Foundation

// MARK: - AnyLanguageModel.LanguageModel conformance

extension AnthropicOAuthLanguageModel: AnyLanguageModel.LanguageModel {
    public typealias UnavailableReason = Never

    public func respond<Content: Generable>(
        within session: LanguageModelSession,
        to _: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        let custom = options[custom: Self.self] ?? .init()
        let schema = try structuredSchema(for: type)
        let instructions = try Self.instructions(session, schema: schema, inPrompt: includeSchemaInPrompt)
        var messages = try Self.buildMessages(from: session)
        let tools = try session.tools.map(Self.convertTool)
        var entries: [Transcript.Entry] = []
        var rounds = 0

        while true {
            let payload = try await send(
                messages: messages,
                instructions: instructions,
                tools: tools.isEmpty ? nil : tools,
                parameters: parameters(options: options, custom: custom, schema: schema)
            )

            let toolCalls = try payload.content.compactMap { block -> ProviderToolCall? in
                guard case let .toolUse(use) = block else { return nil }
                return try Self.providerCall(for: use)
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
            let answer = try finishedContent(text, as: type)

            return LanguageModelSession.Response(
                content: answer.content,
                rawContent: answer.rawContent,
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
    ///
    /// A structured type streams too. The provider writes the JSON as ordinary assistant
    /// text, and a JSON document is readable at every step of being written: each
    /// snapshot carries the fields that have arrived and leaves the rest nil.
    public func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to _: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init { continuation in
            let task = Task {
                do {
                    let custom = options[custom: Self.self] ?? .init()
                    let schema = try structuredSchema(for: type)
                    let instructions = try Self.instructions(session, schema: schema, inPrompt: includeSchemaInPrompt)
                    let tools = try session.tools.map(Self.convertTool)
                    let replayed = try Self.buildMessages(from: session)
                    var messages = replayed
                    var answer = StreamedAnswer<Content>(isStructured: schema != nil)
                    var rounds = 0
                    // The response this turn becomes, so its tool rounds can be found
                    // again when the next turn rebuilds the conversation.
                    let responseIndex = session.transcript.responseCount

                    while true {
                        var toolCalls: [ProviderToolCall] = []
                        var turn: [AnthropicResponse.ContentBlock] = []
                        answer.startTurn()

                        let parts = try await sendStream(
                            messages: messages,
                            instructions: instructions,
                            tools: tools.isEmpty ? nil : tools,
                            parameters: parameters(options: options, custom: custom, schema: schema)
                        )
                        for try await part in parts {
                            switch part {
                            case let .text(delta):
                                if let snapshot = try answer.append(delta) {
                                    continuation.yield(snapshot)
                                }
                            case .thinking:
                                // Reasoning reaches the caller through the model's
                                // `onEvent`, never as part of the answer.
                                break
                            case let .toolUse(use):
                                toolCalls.append(try Self.providerCall(for: use))
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
                    // Everything the exchange added past the conversation it started
                    // from: the tool calls and their results, in the order they went out.
                    Self.streamedTurns.record(
                        Array(messages.dropFirst(replayed.count)),
                        for: session,
                        at: responseIndex
                    )
                    // A provider that ignored the response format and answered in prose
                    // is a failure worth reporting, not a stream that quietly says
                    // nothing.
                    try answer.checkFinished()
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
        custom: CustomGenerationOptions,
        schema: JSONValue?
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
            responseFormat: schema.map { .object(["type": .string("json_schema"), "schema": $0]) },
            extraBody: custom.extraBody
        )
    }

    /// The session's own instructions, with the schema appended where the caller asked
    /// for it in the prompt as well as in `output_config.format`.
    private static func instructions(
        _ session: LanguageModelSession,
        schema: JSONValue?,
        inPrompt: Bool
    ) throws -> String? {
        let instructions = session.instructions?.description
        guard inPrompt, let schema else { return instructions }
        let sentence = try schemaInstruction(schema)
        guard let instructions, !instructions.isEmpty else { return sentence }
        return instructions + "\n\n" + sentence
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

    /// The conversation as it goes back on the wire.
    ///
    /// A response that was streamed with tools in play is put back as the messages that
    /// turn actually sent — the assistant's calls, the results, then the answer — rather
    /// than as the one plain assistant message the transcript remembers. Anything else
    /// diverges from what the provider cached, and a prefix that diverges is re-read at
    /// full price from that point on.
    private static func buildMessages(from session: LanguageModelSession) throws -> [AnthropicRequest.Message] {
        var messages: [AnthropicRequest.Message] = []
        var responseIndex = 0
        for entry in session.transcript {
            switch entry {
            case .instructions:
                break
            case let .prompt(prompt):
                messages.append(.init(role: "user", content: convertSegments(prompt.segments)))
            case let .response(response):
                if let recorded = streamedTurns.messages(for: session, at: responseIndex) {
                    messages.append(contentsOf: recorded)
                }
                responseIndex += 1
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

    /// A model-issued call, as the resolver takes one.
    ///
    /// Throws where the arguments will not parse. Dropping such a call instead would
    /// leave the model waiting on a result for something nobody ran.
    private static func providerCall(for use: AnthropicResponse.ToolUse) throws -> ProviderToolCall {
        try ProviderToolCall(
            id: use.id,
            itemID: use.id,
            name: use.name,
            arguments: GeneratedContent(json: try use.argumentsJSONString())
        )
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
