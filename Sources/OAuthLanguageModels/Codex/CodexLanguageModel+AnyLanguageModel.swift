import AnyLanguageModel
import Foundation

// MARK: - AnyLanguageModel.LanguageModel conformance

extension CodexLanguageModel: AnyLanguageModel.LanguageModel {
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
        var inputs = try await buildInputs(from: session)
        let tools = try session.tools.map(Self.convertTool)
        var entries: [Transcript.Entry] = []
        var rounds = 0

        while true {
            let response = try await send(
                inputs: inputs,
                instructions: instructions,
                tools: tools.isEmpty ? nil : tools,
                parameters: parameters(options: options, custom: custom, schema: schema, type: type)
            )

            // Replay reasoning items in subsequent requests within this loop.
            inputs.append(contentsOf: response.reasoningItems)

            let toolCalls = response.toolCalls
            if !toolCalls.isEmpty {
                await state.remember(toolCalls)
                inputs.append(contentsOf: CodexInputItem.functionCalls(for: toolCalls))

                let providerCalls = try toolCalls.map(Self.providerCall(for:))

                let resolution = try await resolveToolCalls(providerCalls, session: session)
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
                        for invocation in invocations {
                            entries.append(.toolOutput(invocation.output))
                            inputs.append(
                                .functionCallOutput(
                                    callID: invocation.output.id,
                                    output: Self.toolOutputString(invocation.output.segments)
                                )
                            )
                        }
                        rounds += 1
                        guard rounds < maxToolRounds else {
                            throw CodexLanguageModelError.toolLoopLimitExceeded(rounds: rounds)
                        }
                        continue
                    }
                }
            }

            let text = response.text ?? ""
            guard text.isEmpty == false || response.hasOutput else {
                throw CodexLanguageModelError.noResponseGenerated
            }
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
                    let replayed = try await buildInputs(from: session)
                    var inputs = replayed
                    var answer = StreamedAnswer<Content>(isStructured: schema != nil)
                    // The response this turn becomes, so its tool rounds can be found
                    // again when the next turn rebuilds the conversation.
                    let responseIndex = session.transcript.responseCount
                    var wroteAnything = false
                    var hasOutput = false
                    var rounds = 0

                    while true {
                        var toolCalls: [CodexToolCall] = []
                        var reasoningItems: [CodexInputItem] = []
                        answer.startTurn()

                        let parts = try await sendStream(
                            inputs: inputs,
                            instructions: instructions,
                            tools: tools.isEmpty ? nil : tools,
                            parameters: parameters(options: options, custom: custom, schema: schema, type: type)
                        )
                        for try await part in parts {
                            switch part {
                            case let .text(delta):
                                wroteAnything = true
                                if let snapshot = try answer.append(delta) {
                                    continuation.yield(snapshot)
                                }
                            case .reasoning:
                                // Reasoning reaches the caller through the model's
                                // `onEvent`, never as part of the answer.
                                break
                            case let .toolCall(call):
                                toolCalls.append(call)
                            case let .finished(response):
                                reasoningItems = response.reasoningItems
                                hasOutput = hasOutput || response.hasOutput
                            }
                        }

                        // Replayed in subsequent requests: `store: false` means the
                        // provider keeps none of this turn's reasoning for the next one.
                        inputs.append(contentsOf: reasoningItems)

                        guard !toolCalls.isEmpty else { break }
                        await state.remember(toolCalls)
                        inputs.append(contentsOf: CodexInputItem.functionCalls(for: toolCalls))

                        let providerCalls = try toolCalls.map(Self.providerCall(for:))
                        guard case let .invocations(invocations) = try await resolveToolCalls(providerCalls, session: session),
                              !invocations.isEmpty else {
                            break
                        }

                        for invocation in invocations {
                            inputs.append(
                                .functionCallOutput(
                                    callID: invocation.output.id,
                                    output: Self.toolOutputString(invocation.output.segments)
                                )
                            )
                        }
                        rounds += 1
                        guard rounds < maxToolRounds else {
                            throw CodexLanguageModelError.toolLoopLimitExceeded(rounds: rounds)
                        }
                    }

                    // Everything the exchange added past the conversation it started
                    // from: the replayed reasoning, the function calls and their output,
                    // in the order they went out.
                    Self.streamedTurns.record(
                        Array(inputs.dropFirst(replayed.count)),
                        for: session,
                        at: responseIndex
                    )
                    // A turn that produced neither text nor an output array produced
                    // nothing at all, which is worth an error rather than an empty stream
                    // the caller has to interpret.
                    guard wroteAnything || hasOutput else {
                        throw CodexLanguageModelError.noResponseGenerated
                    }
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

    private func parameters<Content: Generable>(
        options: GenerationOptions,
        custom: CustomGenerationOptions,
        schema: JSONValue?,
        type: Content.Type
    ) -> CodexRequestParameters {
        CodexRequestParameters(
            temperature: options.temperature,
            topP: custom.topP,
            maxOutputTokens: custom.maxOutputTokens ?? options.maximumResponseTokens,
            maxToolCalls: custom.maxToolCalls,
            verbosity: custom.verbosity?.rawValue,
            parallelToolCalls: custom.parallelToolCalls,
            reasoningEffort: custom.reasoning?.effort?.rawValue,
            reasoningSummary: custom.reasoning?.summary?.rawValue,
            toolChoice: custom.toolChoice.map(Self.toolChoiceJSON),
            responseFormat: schema.map { Self.responseFormat($0, named: Self.schemaName(for: type)) },
            extraBody: custom.extraBody
        )
    }

    /// The Responses API's `text.format`, which needs a name for the schema as well as
    /// the schema itself.
    ///
    /// Sent without `strict`. Strict mode demands that every property be listed as
    /// required, and expresses an optional one as a union with null instead — where a
    /// `Generable` type leaves its optional properties out of `required` and types them
    /// plainly. Asking for strict against such a schema is refused outright, so a type
    /// with one optional field could not be generated at all; without it the schema still
    /// guides the model, and an answer that does not match is reported when it is
    /// decoded.
    private static func responseFormat(_ schema: JSONValue, named name: String) -> JSONValue {
        .object([
            "type": .string("json_schema"),
            "name": .string(name),
            "schema": schema
        ])
    }

    /// The type's own name, reduced to what the API accepts as a schema name.
    private static func schemaName<Content: Generable>(for type: Content.Type) -> String {
        let name = String(describing: type).filter { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" }
        return name.isEmpty ? "Response" : name
    }

    /// The session's own instructions, with the schema appended where the caller asked
    /// for it in the prompt as well as in `text.format`.
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

    /// The conversation as it goes back on the wire.
    ///
    /// A response that was streamed with tools in play is put back as the items that turn
    /// actually sent — the function calls, their output, and the reasoning to replay —
    /// rather than as the one assistant message the transcript remembers. Anything else
    /// diverges from what the provider cached under this session's `prompt_cache_key`.
    private func buildInputs(from session: LanguageModelSession) async throws -> [CodexInputItem] {
        var input: [CodexInputItem] = []
        var responseIndex = 0
        for entry in session.transcript {
            switch entry {
            case .instructions:
                break
            case let .prompt(prompt):
                input.append(Self.userMessage(from: prompt.segments))
            case let .response(response):
                if let recorded = Self.streamedTurns.messages(for: session, at: responseIndex) {
                    input.append(contentsOf: recorded)
                }
                responseIndex += 1
                input.append(.assistantMessage(textSegments: Self.assistantTexts(from: response.segments)))
            case let .toolCalls(toolCalls):
                for call in toolCalls {
                    let arguments = try Self.encodedJSONString(for: call.arguments)
                    let itemID = await state.itemID(for: call.id) ?? call.id
                    input.append(
                        .functionCall(itemID: itemID, callID: call.id, name: call.toolName, argumentsJSON: arguments)
                    )
                }
            case let .toolOutput(output):
                input.append(.functionCallOutput(callID: output.id, output: Self.toolOutputString(output.segments)))
            }
        }
        return input
    }

    private static func toolChoiceJSON(_ choice: CustomGenerationOptions.ToolChoice) -> JSONValue {
        switch choice {
        case .none: return .string("none")
        case .auto: return .string("auto")
        case .required: return .string("required")
        case let .function(name):
            return .object(["type": .string("function"), "name": .string(name)])
        case let .allowedTools(names, mode):
            let descriptors = names.map { name in
                JSONValue.object(["type": .string("function"), "name": .string(name)])
            }
            return .object([
                "type": .string("allowed_tools"),
                "mode": .string(mode.rawValue),
                "tools": .array(descriptors)
            ])
        }
    }

    /// A model-issued call, as the resolver takes one.
    ///
    /// Throws where the arguments will not parse. Dropping such a call instead would
    /// leave the model waiting on a result for something nobody ran.
    private static func providerCall(for call: CodexToolCall) throws -> ProviderToolCall {
        try ProviderToolCall(
            id: call.id,
            itemID: call.itemID,
            name: call.name,
            arguments: GeneratedContent(json: call.argumentsJSON)
        )
    }

    private static func convertTool(_ tool: any Tool) throws -> OpenResponsesTool {
        try makeOpenResponsesTool(name: tool.name, description: tool.description, schema: tool.parameters)
    }

    private static func userMessage(from segments: [Transcript.Segment]) -> CodexInputItem {
        var texts: [String] = []
        var imageURLs: [String] = []
        for segment in segments {
            switch segment {
            case let .text(text):
                texts.append(text.content)
            case let .structure(structured):
                switch structured.content.kind {
                case let .string(value): texts.append(value)
                default: texts.append(structured.content.jsonString)
                }
            case let .image(image):
                switch image.source {
                case let .url(url):
                    imageURLs.append(url.absoluteString)
                case let .data(data, mimeType):
                    imageURLs.append("data:\(mimeType);base64,\(data.base64EncodedString())")
                }
            }
        }
        return .userMessage(textSegments: texts, imageURLs: imageURLs)
    }

    private static func assistantTexts(from segments: [Transcript.Segment]) -> [String] {
        segments.compactMap { segment in
            switch segment {
            case let .text(text):
                text.content
            case let .structure(structured):
                switch structured.content.kind {
                case let .string(value): value
                default: structured.content.jsonString
                }
            case .image:
                nil
            }
        }
    }

    private static func toolOutputString(_ segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case let .text(text):
                text.content
            case let .structure(structured):
                switch structured.content.kind {
                case let .string(value): value
                default: structured.content.jsonString
                }
            case .image:
                nil
            }
        }.joined(separator: "\n")
    }

    private static func encodedJSONString(for content: GeneratedContent) throws -> String {
        let data = try JSONEncoder.deterministic.encode(content)
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}
