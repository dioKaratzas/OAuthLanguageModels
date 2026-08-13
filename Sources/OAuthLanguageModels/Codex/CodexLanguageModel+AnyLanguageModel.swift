import AnyLanguageModel
import Foundation

// MARK: - AnyLanguageModel.LanguageModel conformance

extension CodexLanguageModel: AnyLanguageModel.LanguageModel {
    public typealias UnavailableReason = Never

    public func respond<Content: Generable>(
        within session: LanguageModelSession,
        to _: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt _: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        guard type == String.self else {
            throw CodexLanguageModelError.unsupportedContentType
        }

        let custom = options[custom: Self.self] ?? .init()
        var inputs = try await buildInputs(from: session.transcript)
        let tools = session.tools.map(Self.convertTool)
        var entries: [Transcript.Entry] = []

        while true {
            let response = try await send(
                inputs: inputs,
                instructions: session.instructions?.description,
                tools: tools.isEmpty ? nil : tools,
                parameters: parameters(options: options, custom: custom)
            )

            // Replay reasoning items in subsequent requests within this loop.
            inputs.append(contentsOf: response.reasoningItems)

            let toolCalls = response.toolCalls
            if !toolCalls.isEmpty {
                await state.remember(toolCalls)
                inputs.append(contentsOf: CodexInputItem.functionCalls(for: toolCalls))

                let providerCalls = toolCalls.compactMap { call -> ProviderToolCall? in
                    guard let arguments = try? GeneratedContent(json: call.argumentsJSON) else { return nil }
                    return ProviderToolCall(id: call.id, itemID: call.itemID, name: call.name, arguments: arguments)
                }

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
                        continue
                    }
                }
            }

            let text = response.text ?? ""
            guard text.isEmpty == false || response.hasOutput else {
                throw CodexLanguageModelError.noResponseGenerated
            }
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
                    let tools = session.tools.map(Self.convertTool)
                    var inputs = try await buildInputs(from: session.transcript)
                    var text = ""

                    while true {
                        var toolCalls: [CodexToolCall] = []
                        var reasoningItems: [CodexInputItem] = []

                        let parts = try await sendStream(
                            inputs: inputs,
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
                            case .reasoning:
                                // Reasoning reaches the caller through the model's
                                // `onEvent`, never as part of the answer.
                                break
                            case let .toolCall(call):
                                toolCalls.append(call)
                            case let .finished(response):
                                reasoningItems = response.reasoningItems
                            }
                        }

                        // Replayed in subsequent requests: `store: false` means the
                        // provider keeps none of this turn's reasoning for the next one.
                        inputs.append(contentsOf: reasoningItems)

                        guard !toolCalls.isEmpty else { break }
                        await state.remember(toolCalls)
                        inputs.append(contentsOf: CodexInputItem.functionCalls(for: toolCalls))

                        let providerCalls = toolCalls.compactMap { call -> ProviderToolCall? in
                            guard let arguments = try? GeneratedContent(json: call.argumentsJSON) else { return nil }
                            return ProviderToolCall(id: call.id, itemID: call.itemID, name: call.name, arguments: arguments)
                        }
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
            extraBody: custom.extraBody
        )
    }

    private func buildInputs(from transcript: Transcript) async throws -> [CodexInputItem] {
        var input: [CodexInputItem] = []
        for entry in transcript {
            switch entry {
            case .instructions:
                break
            case let .prompt(prompt):
                input.append(Self.userMessage(from: prompt.segments))
            case let .response(response):
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

    private static func convertTool(_ tool: any Tool) -> OpenResponsesTool {
        makeOpenResponsesTool(name: tool.name, description: tool.description, schema: tool.parameters)
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
