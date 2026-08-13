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
        // Tools have to be reassembled from their argument fragments and then run, and
        // a structured type is only decodable once it is whole. Both are answered in
        // one piece; only plain text can be handed over as it arrives.
        guard type == String.self, session.tools.isEmpty else {
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
                    var text = ""
                    let parts = try await sendStream(
                        messages: try Self.buildMessages(from: session.transcript),
                        instructions: session.instructions?.description,
                        parameters: parameters(options: options, custom: custom)
                    )
                    for try await part in parts {
                        // Reasoning and the terminal report reach the caller through the
                        // model's `onEvent`; only the answer belongs in a snapshot.
                        guard case let .text(delta) = part else { continue }
                        text += delta
                        // Snapshots are cumulative: each one is the answer so far, not
                        // the piece that just landed.
                        let content = text as! Content
                        continuation.yield(
                            .init(content: content.asPartiallyGenerated(), rawContent: GeneratedContent(text))
                        )
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
