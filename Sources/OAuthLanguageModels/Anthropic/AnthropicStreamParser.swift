import Foundation

// MARK: - AnthropicStreamPart

/// One whole piece of a streamed turn.
///
/// Text and reasoning arrive in fragments, so they are passed on as they land. A tool
/// call is not: its arguments are streamed as JSON fragments that mean nothing until the
/// last one, so a call is only ever handed over complete.
enum AnthropicStreamPart {
    case text(String)
    case thinking(String)
    case toolUse(AnthropicResponse.ToolUse)
    /// The turn is over. `content` is the assistant message as it must be replayed when
    /// tool results are sent back — text, thinking with its signature, and the tool
    /// calls, in the order the model wrote them.
    case finished(content: [AnthropicResponse.ContentBlock], report: TurnReport)
}

// MARK: - AnthropicStreamParser

/// Assembles the Messages API's server-sent events into ``AnthropicStreamPart`` values.
///
/// Block types the package does not model — the server-side tool blocks, fallback
/// markers — are carried through as nothing rather than as an error: they only appear
/// for features this package never asks for, and failing the whole stream over one would
/// lose an answer that is otherwise intact.
struct AnthropicStreamParser {
    // MARK: Internal

    /// Feeds one line of the SSE body in, and returns whatever it completed.
    ///
    /// Lines that are not `data:` payloads — the `event:` names, the blank separators —
    /// complete nothing, as do `ping`s.
    mutating func consume(line: String) throws -> [AnthropicStreamPart] {
        guard line.hasPrefix("data:") else { return [] }
        let json = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
        guard let event = try? JSONDecoder.snakeCase.decode(AnthropicStreamEvent.self, from: Data(json.utf8)) else {
            return []
        }

        switch event.type {
        case "message_start":
            if let usage = event.message?.usage {
                apply(usage)
            }
            return []

        case "content_block_start":
            guard let index = event.index, let start = event.contentBlock else { return [] }
            var block = Block(type: start.type, id: start.id, name: start.name)
            block.text = start.text ?? ""
            block.thinking = start.thinking ?? ""
            block.signature = start.signature
            block.data = start.data
            blocks[index] = block
            return block.text.isEmpty ? [] : [.text(block.text)]

        case "content_block_delta":
            guard let index = event.index, let delta = event.delta else { return [] }
            return apply(delta, at: index)

        case "content_block_stop":
            guard let index = event.index, let block = blocks.removeValue(forKey: index) else { return [] }
            return close(block)

        case "message_delta":
            if let raw = event.delta?.stopReason {
                stopReason = StopReason(anthropic: raw)
            }
            // Cumulative, not incremental: the last one seen is the total.
            if let usage = event.usage {
                apply(usage)
            }
            return []

        case "message_stop":
            return finish()

        case "error":
            // Mid-stream failures arrive as an event, with a 200 already on the wire.
            throw AnthropicOAuthLanguageModelError.requestFailed(
                statusCode: 200,
                message: event.error?.message ?? "Anthropic ended the stream with an error."
            )

        default:
            return []
        }
    }

    /// The terminal part, for a body that ended without a `message_stop` — a dropped
    /// connection, or a provider that stopped talking mid-answer. The report then carries
    /// no stop reason, which is how a caller tells the two endings apart.
    mutating func finish() -> [AnthropicStreamPart] {
        guard !isFinished else { return [] }
        isFinished = true
        // Whatever was still open never got its `content_block_stop`; keep it anyway so
        // the assistant turn replays with the text that did arrive.
        for index in blocks.keys.sorted() {
            guard let block = blocks.removeValue(forKey: index) else { continue }
            _ = close(block)
        }
        return [.finished(content: content, report: TurnReport(usage: usage, stopReason: stopReason))]
    }

    // MARK: Private

    private struct Block {
        var type: String
        var id: String?
        var name: String?
        var text = ""
        var thinking = ""
        var signature: String?
        var data: String?
        var arguments = ""
    }

    private var blocks: [Int: Block] = [:]
    private var content: [AnthropicResponse.ContentBlock] = []
    private var usage = TokenUsage()
    private var stopReason: StopReason?
    private var isFinished = false

    private mutating func apply(_ usage: AnthropicUsage) {
        if let input = usage.inputTokens { self.usage.inputTokens = input }
        if let output = usage.outputTokens { self.usage.outputTokens = output }
        if let write = usage.cacheCreationInputTokens { self.usage.cacheWriteTokens = write }
        if let read = usage.cacheReadInputTokens { self.usage.cacheReadTokens = read }
    }

    private mutating func apply(_ delta: AnthropicStreamEvent.Delta, at index: Int) -> [AnthropicStreamPart] {
        guard var block = blocks[index] else { return [] }
        defer { blocks[index] = block }

        switch delta.type {
        case "text_delta":
            guard let text = delta.text else { return [] }
            block.text += text
            return [.text(text)]
        case "thinking_delta":
            guard let thinking = delta.thinking else { return [] }
            block.thinking += thinking
            return [.thinking(thinking)]
        case "signature_delta":
            block.signature = (block.signature ?? "") + (delta.signature ?? "")
            return []
        case "input_json_delta":
            block.arguments += delta.partialJson ?? ""
            return []
        default:
            return []
        }
    }

    private mutating func close(_ block: Block) -> [AnthropicStreamPart] {
        switch block.type {
        case "text":
            content.append(.text(.init(text: block.text)))
        case "thinking":
            content.append(.thinking(.init(thinking: block.thinking, signature: block.signature)))
        case "redacted_thinking":
            content.append(.redactedThinking(.init(data: block.data ?? "")))
        case "tool_use":
            guard let id = block.id, let name = block.name else { return [] }
            let arguments = block.arguments.isEmpty ? "{}" : block.arguments
            let toolUse = makeAnthropicToolUseBlock(id: id, name: name, argumentsJSONString: arguments)
            content.append(toolUse)
            guard case let .toolUse(use) = toolUse else { return [] }
            return [.toolUse(use)]
        default:
            break
        }
        return []
    }
}

// MARK: - AnthropicUsage

/// The token counts Anthropic reports, streaming and not.
struct AnthropicUsage: Decodable {
    var inputTokens: Int?
    var outputTokens: Int?
    var cacheCreationInputTokens: Int?
    var cacheReadInputTokens: Int?
}

extension TokenUsage {
    init(anthropic usage: AnthropicUsage) {
        self.init(
            inputTokens: usage.inputTokens ?? 0,
            outputTokens: usage.outputTokens ?? 0,
            cacheWriteTokens: usage.cacheCreationInputTokens ?? 0,
            cacheReadTokens: usage.cacheReadInputTokens ?? 0
        )
    }
}

// MARK: - AnthropicStreamEvent

struct AnthropicStreamEvent: Decodable {
    struct Message: Decodable {
        var usage: AnthropicUsage?
        var stopReason: String?
    }

    /// The opening shape of a content block. Decoded loosely rather than as an
    /// `AnthropicResponse.ContentBlock` so that a block type the package does not model
    /// opens the stream instead of failing it.
    struct BlockStart: Decodable {
        var type: String
        var id: String?
        var name: String?
        var text: String?
        var thinking: String?
        var signature: String?
        var data: String?
    }

    struct Delta: Decodable {
        var type: String?
        var text: String?
        var thinking: String?
        var signature: String?
        var partialJson: String?
        var stopReason: String?
    }

    struct StreamError: Decodable {
        var type: String?
        var message: String?
    }

    let type: String
    var index: Int?
    var message: Message?
    var contentBlock: BlockStart?
    var delta: Delta?
    var usage: AnthropicUsage?
    var error: StreamError?
}
