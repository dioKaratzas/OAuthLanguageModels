import Foundation
@testable import OAuthLanguageModels

/// Drives ``AnthropicStreamParser`` over a captured SSE body and collects what it made
/// of it.
func drainAnthropic(_ trace: String) throws -> [AnthropicStreamPart] {
    var parser = AnthropicStreamParser()
    var parts: [AnthropicStreamPart] = []
    for line in trace.split(separator: "\n", omittingEmptySubsequences: false) {
        parts += try parser.consume(line: String(line))
    }
    return parts + parser.finish()
}

/// Drives ``CodexStreamParser`` over a captured SSE body and collects what it made of it.
func drainCodex(_ trace: String) throws -> [CodexStreamPart] {
    var parser = CodexStreamParser()
    var parts: [CodexStreamPart] = []
    for line in trace.split(separator: "\n", omittingEmptySubsequences: false) {
        guard line.hasPrefix("data:") else { continue }
        let payload = String(line.dropFirst("data:".count)).trimmingCharacters(in: .whitespaces)
        parts += try parser.consume(payload: payload)
    }
    return parts + parser.finish()
}

extension [AnthropicStreamPart] {
    var text: String {
        compactMap { if case let .text(delta) = $0 { delta } else { nil } }.joined()
    }

    var thinking: String {
        compactMap { if case let .thinking(delta) = $0 { delta } else { nil } }.joined()
    }

    var toolUses: [AnthropicResponse.ToolUse] {
        compactMap { if case let .toolUse(use) = $0 { use } else { nil } }
    }

    var report: TurnReport? {
        for case let .finished(_, report) in self { return report }
        return nil
    }

    var replayedContent: [AnthropicResponse.ContentBlock] {
        for case let .finished(content, _) in self { return content }
        return []
    }
}

extension [CodexStreamPart] {
    var text: String {
        compactMap { if case let .text(delta) = $0 { delta } else { nil } }.joined()
    }

    var reasoning: String {
        compactMap { if case let .reasoning(delta) = $0 { delta } else { nil } }.joined()
    }

    var toolCalls: [CodexToolCall] {
        compactMap { if case let .toolCall(call) = $0 { call } else { nil } }
    }

    var response: CodexStreamingResponse? {
        for case let .finished(response) in self { return response }
        return nil
    }
}
