import AnyLanguageModel
import Foundation

// MARK: - CodexStreamPart

/// One whole piece of a streamed turn.
///
/// Text and reasoning arrive in fragments and are passed on as they land. A function
/// call is not: its arguments stream as JSON fragments that mean nothing until the item
/// closes, so a call is only ever handed over complete.
enum CodexStreamPart: Sendable {
    case text(String)
    case reasoning(String)
    case toolCall(CodexToolCall)
    case finished(CodexStreamingResponse)
}

// MARK: - CodexStreamParser

/// Assembles the Responses API's server-sent events into ``CodexStreamPart`` values.
struct CodexStreamParser {
    // MARK: Internal

    /// Feeds one `data:` payload in, and returns whatever it completed.
    mutating func consume(payload: String) throws -> [CodexStreamPart] {
        guard payload != "[DONE]", !payload.isEmpty else { return [] }
        guard let data = payload.data(using: .utf8),
              let object = (try? JSONDecoder().decode(JSONValue.self, from: data))?.objectValue else {
            return []
        }

        // Every event that carries a response snapshot carries the whole output array
        // with it, so the last one seen is the turn as the provider has it.
        if let response = object["response"]?.objectValue {
            if let output = response["output"]?.arrayValue { latestOutput = output }
            if let text = response["output_text"]?.stringValue { latestOutputText = text }
            if let usage = response["usage"]?.objectValue { apply(usage) }
        } else if let output = object["output"]?.arrayValue {
            latestOutput = output
        }
        if let text = object["output_text"]?.stringValue { latestOutputText = text }

        switch object["type"]?.stringValue {
        case "response.output_text.delta":
            guard let delta = object["delta"]?.stringValue else { return [] }
            text += delta
            return [.text(delta)]

        case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
            guard let delta = object["delta"]?.stringValue else { return [] }
            return [.reasoning(delta)]

        case "response.output_item.added":
            guard let item = object["item"]?.objectValue, Self.isFunctionCall(item) else { return [] }
            open(item)
            return []

        case "response.function_call_arguments.delta":
            guard let itemID = object["item_id"]?.stringValue,
                  let delta = object["delta"]?.stringValue else {
                return []
            }
            pending[itemID, default: PendingCall()].arguments += delta
            return []

        case "response.function_call_arguments.done":
            guard let itemID = object["item_id"]?.stringValue,
                  let arguments = object["arguments"]?.stringValue else {
                return []
            }
            pending[itemID, default: PendingCall()].arguments = arguments
            return []

        case "response.output_item.done":
            guard let item = object["item"]?.objectValue, Self.isFunctionCall(item) else { return [] }
            open(item)
            return close(itemID: item["id"]?.stringValue).map { [.toolCall($0)] } ?? []

        case "response.completed", "response.incomplete":
            return finish(stopReason: Self.stopReason(in: object))

        case "response.failed", "error":
            throw CodexLanguageModelError.requestFailed(
                statusCode: 200,
                message: Self.errorMessage(in: object) ?? "Codex ended the stream with an error."
            )

        default:
            return []
        }
    }

    /// The terminal part, for a body that ended without a `response.completed` — a
    /// dropped connection, or a provider that stopped talking mid-answer. The report then
    /// carries no stop reason, which is how a caller tells the two endings apart.
    mutating func finish(stopReason: StopReason? = nil) -> [CodexStreamPart] {
        guard !isFinished else { return [] }
        isFinished = true

        // Some turns report their function calls only in the final output array.
        var missed: [CodexStreamPart] = []
        for item in latestOutput ?? [] {
            guard let object = item.objectValue, Self.isFunctionCall(object) else { continue }
            open(object)
            if let call = close(itemID: object["id"]?.stringValue) {
                missed.append(.toolCall(call))
            }
        }

        let answer = text.isEmpty ? latestOutputText : text
        let response = CodexStreamingResponse(
            text: answer ?? Self.extractText(from: latestOutput),
            hasOutput: latestOutput != nil,
            toolCalls: emitted,
            reasoningItems: Self.extractReasoningItems(from: latestOutput).map { CodexInputItem(json: $0) },
            report: TurnReport(
                usage: usage,
                // A turn that ends holding tool calls is waiting on their results, whatever
                // the transport called it.
                stopReason: stopReason == StopReason.endTurn && !emitted.isEmpty ? .toolUse : stopReason
            )
        )
        return missed + [.finished(response)]
    }

    // MARK: Private

    private struct PendingCall {
        var callID: String?
        var name: String?
        var arguments = ""
    }

    private var text = ""
    private var latestOutput: [JSONValue]?
    private var latestOutputText: String?
    private var pending: [String: PendingCall] = [:]
    private var emitted: [CodexToolCall] = []
    private var emittedIDs: Set<String> = []
    private var usage = TokenUsage()
    private var isFinished = false

    private static func isFunctionCall(_ item: [String: JSONValue]) -> Bool {
        guard let type = item["type"]?.stringValue else { return false }
        return ["function_call", "tool_call", "tool_use"].contains(type)
    }

    private static func stopReason(in object: [String: JSONValue]) -> StopReason {
        guard let reason = object["response"]?.objectValue?["incomplete_details"]?
            .objectValue?["reason"]?.stringValue else {
            return .endTurn
        }
        // The Responses API has no `stop_reason`: a truncated turn is a terminal
        // `response.incomplete` whose details name the ceiling that was hit.
        return reason == "max_output_tokens" ? .maxTokens : .other(reason)
    }

    private static func errorMessage(in object: [String: JSONValue]) -> String? {
        let error = object["response"]?.objectValue?["error"] ?? object["error"]
        return error?.objectValue?["message"]?.stringValue ?? object["message"]?.stringValue
    }

    private static func extractText(from output: [JSONValue]?) -> String? {
        guard let output else { return nil }
        let parts = output.compactMap { item -> [JSONValue]? in
            guard let object = item.objectValue,
                  object["type"]?.stringValue == "message" else {
                return nil
            }
            return object["content"]?.arrayValue
        }.flatMap { content in
            content.compactMap { block -> String? in
                guard let object = block.objectValue,
                      object["type"]?.stringValue == "output_text" else {
                    return nil
                }
                return object["text"]?.stringValue
            }
        }
        return parts.isEmpty ? nil : parts.joined()
    }

    /// Pull `reasoning` items out of the final response output array, in their original
    /// order, to replay across turns when `store: false`.
    private static func extractReasoningItems(from output: [JSONValue]?) -> [JSONValue] {
        (output ?? []).filter { $0.objectValue?["type"]?.stringValue == "reasoning" }
    }

    private mutating func apply(_ usage: [String: JSONValue]) {
        let cached = usage["input_tokens_details"]?.objectValue?["cached_tokens"]?.intValue ?? 0
        // The Responses API counts cached tokens inside `input_tokens`; `TokenUsage`
        // keeps the two apart so a cache read is visible rather than looking like a
        // full-price read.
        self.usage.inputTokens = max(0, (usage["input_tokens"]?.intValue ?? 0) - cached)
        self.usage.cacheReadTokens = cached
        self.usage.outputTokens = usage["output_tokens"]?.intValue ?? 0
    }

    private mutating func open(_ item: [String: JSONValue]) {
        let itemID = item["id"]?.stringValue
        let key = itemID ?? item["call_id"]?.stringValue ?? ""
        guard !key.isEmpty else { return }
        var call = pending[key] ?? PendingCall()
        call.callID = item["call_id"]?.stringValue ?? call.callID ?? itemID
        call.name = item["name"]?.stringValue ?? call.name
        if let arguments = item["arguments"]?.stringValue, !arguments.isEmpty {
            call.arguments = arguments
        }
        pending[key] = call
    }

    private mutating func close(itemID: String?) -> CodexToolCall? {
        guard let key = itemID, let call = pending.removeValue(forKey: key),
              let callID = call.callID, let name = call.name,
              !callID.isEmpty, !name.isEmpty, !emittedIDs.contains(callID) else {
            return nil
        }
        emittedIDs.insert(callID)
        let resolved = CodexToolCall(
            id: callID,
            itemID: key,
            name: name,
            argumentsJSON: call.arguments.isEmpty ? "{}" : call.arguments
        )
        emitted.append(resolved)
        return resolved
    }
}
