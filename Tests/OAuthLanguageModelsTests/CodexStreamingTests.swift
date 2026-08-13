import Foundation
import Testing
@testable import OAuthLanguageModels

@Suite("Codex SSE streaming")
struct CodexStreamingTests {
    /// A plain answer, ended by the model, off a warm prompt cache.
    static let cached = """
    data: {"type":"response.created","response":{"id":"resp_01","status":"in_progress","output":[]}}

    data: {"type":"response.output_text.delta","item_id":"msg_01","delta":"A dependency "}

    data: {"type":"response.output_text.delta","item_id":"msg_01","delta":"graph."}

    data: {"type":"response.completed","response":{"id":"resp_01","status":"completed","output":[{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"output_text","text":"A dependency graph."}]}],"usage":{"input_tokens":15248,"input_tokens_details":{"cached_tokens":15234},"output_tokens":6,"total_tokens":15254}}}
    """

    /// The ceiling stopped this one, not the model.
    static let truncated = """
    data: {"type":"response.output_text.delta","item_id":"msg_02","delta":"func parse(line: String) -> Toke"}

    data: {"type":"response.incomplete","response":{"id":"resp_02","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[],"usage":{"input_tokens":300,"input_tokens_details":{"cached_tokens":0},"output_tokens":8192}}}
    """

    /// Reasoning, then text, then one function call whose arguments arrive in fragments.
    static let toolCall = """
    data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_01","delta":"The user wants "}

    data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_01","delta":"today's weather."}

    data: {"type":"response.output_text.delta","item_id":"msg_03","delta":"Let me check."}

    data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fc_01","type":"function_call","status":"in_progress","arguments":"","call_id":"call_01","name":"get_weather"}}

    data: {"type":"response.function_call_arguments.delta","item_id":"fc_01","output_index":1,"delta":"{\\"location\\":"}

    data: {"type":"response.function_call_arguments.delta","item_id":"fc_01","output_index":1,"delta":" \\"San Francisco, CA\\"}"}

    data: {"type":"response.function_call_arguments.done","item_id":"fc_01","output_index":1,"arguments":"{\\"location\\": \\"San Francisco, CA\\"}"}

    data: {"type":"response.output_item.done","output_index":1,"item":{"id":"fc_01","type":"function_call","status":"completed","arguments":"{\\"location\\": \\"San Francisco, CA\\"}","call_id":"call_01","name":"get_weather"}}

    data: {"type":"response.completed","response":{"id":"resp_03","status":"completed","output":[{"id":"rs_01","type":"reasoning","encrypted_content":"gAAAAAB"},{"id":"msg_03","type":"message","role":"assistant","content":[{"type":"output_text","text":"Let me check."}]},{"id":"fc_01","type":"function_call","arguments":"{\\"location\\": \\"San Francisco, CA\\"}","call_id":"call_01","name":"get_weather"}],"usage":{"input_tokens":472,"input_tokens_details":{"cached_tokens":0},"output_tokens":89}}}
    """

    @Test
    func `Each text delta is surfaced as it is parsed, in order`() throws {
        let parts = try drainCodex(Self.cached)

        // The deltas reach the caller one at a time rather than only as a final blob, which
        // is what lets a UI render the answer as it arrives.
        #expect(parts.compactMap { if case let .text(delta) = $0 { delta } else { nil } } == ["A dependency ", "graph."])
        #expect(parts.response?.text == "A dependency graph.")
    }

    @Test
    func `A cached turn reports the prefix as read rather than as input`() throws {
        let usage = try drainCodex(Self.cached).response?.report.usage

        // The Responses API counts cached tokens inside `input_tokens`; the two are
        // separated here so a cache read does not read as a full-price one.
        #expect(usage == TokenUsage(inputTokens: 14, outputTokens: 6, cacheReadTokens: 15234))
    }

    @Test
    func `An answer cut off at the ceiling is reported as truncated`() throws {
        let response = try drainCodex(Self.truncated).response

        #expect(response?.report.stopReason == .maxTokens)
        #expect(response?.report.usage.outputTokens == 8192)
    }

    @Test
    func `An answer that finished is not reported as truncated`() throws {
        #expect(try drainCodex(Self.cached).response?.report.stopReason == .endTurn)
    }

    @Test
    func `A turn waiting on tools says so`() throws {
        #expect(try drainCodex(Self.toolCall).response?.report.stopReason == .toolUse)
    }

    @Test
    func `A function call is handed over once, whole, with the text around it intact`() throws {
        let parts = try drainCodex(Self.toolCall)

        #expect(parts.text == "Let me check.")
        #expect(parts.toolCalls.count == 1)
        // The fragments mean nothing until the item closes, so the call surfaces only
        // once — and the final output array must not produce a second copy of it.
        #expect(parts.toolCalls.first?.argumentsJSON == #"{"location": "San Francisco, CA"}"#)
        #expect(parts.toolCalls.first?.id == "call_01")
        #expect(parts.toolCalls.first?.itemID == "fc_01")
        #expect(parts.response?.toolCalls.count == 1)
    }

    @Test
    func `Reasoning is kept apart from the answer`() throws {
        let parts = try drainCodex(Self.toolCall)

        #expect(parts.reasoning == "The user wants today's weather.")
        #expect(parts.text == "Let me check.")
    }

    @Test
    func `Encrypted reasoning is kept for the next turn to replay`() throws {
        #expect(try drainCodex(Self.toolCall).response?.reasoningItems.count == 1)
    }

    @Test
    func `An event that is not a text delta surfaces nothing`() throws {
        let parts = try drainCodex("""
        data: {"type":"response.content_part.added","item_id":"msg_01","part":{"type":"output_text","text":""}}

        data: {"type":"response.created","response":{"output":[]}}
        """)

        #expect(parts.text.isEmpty)
    }

    @Test
    func `The final full text is not replayed as a delta`() throws {
        // A caller summing deltas would otherwise be handed the answer twice.
        let parts = try drainCodex("""
        data: {"type":"response.completed","response":{"status":"completed","output_text":"the whole answer","output":[]}}
        """)

        #expect(parts.text.isEmpty)
        #expect(parts.response?.text == "the whole answer")
    }

    @Test
    func `A failure mid-stream is thrown rather than ending the answer quietly`() throws {
        let trace = """
        data: {"type":"response.output_text.delta","item_id":"msg_06","delta":"Half an "}

        data: {"type":"response.failed","response":{"id":"resp_06","status":"failed","error":{"code":"server_error","message":"The model is overloaded."}}}
        """

        #expect(throws: CodexLanguageModelError.self) {
            try drainCodex(trace)
        }
    }

    @Test
    func `A stream that stops talking reports no reason at all`() throws {
        let parts = try drainCodex("""
        data: {"type":"response.output_text.delta","item_id":"msg_07","delta":"Half an "}
        """)

        #expect(parts.response?.report.stopReason == nil)
        #expect(parts.response?.text == "Half an ")
    }
}
