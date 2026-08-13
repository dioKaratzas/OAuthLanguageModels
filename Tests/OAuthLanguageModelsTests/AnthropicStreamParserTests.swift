import Foundation
import Testing
@testable import OAuthLanguageModels

@Suite("Anthropic SSE streaming")
struct AnthropicStreamParserTests {
    /// A plain answer that ran out of room: nothing was cached, and the ceiling stopped it
    /// rather than the model finishing.
    static let truncated = """
    event: message_start
    data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"usage":{"input_tokens":2679,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}

    event: content_block_start
    data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

    event: ping
    data: {"type": "ping"}

    event: content_block_delta
    data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"func parse("}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"line: String) -> Toke"}}

    event: content_block_stop
    data: {"type":"content_block_stop","index":0}

    event: message_delta
    data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":8192}}

    event: message_stop
    data: {"type":"message_stop"}
    """

    /// The same conversation a turn later: the prefix came back out of the cache.
    static let cached = """
    event: message_start
    data: {"type":"message_start","message":{"id":"msg_02","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"usage":{"input_tokens":14,"cache_creation_input_tokens":0,"cache_read_input_tokens":15234,"output_tokens":1}}}

    event: content_block_start
    data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Yes."}}

    event: content_block_stop
    data: {"type":"content_block_stop","index":0}

    event: message_delta
    data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4}}

    event: message_stop
    data: {"type":"message_stop"}
    """

    /// Text, then a tool call whose arguments arrive in six fragments.
    static let toolUse = """
    event: message_start
    data: {"type":"message_start","message":{"id":"msg_03","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"usage":{"input_tokens":472,"cache_creation_input_tokens":128,"cache_read_input_tokens":0,"output_tokens":2}}}

    event: content_block_start
    data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check."}}

    event: content_block_stop
    data: {"type":"content_block_stop","index":0}

    event: content_block_start
    data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"location\\":"}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \\"San"}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" Francisc"}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"o, CA\\"}"}}

    event: content_block_stop
    data: {"type":"content_block_stop","index":1}

    event: message_delta
    data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}

    event: message_stop
    data: {"type":"message_stop"}
    """

    /// A thinking block, signed, followed by the answer it led to.
    static let thinking = """
    event: message_start
    data: {"type":"message_start","message":{"id":"msg_04","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"usage":{"input_tokens":30,"output_tokens":1}}}

    event: content_block_start
    data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"1071 = 2 x 462 + 147"}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"\\n462 = 3 x 147 + 21"}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM1gbcDa9GJwZA2b3h"}}

    event: content_block_stop
    data: {"type":"content_block_stop","index":0}

    event: content_block_start
    data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The GCD is 21."}}

    event: content_block_stop
    data: {"type":"content_block_stop","index":1}

    event: message_delta
    data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":320}}

    event: message_stop
    data: {"type":"message_stop"}
    """

    @Test
    func `A turn that read nothing from the cache reports it as full-price input`() throws {
        let report = try drainAnthropic(Self.truncated).report

        #expect(report?.usage == TokenUsage(inputTokens: 2679, outputTokens: 8192))
        #expect(report?.usage.cacheReadTokens == 0)
    }

    @Test
    func `A cached turn reports the prefix as read rather than as input`() throws {
        let report = try drainAnthropic(Self.cached).report

        // The whole point of the breakpoints: 15k tokens of brief cost a tenth here, and
        // without this number there is no way to tell that from paying for it again.
        #expect(report?.usage.cacheReadTokens == 15234)
        #expect(report?.usage.inputTokens == 14)
        #expect(report?.usage.totalInputTokens == 15248)
    }

    @Test
    func `A turn says what its prompt cost before it writes anything`() throws {
        let parts = try drainAnthropic(Self.cached)

        // `message_start` carries the whole prompt side, so a caller can show what a
        // question cost to ask without waiting for the answer.
        guard case let .started(opening) = parts.first else {
            Issue.record("The turn did not open with its prompt count.")
            return
        }
        #expect(opening.inputTokens == 14)
        #expect(opening.cacheReadTokens == 15234)
        // Nothing has been written yet; the one token counted here measures nothing.
        #expect(opening.outputTokens == 0)
    }

    @Test
    func `A cache write is counted apart from a cache read`() throws {
        let report = try drainAnthropic(Self.toolUse).report

        #expect(report?.usage.cacheWriteTokens == 128)
        #expect(report?.usage.cacheReadTokens == 0)
    }

    @Test
    func `An answer cut off at the ceiling is reported as truncated`() throws {
        #expect(try drainAnthropic(Self.truncated).report?.stopReason == .maxTokens)
    }

    @Test
    func `An answer that finished is not reported as truncated`() throws {
        #expect(try drainAnthropic(Self.cached).report?.stopReason == .endTurn)
    }

    @Test
    func `A turn waiting on tools says so`() throws {
        #expect(try drainAnthropic(Self.toolUse).report?.stopReason == .toolUse)
    }

    @Test
    func `A tool call is handed over once, whole, with the text around it intact`() throws {
        let parts = try drainAnthropic(Self.toolUse)

        #expect(parts.text == "Let me check.")
        #expect(parts.toolUses.count == 1)
        // Six fragments went in; a caller must never see the half of one that parses as
        // nothing, so the call surfaces only once its arguments close.
        #expect(try parts.toolUses.first?.argumentsJSONString() == #"{"location":"San Francisco, CA"}"#)
        #expect(parts.toolUses.first?.name == "get_weather")
        #expect(parts.toolUses.first?.id == "toolu_01")
    }

    @Test
    func `Reasoning is kept apart from the answer`() throws {
        let parts = try drainAnthropic(Self.thinking)

        #expect(parts.thinking == "1071 = 2 x 462 + 147\n462 = 3 x 147 + 21")
        #expect(parts.text == "The GCD is 21.")
    }

    @Test
    func `A signed thinking block is replayed as the provider sent it`() throws {
        let content = try drainAnthropic(Self.thinking).replayedContent

        // Anthropic refuses a tool round whose thinking block comes back unsigned.
        guard case let .thinking(block) = content.first else {
            Issue.record("The turn did not replay its thinking block.")
            return
        }
        #expect(block.signature == "EqQBCgIYAhIM1gbcDa9GJwZA2b3h")
    }

    @Test
    func `The assistant turn is replayed in the order it was written`() throws {
        let content = try drainAnthropic(Self.toolUse).replayedContent

        #expect(content.count == 2)
        if case .text = content.first {} else { Issue.record("The text block was not replayed first.") }
        if case .toolUse = content.last {} else { Issue.record("The tool call was not replayed last.") }
    }

    @Test
    func `A block that opened and closed empty is left out of the replayed turn`() throws {
        let trace = """
        event: content_block_start
        data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

        event: content_block_stop
        data: {"type":"content_block_stop","index":0}

        event: content_block_start
        data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}

        event: content_block_stop
        data: {"type":"content_block_stop","index":1}

        event: message_stop
        data: {"type":"message_stop"}
        """
        let content = try drainAnthropic(trace).replayedContent

        // A model that goes straight to a tool call still opens a text block, and the API
        // refuses an assistant turn that comes back carrying an empty one.
        #expect(content.count == 1)
        if case .toolUse = content.first {} else { Issue.record("The tool call was not replayed.") }
    }

    @Test
    func `An error mid-stream is thrown rather than ending the answer quietly`() throws {
        let trace = """
        event: content_block_start
        data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

        event: content_block_delta
        data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Half an "}}

        event: error
        data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}
        """

        #expect(throws: AnthropicOAuthLanguageModelError.self) {
            try drainAnthropic(trace)
        }
    }

    @Test
    func `A stream that stops talking reports no reason at all`() throws {
        let trace = """
        event: message_start
        data: {"type":"message_start","message":{"id":"msg_05","type":"message","role":"assistant","content":[],"usage":{"input_tokens":9,"output_tokens":1}}}

        event: content_block_start
        data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

        event: content_block_delta
        data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Half an "}}
        """
        let parts = try drainAnthropic(trace)

        // A dropped connection and a finished answer have to be distinguishable, so the
        // terminal report is still made — with nothing in the stop reason.
        #expect(parts.report?.stopReason == nil)
        // The text that did arrive is still replayable as the assistant turn.
        #expect(parts.replayedContent.count == 1)
    }

    @Test
    func `A block type the package does not model does not fail the answer around it`() throws {
        let trace = """
        event: content_block_start
        data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_01","name":"web_search","input":{}}}

        event: content_block_stop
        data: {"type":"content_block_stop","index":0}

        event: content_block_start
        data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

        event: content_block_delta
        data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"It is raining."}}

        event: content_block_stop
        data: {"type":"content_block_stop","index":1}

        event: message_stop
        data: {"type":"message_stop"}
        """

        #expect(try drainAnthropic(trace).text == "It is raining.")
    }
}
