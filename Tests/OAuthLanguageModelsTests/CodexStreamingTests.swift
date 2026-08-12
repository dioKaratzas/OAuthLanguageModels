import Foundation
import Testing
import AnyLanguageModel
@testable import OAuthLanguageModels

/// Somewhere for the synchronous `onDelta` callback to record into. The callback fires
/// inline while `processEvent` runs, so a plain reference class under a lock is enough.
private final class Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = [String]()

    func append(_ value: String) {
        lock.lock(); defer { lock.unlock() }
        storage.append(value)
    }

    var values: [String] {
        lock.lock(); defer { lock.unlock() }
        return storage
    }
}

@Suite("Codex SSE streaming")
struct CodexStreamingTests {
    private func drain(
        _ events: [String]
    ) throws -> (text: String, deltas: [String]) {
        var accumulatedText = ""
        var latestOutput: [JSONValue]?
        var latestOutputText: String?
        var toolCallsByID: [String: CodexToolCall] = [:]
        let deltas = Recorder()

        for payload in events {
            try CodexLanguageModel.processEvent(
                payload,
                accumulatedText: &accumulatedText,
                latestOutput: &latestOutput,
                latestOutputText: &latestOutputText,
                toolCallsByID: &toolCallsByID,
                onDelta: { deltas.append($0) }
            )
        }
        return (accumulatedText, deltas.values)
    }

    @Test
    func `Each text delta is surfaced as it is parsed, in order`() throws {
        let result = try drain([
            #"{"type":"response.output_text.delta","delta":"A dependency "}"#,
            #"{"type":"response.output_text.delta","delta":"graph."}"#,
        ])

        // The deltas reach the caller one at a time rather than only as a final blob, which
        // is what lets a UI render the answer as it arrives.
        #expect(result.deltas == ["A dependency ", "graph."])
        #expect(result.text == "A dependency graph.")
    }

    @Test
    func `An event that is not a text delta surfaces nothing`() throws {
        let result = try drain([
            #"{"type":"response.reasoning.delta","delta":"thinking"}"#,
            #"{"type":"response.created","response":{"output":[]}}"#,
        ])

        #expect(result.deltas.isEmpty)
        #expect(result.text.isEmpty)
    }

    @Test
    func `A stream with no deltas leaves the caller nothing to render early`() throws {
        // The final full-text event still lands, but it is not a delta and must not be
        // replayed as one — a caller summing deltas would otherwise double the answer.
        let result = try drain([
            #"{"type":"response.completed","response":{"output_text":"the whole answer"}}"#,
        ])

        #expect(result.deltas.isEmpty)
    }
}
