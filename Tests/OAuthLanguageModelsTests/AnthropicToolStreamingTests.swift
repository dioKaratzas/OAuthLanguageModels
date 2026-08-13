import AnyLanguageModel
import Foundation
import Testing
@testable import OAuthLanguageModels

/// The turns a stubbed provider will serve, and what it was asked for, shared between the
/// test and the `URLProtocol` the request lands in.
private final class Exchange: @unchecked Sendable {
    // MARK: Internal

    nonisolated(unsafe) static let shared = Exchange()

    var requestBodies: [String] {
        lock.lock(); defer { lock.unlock() }
        return sent
    }

    func serve(_ turns: [String]) {
        lock.lock(); defer { lock.unlock() }
        pending = turns
        sent = []
    }

    func record(_ body: String) {
        lock.lock(); defer { lock.unlock() }
        sent.append(body)
    }

    func next() -> String {
        lock.lock(); defer { lock.unlock() }
        return pending.isEmpty ? "" : pending.removeFirst()
    }

    // MARK: Private

    private let lock = NSLock()
    private var pending: [String] = []
    private var sent: [String] = []
}

private final class StubProtocol: URLProtocol {
    override class func canInit(with _: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Exchange.shared.record(Self.body(of: request))
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(Exchange.shared.next().utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    /// `URLProtocol` hands the body over as a stream once the request has been sent.
    private static func body(of request: URLRequest) -> String {
        if let data = request.httpBody { return String(decoding: data, as: UTF8.self) }
        guard let stream = request.httpBodyStream else { return "" }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while stream.hasBytesAvailable {
            let read = stream.read(&buffer, maxLength: buffer.count)
            guard read > 0 else { break }
            data.append(contentsOf: buffer[0..<read])
        }
        return String(decoding: data, as: UTF8.self)
    }
}

private struct WeatherTool: Tool {
    let callCount: Counter

    var name: String { "get_weather" }
    var description: String { "Reports the weather somewhere." }
    var parameters: GenerationSchema {
        GenerationSchema(type: String.self, description: "The place.", properties: [])
    }

    func call(arguments _: GeneratedContent) async throws -> String {
        callCount.increment()
        return "17C and clear."
    }
}

private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    var count: Int {
        lock.lock(); defer { lock.unlock() }
        return value
    }

    func increment() {
        lock.lock(); defer { lock.unlock() }
        value += 1
    }
}

@Suite("Anthropic streaming with tools", .serialized)
struct AnthropicToolStreamingTests {
    // MARK: Internal

    /// The model says something, calls a tool, and is served again once the result is in.
    static let askingForTheTool = """
    data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"usage":{"input_tokens":40,"output_tokens":1}}}

    data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

    data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check. "}}

    data: {"type":"content_block_stop","index":0}

    data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}

    data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"location\\":"}}

    data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \\"Athens\\"}"}}

    data: {"type":"content_block_stop","index":1}

    data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":40}}

    data: {"type":"message_stop"}

    """

    static let answeringWithIt = """
    data: {"type":"message_start","message":{"id":"msg_02","type":"message","role":"assistant","content":[],"usage":{"input_tokens":90,"output_tokens":1}}}

    data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

    data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"It is 17C in Athens."}}

    data: {"type":"content_block_stop","index":0}

    data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":12}}

    data: {"type":"message_stop"}

    """

    @Test
    func `A streamed turn runs its tools and keeps writing afterwards`() async throws {
        let calls = Counter()
        let snapshots = try await Self.stream(
            turns: [Self.askingForTheTool, Self.answeringWithIt],
            tools: [WeatherTool(callCount: calls)]
        )

        #expect(calls.count == 1)
        // Text from both sides of the tool call, in one continuous answer: the snapshots
        // stay cumulative across the round trip rather than restarting after it.
        #expect(snapshots.last == "Let me check. It is 17C in Athens.")
        #expect(snapshots.contains("Let me check. "))
        // The fragments the arguments arrived in are not text and must never be shown as
        // such, so every snapshot is a prefix of the finished answer.
        #expect(snapshots.allSatisfy { snapshots.last?.hasPrefix($0) == true })
    }

    @Test
    func `The second turn carries the tool call and its result`() async throws {
        _ = try await Self.stream(
            turns: [Self.askingForTheTool, Self.answeringWithIt],
            tools: [WeatherTool(callCount: Counter())]
        )
        let bodies = Exchange.shared.requestBodies

        #expect(bodies.count == 2)
        // The tools have to travel with the streaming request too, or the model has
        // nothing to call.
        #expect(bodies.first?.contains(#""name":"get_weather""#) == true)
        #expect(bodies.last?.contains(#""type":"tool_use""#) == true)
        #expect(bodies.last?.contains(#""tool_use_id":"toolu_01""#) == true)
        #expect(bodies.last?.contains("17C and clear.") == true)
    }

    @Test
    func `Both turns are reported, each with its own usage`() async throws {
        let reports = Reports()
        _ = try await Self.stream(
            turns: [Self.askingForTheTool, Self.answeringWithIt],
            tools: [WeatherTool(callCount: Counter())],
            onEvent: { event in
                if case let .turnFinished(report) = event { reports.append(report) }
            }
        )

        #expect(reports.values.map(\.stopReason) == [.toolUse, .endTurn])
        #expect(reports.values.map(\.usage.inputTokens) == [40, 90])
    }

    // MARK: Private

    private final class Reports: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [TurnReport] = []

        var values: [TurnReport] {
            lock.lock(); defer { lock.unlock() }
            return storage
        }

        func append(_ report: TurnReport) {
            lock.lock(); defer { lock.unlock() }
            storage.append(report)
        }
    }

    private static func stream(
        turns: [String],
        tools: [any Tool],
        onEvent: (@Sendable (GenerationEvent) -> Void)? = nil
    ) async throws -> [String] {
        URLProtocol.registerClass(StubProtocol.self)
        defer { URLProtocol.unregisterClass(StubProtocol.self) }
        Exchange.shared.serve(turns)

        let model = AnthropicOAuthLanguageModel(
            tokenProvider: { "token" },
            model: "claude-opus-5",
            onEvent: onEvent
        )
        let session = LanguageModelSession(model: model, tools: tools, transcript: Transcript())
        var snapshots: [String] = []
        for try await snapshot in model.streamResponse(
            within: session,
            to: Prompt("What is the weather in Athens?"),
            generating: String.self,
            includeSchemaInPrompt: false,
            options: GenerationOptions()
        ) {
            snapshots.append(snapshot.content)
        }
        return snapshots
    }
}
