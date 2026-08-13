import AnyLanguageModel
import Foundation
import Testing
@testable import OAuthLanguageModels

/// Every exchange driven against the stub, under one roof.
///
/// `.serialized` covers a suite and everything nested inside it, but not its siblings —
/// and the stub is a single `URLProtocol` reached through `URLSession.shared`, so two
/// suites running side by side would answer each other's requests. Nesting them here is
/// what keeps them apart.
@Suite("Streamed exchanges", .serialized)
struct StreamedExchange {}

/// The turns a stubbed provider will serve, and what it was asked for, shared between a
/// test and the `URLProtocol` the request lands in.
final class Exchange: @unchecked Sendable {
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
        repeatsLast = false
    }

    func record(_ body: String) {
        lock.lock(); defer { lock.unlock() }
        sent.append(body)
    }

    func next() -> String {
        lock.lock(); defer { lock.unlock() }
        if pending.count == 1, repeatsLast { return pending[0] }
        return pending.isEmpty ? "" : pending.removeFirst()
    }

    /// Serves the last turn over and over once the rest have been handed out, for the
    /// exchanges that are meant never to settle.
    func serveForever(_ turns: [String]) {
        lock.lock(); defer { lock.unlock() }
        pending = turns
        sent = []
        repeatsLast = true
    }

    // MARK: Private

    private let lock = NSLock()
    private var pending: [String] = []
    private var sent: [String] = []
    private var repeatsLast = false
}

// MARK: - StubProtocol

/// Answers every request out of ``Exchange`` instead of the network.
final class StubProtocol: URLProtocol {
    // MARK: Internal

    /// Registers the stub for the rest of the process.
    ///
    /// Never unregistered: a test that took it away while another was mid-request would
    /// send that request to the real endpoint.
    static func install() {
        guard !isInstalled else { return }
        isInstalled = true
        URLProtocol.registerClass(StubProtocol.self)
    }

    override class func canInit(with _: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Exchange.shared.record(Self.body(of: request))
        guard let url = request.url,
              let response = HTTPURLResponse(
                  url: url,
                  statusCode: 200,
                  httpVersion: "HTTP/1.1",
                  headerFields: ["Content-Type": "text/event-stream"]
              ) else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(Exchange.shared.next().utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    // MARK: Private

    nonisolated(unsafe) private static var isInstalled = false

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

// MARK: - WeatherTool

struct WeatherTool: Tool {
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

// MARK: - Counter

final class Counter: @unchecked Sendable {
    // MARK: Internal

    var count: Int {
        lock.lock(); defer { lock.unlock() }
        return value
    }

    func increment() {
        lock.lock(); defer { lock.unlock() }
        value += 1
    }

    // MARK: Private

    private let lock = NSLock()
    private var value = 0
}

// MARK: - Reports

final class Reports: @unchecked Sendable {
    // MARK: Internal

    var values: [TurnReport] {
        lock.lock(); defer { lock.unlock() }
        return storage
    }

    var handler: @Sendable (GenerationEvent) -> Void {
        { event in
            guard case let .turnFinished(report) = event else { return }
            self.append(report)
        }
    }

    // MARK: Private

    private let lock = NSLock()
    private var storage: [TurnReport] = []

    private func append(_ report: TurnReport) {
        lock.lock(); defer { lock.unlock() }
        storage.append(report)
    }
}

// MARK: - ToolCalls

/// The tool calls reported through a model's event channel.
final class ToolCalls: @unchecked Sendable {
    // MARK: Internal

    var values: [(name: String, arguments: String)] {
        lock.lock(); defer { lock.unlock() }
        return storage
    }

    var handler: @Sendable (GenerationEvent) -> Void {
        { event in
            guard case let .toolCall(name, arguments) = event else { return }
            self.lock.lock(); defer { self.lock.unlock() }
            self.storage.append((name, arguments))
        }
    }

    // MARK: Private

    private let lock = NSLock()
    private var storage: [(name: String, arguments: String)] = []
}

/// Runs a whole streamed exchange against `turns`, and hands back every snapshot the
/// caller would have seen.
func streamedSnapshots(
    from model: some AnyLanguageModel.LanguageModel,
    turns: [String],
    tools: [any Tool],
    repeatingLastTurn: Bool = false,
    delegate: (any ToolExecutionDelegate)? = nil
) async throws -> [String] {
    StubProtocol.install()
    if repeatingLastTurn {
        Exchange.shared.serveForever(turns)
    } else {
        Exchange.shared.serve(turns)
    }

    let session = LanguageModelSession(model: model, tools: tools, transcript: Transcript())
    session.toolExecutionDelegate = delegate
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

/// Runs a whole streamed exchange against `turns`, and hands back every partially
/// generated value the caller would have seen.
func streamedStructured<Content: Generable>(
    from model: some AnyLanguageModel.LanguageModel,
    turns: [String],
    tools: [any Tool] = [],
    generating type: Content.Type
) async throws -> [Content.PartiallyGenerated] where Content.PartiallyGenerated: Sendable {
    StubProtocol.install()
    Exchange.shared.serve(turns)

    let session = LanguageModelSession(model: model, tools: tools, transcript: Transcript())
    var snapshots: [Content.PartiallyGenerated] = []
    for try await snapshot in model.streamResponse(
        within: session,
        to: Prompt("What is the weather in Athens?"),
        generating: type,
        includeSchemaInPrompt: false,
        options: GenerationOptions()
    ) {
        snapshots.append(snapshot.content)
    }
    return snapshots
}
