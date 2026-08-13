import AnyLanguageModel
import Foundation
import Testing
@testable import OAuthLanguageModels

/// Serialized, and asserting by `contains`: the observer is one global, and other suites
/// drop reserved keys of their own while this one runs.
@Suite("Dropped keys are reported", .serialized)
struct DroppedKeyLoggingTests {
    // MARK: Internal

    @Test
    func `A reserved Anthropic body key is named as dropped`() throws {
        let dropped = try Self.observing {
            _ = try AnthropicOAuthLanguageModel.encodeBody(
                Self.request,
                mergingExtraBody: ["messages": .array([]), "speed": .string("fast")]
            )
        }

        #expect(dropped.contains(Drop(kind: "body key", name: "messages", source: "extraBody")))
        // Only the reserved one: a key that is merely unusual is not worth a line.
        #expect(!dropped.contains { $0.name == "speed" })
    }

    @Test
    func `A reserved Anthropic header is named as dropped`() async throws {
        let model = AnthropicOAuthLanguageModel(
            tokenProvider: { "token" },
            model: "claude-opus-5",
            extraHeaders: ["User-Agent": "curl/8", "x-trace-id": "abc"]
        )
        let dropped = try await Self.observing {
            _ = try await model.makeRequest(
                streaming: false,
                messages: [],
                instructions: nil,
                tools: nil,
                parameters: AnthropicRequestParameters()
            )
        }

        // Matched case-insensitively, and reported under the caller's own spelling.
        #expect(dropped.contains(Drop(kind: "header field", name: "User-Agent", source: "extraHeaders")))
        #expect(!dropped.contains { $0.name == "x-trace-id" })
    }

    @Test
    func `A reserved Codex body key is named as dropped`() throws {
        var parameters = CodexRequestParameters()
        parameters.extraBody = ["store": .bool(true), "service_tier": .string("priority")]
        let dropped = try Self.observing {
            _ = CodexLanguageModel.makeRequestBody(
                model: "gpt-5",
                instructions: "Be brief.",
                inputs: [],
                tools: nil,
                promptCacheKey: "session",
                parameters: parameters
            )
        }

        #expect(dropped.contains(Drop(kind: "body key", name: "store", source: "extraBody")))
        #expect(!dropped.contains { $0.name == "service_tier" })
    }

    // MARK: Private

    private struct Drop: Hashable {
        let kind: String
        let name: String
        let source: String
    }

    private static let request = AnthropicRequest(
        model: "claude-opus-5",
        maxTokens: 4096,
        system: [.init(text: claudeCodeSystemPreamble)],
        messages: [.init(role: "user", content: [.text(.init(text: "Hello"))])],
        tools: nil
    )

    /// Runs `work` with the log observer installed, and hands back what it saw.
    private static func observing(_ work: () async throws -> Void) async rethrows -> [Drop] {
        let seen = Seen()
        droppedKeyObserver = { kind, name, source in
            seen.append(Drop(kind: kind, name: name, source: source))
        }
        defer { droppedKeyObserver = nil }
        try await work()
        return seen.values
    }

    private static func observing(_ work: () throws -> Void) rethrows -> [Drop] {
        let seen = Seen()
        droppedKeyObserver = { kind, name, source in
            seen.append(Drop(kind: kind, name: name, source: source))
        }
        defer { droppedKeyObserver = nil }
        try work()
        return seen.values
    }

    private final class Seen: @unchecked Sendable {
        var values: [Drop] {
            lock.lock(); defer { lock.unlock() }
            return storage
        }

        func append(_ drop: Drop) {
            lock.lock(); defer { lock.unlock() }
            storage.append(drop)
        }

        private let lock = NSLock()
        private var storage: [Drop] = []
    }
}
