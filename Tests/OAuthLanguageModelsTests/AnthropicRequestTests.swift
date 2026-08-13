import AnyLanguageModel
import Foundation
import Testing
@testable import OAuthLanguageModels

@Suite("Anthropic request shape")
struct AnthropicRequestTests {
    // MARK: Internal

    @Test
    func `A message ending in an image still carries the cache breakpoint`() throws {
        var message = AnthropicRequest.Message(
            role: "user",
            content: [
                .text(.init(text: "What is in this?")),
                .image(.init(base64Data: "aGk=", mimeType: "image/png"))
            ]
        )
        message.markLastBlockCached(with: .ephemeral)

        // Without a breakpoint here the whole conversation behind this message is read at
        // full price on every later turn, and nothing says so.
        #expect(try Self.cacheControlIndices(in: message) == [1])
    }

    @Test
    func `A message ending in a tool result carries the breakpoint on the result`() throws {
        var message = AnthropicRequest.Message(
            role: "user",
            content: [.toolResult(.init(toolUseID: "toolu_01", content: [.text(.init(text: "17C"))]))]
        )
        message.markLastBlockCached(with: .ephemeral)

        #expect(try Self.cacheControlIndices(in: message) == [0])
    }

    @Test
    func `A message ending in a tool call carries the breakpoint on the call`() throws {
        var message = AnthropicRequest.Message(
            role: "assistant",
            content: [
                .text(.init(text: "Checking.")),
                .toolUse(.init(id: "toolu_01", name: "get_weather", input: [:]))
            ]
        )
        message.markLastBlockCached(with: .ephemeral)

        #expect(try Self.cacheControlIndices(in: message) == [1])
    }

    @Test
    func `A message ending in thinking puts the breakpoint on the block before it`() throws {
        var message = AnthropicRequest.Message(
            role: "assistant",
            content: [
                .text(.init(text: "The GCD is 21.")),
                .thinking(.init(thinking: "1071 = 2 x 462 + 147", signature: "EqQB"))
            ]
        )
        message.markLastBlockCached(with: .ephemeral)

        // A thinking block is refused `cache_control`; it is cached implicitly with the
        // turn around it, so the walk has to step back past it rather than give up.
        #expect(try Self.cacheControlIndices(in: message) == [0])
    }

    @Test
    func `The system preamble and the last message both carry a breakpoint`() async throws {
        let model = AnthropicOAuthLanguageModel(tokenProvider: { "token" }, model: "claude-opus-5")
        let request = try await model.makeRequest(
            streaming: true,
            messages: [.init(role: "user", content: [.text(.init(text: "Hello"))])],
            instructions: "Be brief.",
            tools: nil,
            parameters: AnthropicRequestParameters()
        )
        let body = try #require(Self.decode(request.httpBody).objectValue)
        let system = try #require(body["system"]?.arrayValue)

        #expect(system.first?.objectValue?["text"]?.stringValue == claudeCodeSystemPreamble)
        #expect(system.last?.objectValue?["cache_control"] != nil)
        #expect(body["messages"]?.arrayValue?.last?.objectValue?["content"]?
            .arrayValue?.last?.objectValue?["cache_control"] != nil)
    }

    @Test
    func `A key the OAuth request shape depends on is dropped`() throws {
        let body = try Self.encoded(extraBody: [
            "model": .string("gpt-5"),
            "system": .string("You are a pirate."),
            "speed": .string("fast")
        ])

        #expect(body["model"]?.stringValue == "claude-opus-5")
        #expect(body["system"]?.arrayValue?.count == 1)
        // Everything else still lands: dropping is about the request shape, not about
        // refusing the caller.
        #expect(body["speed"]?.stringValue == "fast")
    }

    @Test
    func `A field added to an object the package builds joins it rather than replacing it`() throws {
        let body = try Self.encoded(
            thinkingBudgetTokens: 4096,
            extraBody: ["thinking": .object(["display": .string("omitted")])]
        )
        let thinking = body["thinking"]?.objectValue

        #expect(thinking?["display"]?.stringValue == "omitted")
        #expect(thinking?["budget_tokens"]?.intValue == 4096)
        #expect(thinking?["type"]?.stringValue == "enabled")
    }

    @Test
    func `A key the package does not build is added whole`() throws {
        let body = try Self.encoded(extraBody: ["output_config": .object(["effort": .string("high")])])

        #expect(body["output_config"]?.objectValue?["effort"]?.stringValue == "high")
    }

    @Test
    func `A header the OAuth request shape depends on is dropped`() async throws {
        let model = AnthropicOAuthLanguageModel(
            tokenProvider: { "token" },
            model: "claude-opus-5",
            extraHeaders: ["user-agent": "curl/8", "x-trace-id": "abc"]
        )
        let request = try await model.makeRequest(
            streaming: false,
            messages: [],
            instructions: nil,
            tools: nil,
            parameters: AnthropicRequestParameters()
        )

        #expect(request.value(forHTTPHeaderField: "user-agent")?.hasPrefix("claude-cli/") == true)
        #expect(request.value(forHTTPHeaderField: "x-trace-id") == "abc")
    }

    // MARK: Private

    /// The indices of the blocks that came out carrying a `cache_control`, read back off
    /// the encoded body rather than the model, since the wire is what matters.
    private static func cacheControlIndices(in message: AnthropicRequest.Message) throws -> [Int] {
        let content = try #require(
            decode(JSONEncoder.snakeCase.encode(message)).objectValue?["content"]?.arrayValue
        )
        return content.indices.filter { content[$0].objectValue?["cache_control"] != nil }
    }

    private static func encoded(
        thinkingBudgetTokens: Int? = nil,
        extraBody: [String: JSONValue]
    ) throws -> [String: JSONValue] {
        let request = AnthropicRequest(
            model: "claude-opus-5",
            maxTokens: 4096,
            system: [.init(text: claudeCodeSystemPreamble)],
            messages: [.init(role: "user", content: [.text(.init(text: "Hello"))])],
            tools: nil,
            thinking: thinkingBudgetTokens.map { .init(budgetTokens: $0) }
        )
        let data = try AnthropicOAuthLanguageModel.encodeBody(request, mergingExtraBody: extraBody)
        return try #require(decode(data).objectValue)
    }

    private static func decode(_ data: Data?) -> JSONValue {
        guard let data, let value = try? JSONDecoder().decode(JSONValue.self, from: data) else {
            return .null
        }
        return value
    }
}
