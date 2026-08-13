import AnyLanguageModel
import Foundation
import Testing
@testable import OAuthLanguageModels

@Suite("Codex request shape")
struct CodexRequestTests {
    // MARK: Internal

    @Test
    func `A key the OAuth request shape depends on is dropped`() throws {
        let body = try Self.body(extraBody: [
            "model": .string("claude-opus-5"),
            "store": .bool(true),
            "service_tier": .string("priority")
        ])

        #expect(body["model"]?.stringValue == "gpt-5")
        #expect(body["store"]?.boolValue == false)
        // Everything else still lands: dropping is about the request shape, not about
        // refusing the caller.
        #expect(body["service_tier"]?.stringValue == "priority")
    }

    @Test
    func `A field added to an object the package builds joins it rather than replacing it`() throws {
        let body = try Self.body(extraBody: ["text": .object(["format": .object(["type": .string("text")])])])
        let text = body["text"]?.objectValue

        #expect(text?["format"]?.objectValue?["type"]?.stringValue == "text")
        // Nothing the package put there has to be restated to add one field beside it.
        #expect(text?["verbosity"]?.stringValue == "medium")
    }

    @Test
    func `A key the package does not build is added whole`() throws {
        let body = try Self.body(extraBody: ["metadata": .object(["run": .string("42")])])

        #expect(body["metadata"]?.objectValue?["run"]?.stringValue == "42")
    }

    // MARK: Private

    private static func body(extraBody: [String: JSONValue]) throws -> [String: JSONValue] {
        var parameters = CodexRequestParameters()
        parameters.extraBody = extraBody
        let value = CodexLanguageModel.makeRequestBody(
            model: "gpt-5",
            instructions: "Be brief.",
            inputs: [],
            tools: nil,
            promptCacheKey: "session",
            parameters: parameters
        )
        return try #require(value.objectValue)
    }
}
