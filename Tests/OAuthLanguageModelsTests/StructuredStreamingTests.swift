import AnyLanguageModel
import Foundation
import Testing
@testable import OAuthLanguageModels

/// A two-field type, so a snapshot can be checked for having one field and not yet the
/// other.
@Generable
struct Forecast: Equatable {
    @Guide(description: "Where the weather is.")
    var place: String
    @Guide(description: "Degrees celsius.")
    var degrees: Int
}

/// A type with an optional property, which is what strict mode cannot express.
@Generable
struct Sighting: Equatable {
    @Guide(description: "Where.")
    var place: String
    @Guide(description: "Anything else worth saying.")
    var note: String?
}

/// Both providers writing a structured answer, against a stubbed endpoint.
///
/// Serialized for the same reason as the tool suite: the stub is reached through
/// `URLSession.shared`.
extension StreamedExchange {
    @Suite("structured")
    struct StructuredStreamingTests {
        // MARK: Internal

        enum Anthropic {
            /// The JSON arrives in three text deltas, as any other answer would.
            static let json = """
            data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"usage":{"input_tokens":40,"output_tokens":1}}}

            data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

            data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\\"place\\": \\"Ath"}}

            data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ens\\", \\"degre"}}

            data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"es\\": 17}"}}

            data: {"type":"content_block_stop","index":0}

            data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":20}}

            data: {"type":"message_stop"}

            """

            /// The ceiling stopped it halfway through the second field.
            static let cutOff = """
            data: {"type":"message_start","message":{"id":"msg_02","type":"message","role":"assistant","content":[],"usage":{"input_tokens":40,"output_tokens":1}}}

            data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

            data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\\"place\\": \\"Athens\\", \\"degre"}}

            data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":8192}}

            data: {"type":"message_stop"}

            """

            /// What a model asked in the prompt alone tends to write.
            static let fenced = """
            data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

            data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"```json\\n{\\"place\\": \\"Athens\\", \\"degrees\\": 17}\\n```"}}

            data: {"type":"content_block_stop","index":0}

            data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":20}}

            data: {"type":"message_stop"}

            """

            /// A refusal where a schema was asked for.
            static let prose = """
            data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

            data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I am afraid I cannot say."}}

            data: {"type":"content_block_stop","index":0}

            data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":20}}

            data: {"type":"message_stop"}

            """

            static let askingForTheTool = """
            data: {"type":"message_start","message":{"id":"msg_03","type":"message","role":"assistant","content":[],"usage":{"input_tokens":40,"output_tokens":1}}}

            data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}

            data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"location\\": \\"Athens\\"}"}}

            data: {"type":"content_block_stop","index":0}

            data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":40}}

            data: {"type":"message_stop"}

            """
        }

        enum Codex {
            static let json = """
            data: {"type":"response.created","response":{"id":"resp_01","status":"in_progress","output":[]}}

            data: {"type":"response.output_text.delta","item_id":"msg_01","delta":"{\\"place\\": \\"Ath"}

            data: {"type":"response.output_text.delta","item_id":"msg_01","delta":"ens\\", \\"degre"}

            data: {"type":"response.output_text.delta","item_id":"msg_01","delta":"es\\": 17}"}

            data: {"type":"response.completed","response":{"id":"resp_01","status":"completed","output":[{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"output_text","text":"{\\"place\\": \\"Athens\\", \\"degrees\\": 17}"}]}],"usage":{"input_tokens":40,"input_tokens_details":{"cached_tokens":0},"output_tokens":20}}}

            """

            static let cutOff = """
            data: {"type":"response.output_text.delta","item_id":"msg_02","delta":"{\\"place\\": \\"Athens\\", \\"degre"}

            data: {"type":"response.incomplete","response":{"id":"resp_02","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[],"usage":{"input_tokens":40,"input_tokens_details":{"cached_tokens":0},"output_tokens":8192}}}

            """
        }

        @Test
        func `Anthropic fills the fields one at a time as the JSON is written`() async throws {
            let snapshots = try await streamedStructured(
                from: Self.anthropic(),
                turns: [Anthropic.json],
                generating: Forecast.self
            )

            Self.expectFilledInOrder(snapshots)
        }

        @Test
        func `Codex fills the fields one at a time as the JSON is written`() async throws {
            let snapshots = try await streamedStructured(
                from: Self.codex(),
                turns: [Codex.json],
                generating: Forecast.self
            )

            Self.expectFilledInOrder(snapshots)
        }

        @Test
        func `A field that has arrived is never taken back`() async throws {
            let snapshots = try await streamedStructured(
                from: Self.anthropic(),
                turns: [Anthropic.json],
                generating: Forecast.self
            )

            // Fields arrive in the order the type declares them, so a caller reading one off
            // a snapshot must never find it nil again on the next.
            var seenPlace = false
            for snapshot in snapshots {
                if snapshot.place != nil { seenPlace = true }
                #expect(!seenPlace || snapshot.place != nil)
            }
        }

        @Test
        func `An answer cut off mid-JSON is reported as truncated rather than as a decode failure`() async throws {
            let reports = Reports()
            let snapshots = try await streamedStructured(
                from: Self.anthropic(onEvent: reports.handler),
                turns: [Anthropic.cutOff],
                generating: Forecast.self
            )

            // Half a JSON document is what a cut-off answer looks like; the reason it stopped
            // is what says so, and throwing here would hide it behind a parse error.
            #expect(reports.values.last?.stopReason == .maxTokens)
            #expect(snapshots.last?.place == "Athens")
            #expect(snapshots.last?.degrees == nil)
        }

        @Test
        func `A Codex answer cut off mid-JSON is reported as truncated too`() async throws {
            let reports = Reports()
            let snapshots = try await streamedStructured(
                from: Self.codex(onEvent: reports.handler),
                turns: [Codex.cutOff],
                generating: Forecast.self
            )

            #expect(reports.values.last?.stopReason == .maxTokens)
            #expect(snapshots.last?.place == "Athens")
        }

        @Test
        func `A structured turn runs its tools first and still fills the fields`() async throws {
            let calls = Counter()
            let snapshots = try await streamedStructured(
                from: Self.anthropic(),
                turns: [Anthropic.askingForTheTool, Anthropic.json],
                tools: [WeatherTool(callCount: calls)],
                generating: Forecast.self
            )

            #expect(calls.count == 1)
            #expect(snapshots.last?.place == "Athens")
            #expect(snapshots.last?.degrees == 17)
        }

        @Test
        func `The schema travels with the Anthropic request`() async throws {
            _ = try await streamedStructured(
                from: Self.anthropic(),
                turns: [Anthropic.json],
                generating: Forecast.self
            )
            let body = try #require(Exchange.shared.requestBodies.first)

            // `output_config` is where the answer's shape is asked for; a caller adding
            // `effort` under the same key through `extraBody` merges with it rather than
            // replacing it.
            #expect(body.contains(#""output_config":{"format":{"schema":"#))
            #expect(body.contains(#""type":"json_schema""#))
            #expect(body.contains(#""place""#))
        }

        @Test
        func `The schema travels with the Codex request`() async throws {
            _ = try await streamedStructured(
                from: Self.codex(),
                turns: [Codex.json],
                generating: Forecast.self
            )
            let body = try #require(Exchange.shared.requestBodies.first)

            #expect(body.contains(#""format":{"name":"Forecast""#))
            // No `strict`: it demands every property be required, which a type with an
            // optional field can never satisfy.
            #expect(!body.contains("strict"))
            #expect(body.contains(#""place""#))
        }

        @Test
        func `The schema joins an output_config a caller already set`() async throws {
            var options = GenerationOptions()
            options[custom: AnthropicOAuthLanguageModel.self] = .init(
                extraBody: ["output_config": .object(["effort": .string("high")])]
            )
            StubProtocol.install()
            Exchange.shared.serve([Anthropic.json])
            let model = Self.anthropic()
            let session = LanguageModelSession(model: model, transcript: Transcript())
            for try await _ in model.streamResponse(
                within: session,
                to: Prompt("What is the weather in Athens?"),
                generating: Forecast.self,
                includeSchemaInPrompt: false,
                options: options
            ) {}
            let body = try #require(Exchange.shared.requestBodies.first)

            // Both halves survive: the effort a caller dials and the shape the package
            // asks for live under the same key, and one must not erase the other.
            #expect(body.contains(#""effort":"high""#))
            #expect(body.contains(#""type":"json_schema""#))
        }

        @Test
        func `A whole Codex answer decodes into the value it describes`() async throws {
            StubProtocol.install()
            Exchange.shared.serve([Codex.json])
            let model = Self.codex()
            let session = LanguageModelSession(model: model, transcript: Transcript())
            let response = try await model.respond(
                within: session,
                to: Prompt("What is the weather in Athens?"),
                generating: Forecast.self,
                includeSchemaInPrompt: false,
                options: GenerationOptions()
            )

            #expect(response.content.place == "Athens")
            #expect(response.content.degrees == 17)
        }

        @Test
        func `A fenced answer is read out of its fence`() async throws {
            let snapshots = try await streamedStructured(
                from: Self.anthropic(),
                turns: [Anthropic.fenced],
                generating: Forecast.self
            )

            // A response format leaves no fence, but a model asked in the prompt alone
            // writes one, and the answer inside it is still the answer.
            #expect(snapshots.last?.place == "Athens")
            #expect(snapshots.last?.degrees == 17)
        }

        @Test
        func `An answer in prose where JSON was asked for is reported, not swallowed`() async throws {
            await #expect(throws: (any Error).self) {
                try await streamedStructured(
                    from: Self.anthropic(),
                    turns: [Anthropic.prose],
                    generating: Forecast.self
                )
            }
        }

        @Test
        func `An optional property stays optional in the schema that goes out`() throws {
            let produced = try structuredSchema(for: Sighting.self)
            let schema = try #require(produced)
            let names = schema.objectValue?["required"]?.arrayValue?.compactMap(\.stringValue)
            let required = try #require(names)

            // Strict mode would demand `note` here and refuse the request without it, so
            // a type with one optional property could not be generated at all.
            #expect(required == ["place"])
        }

        @Test
        func `A new turn is read on its own, not against the one before it`() {
            var partial = PartialContent<Forecast>()
            _ = partial.append(#"{"place": "Athens", "degrees": 17}"#)
            partial.startTurn()

            // The fields belong to the document. A turn that wrote half an object before
            // breaking off to call a tool must not hold the next turn silent.
            #expect(partial.append(#"{"place": "Rome"}"#) != nil)
        }

        @Test
        func `A plain-text turn asks for no schema at all`() async throws {
            _ = try await streamedSnapshots(from: Self.anthropic(), turns: [Anthropic.json], tools: [])
            let body = try #require(Exchange.shared.requestBodies.first)

            #expect(!body.contains("output_config"))
        }

        // MARK: Private

        private static func anthropic(
            onEvent: (@Sendable (GenerationEvent) -> Void)? = nil
        ) -> AnthropicOAuthLanguageModel {
            AnthropicOAuthLanguageModel(tokenProvider: { "token" }, model: "claude-opus-5", onEvent: onEvent)
        }

        private static func codex(
            onEvent: (@Sendable (GenerationEvent) -> Void)? = nil
        ) -> CodexLanguageModel {
            CodexLanguageModel(
                tokenProvider: { CodexToken(accessToken: "token", accountID: "account") },
                model: "gpt-5",
                onEvent: onEvent
            )
        }

        private static func expectFilledInOrder(_ snapshots: [Forecast.PartiallyGenerated]) {
            // The point of streaming a structured answer: the first field is readable while
            // the second is still being written.
            #expect(snapshots.contains { $0.place != nil && $0.degrees == nil })
            #expect(snapshots.last?.place == "Athens")
            #expect(snapshots.last?.degrees == 17)
        }
    }
}
