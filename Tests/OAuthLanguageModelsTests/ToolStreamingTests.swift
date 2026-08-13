import AnyLanguageModel
import Foundation
import Testing
@testable import OAuthLanguageModels

/// Both providers driven through a whole tool round against a stubbed endpoint.
///
/// Serialized, and one suite rather than two: the stub is reached through
/// `URLSession.shared`, so only one exchange can be in flight at a time.
extension StreamedExchange {
    @Suite("with tools")
    struct ToolStreamingTests {
        // MARK: Internal

        enum Anthropic {
            /// The model says something, then calls a tool.
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
        }

        enum Codex {
            static let askingForTheTool = """
            data: {"type":"response.created","response":{"id":"resp_01","status":"in_progress","output":[]}}

            data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_01","delta":"They want Athens."}

            data: {"type":"response.output_text.delta","item_id":"msg_01","delta":"Let me check. "}

            data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fc_01","type":"function_call","status":"in_progress","arguments":"","call_id":"call_01","name":"get_weather"}}

            data: {"type":"response.function_call_arguments.delta","item_id":"fc_01","output_index":1,"delta":"{\\"location\\":"}

            data: {"type":"response.function_call_arguments.delta","item_id":"fc_01","output_index":1,"delta":" \\"Athens\\"}"}

            data: {"type":"response.output_item.done","output_index":1,"item":{"id":"fc_01","type":"function_call","status":"completed","arguments":"{\\"location\\": \\"Athens\\"}","call_id":"call_01","name":"get_weather"}}

            data: {"type":"response.completed","response":{"id":"resp_01","status":"completed","output":[{"id":"rs_01","type":"reasoning","encrypted_content":"gAAAAAB"},{"id":"fc_01","type":"function_call","arguments":"{\\"location\\": \\"Athens\\"}","call_id":"call_01","name":"get_weather"}],"usage":{"input_tokens":40,"input_tokens_details":{"cached_tokens":0},"output_tokens":40}}}

            """

            static let answeringWithIt = """
            data: {"type":"response.output_text.delta","item_id":"msg_02","delta":"It is 17C in Athens."}

            data: {"type":"response.completed","response":{"id":"resp_02","status":"completed","output":[{"id":"msg_02","type":"message","role":"assistant","content":[{"type":"output_text","text":"It is 17C in Athens."}]}],"usage":{"input_tokens":90,"input_tokens_details":{"cached_tokens":0},"output_tokens":12}}}

            """
        }

        @Test
        func `Anthropic runs its tools mid-stream and keeps writing afterwards`() async throws {
            let calls = Counter()
            let snapshots = try await streamedSnapshots(
                from: Self.anthropic(),
                turns: [Anthropic.askingForTheTool, Anthropic.answeringWithIt],
                tools: [WeatherTool(callCount: calls)]
            )

            #expect(calls.count == 1)
            Self.expectOneContinuousAnswer(snapshots)
        }

        @Test
        func `Anthropic sends the tools, then sends back the call and its result`() async throws {
            _ = try await streamedSnapshots(
                from: Self.anthropic(),
                turns: [Anthropic.askingForTheTool, Anthropic.answeringWithIt],
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
        func `Anthropic reports both turns, each with its own usage`() async throws {
            let reports = Reports()
            _ = try await streamedSnapshots(
                from: Self.anthropic(onEvent: reports.handler),
                turns: [Anthropic.askingForTheTool, Anthropic.answeringWithIt],
                tools: [WeatherTool(callCount: Counter())]
            )

            #expect(reports.values.map(\.stopReason) == [.toolUse, .endTurn])
            #expect(reports.values.map(\.usage.inputTokens) == [40, 90])
        }

        @Test
        func `Codex runs its tools mid-stream and keeps writing afterwards`() async throws {
            let calls = Counter()
            let snapshots = try await streamedSnapshots(
                from: Self.codex(),
                turns: [Codex.askingForTheTool, Codex.answeringWithIt],
                tools: [WeatherTool(callCount: calls)]
            )

            #expect(calls.count == 1)
            Self.expectOneContinuousAnswer(snapshots)
        }

        @Test
        func `Codex sends the tools, then sends back the call, its result and the reasoning`() async throws {
            _ = try await streamedSnapshots(
                from: Self.codex(),
                turns: [Codex.askingForTheTool, Codex.answeringWithIt],
                tools: [WeatherTool(callCount: Counter())]
            )
            let bodies = Exchange.shared.requestBodies

            #expect(bodies.count == 2)
            #expect(bodies.first?.contains(#""name":"get_weather""#) == true)
            #expect(bodies.last?.contains(#""type":"function_call_output""#) == true)
            #expect(bodies.last?.contains(#""call_id":"call_01""#) == true)
            #expect(bodies.last?.contains("17C and clear.") == true)
            // `store: false` means the provider kept none of the first turn's reasoning, so
            // the second request has to carry it back.
            #expect(bodies.last?.contains("gAAAAAB") == true)
        }

        @Test
        func `Codex reports both turns, each with its own usage`() async throws {
            let reports = Reports()
            _ = try await streamedSnapshots(
                from: Self.codex(onEvent: reports.handler),
                turns: [Codex.askingForTheTool, Codex.answeringWithIt],
                tools: [WeatherTool(callCount: Counter())]
            )

            #expect(reports.values.map(\.stopReason) == [.toolUse, .endTurn])
            #expect(reports.values.map(\.usage.inputTokens) == [40, 90])
        }

        @Test
        func `Reasoning is reported without reaching the answer`() async throws {
            let deltas = Deltas()
            let snapshots = try await streamedSnapshots(
                from: Self.codex(onEvent: deltas.handler),
                turns: [Codex.askingForTheTool, Codex.answeringWithIt],
                tools: [WeatherTool(callCount: Counter())]
            )

            #expect(deltas.values == ["They want Athens."])
            #expect(snapshots.last?.contains("They want Athens.") == false)
        }

        @Test
        func `A model that only ever calls tools is cut off rather than run forever`() async throws {
            let calls = Counter()

            await #expect(throws: AnthropicOAuthLanguageModelError.self) {
                try await streamedSnapshots(
                    from: Self.anthropic(maxToolRounds: 3),
                    turns: [Anthropic.askingForTheTool],
                    tools: [WeatherTool(callCount: calls)],
                    repeatingLastTurn: true
                )
            }
            // Three rounds fed back, and the fourth request never made.
            #expect(calls.count == 3)
        }

        @Test
        func `A Codex model that only ever calls tools is cut off too`() async throws {
            let calls = Counter()

            await #expect(throws: CodexLanguageModelError.self) {
                try await streamedSnapshots(
                    from: Self.codex(maxToolRounds: 3),
                    turns: [Codex.askingForTheTool],
                    tools: [WeatherTool(callCount: calls)],
                    repeatingLastTurn: true
                )
            }
            #expect(calls.count == 3)
        }

        @Test
        func `A stopped tool call leaves the answer as far as it got`() async throws {
            let calls = Counter()
            let snapshots = try await streamedSnapshots(
                from: Self.anthropic(),
                turns: [Anthropic.askingForTheTool, Anthropic.answeringWithIt],
                tools: [WeatherTool(callCount: calls)],
                delegate: StoppingDelegate()
            )

            // A snapshot the caller has already been shown cannot be withdrawn, so the text
            // written before the call stands — but the tool never runs and no second turn is
            // opened for it.
            #expect(calls.count == 0)
            #expect(snapshots.last == "Let me check. ")
            #expect(Exchange.shared.requestBodies.count == 1)
        }

        // MARK: Private

        private struct StoppingDelegate: ToolExecutionDelegate {
            func toolCallDecision(
                for _: Transcript.ToolCall,
                in _: LanguageModelSession
            ) async -> ToolExecutionDecision {
                .stop
            }
        }

        private final class Deltas: @unchecked Sendable {
            var values: [String] {
                lock.lock(); defer { lock.unlock() }
                return storage
            }

            var handler: @Sendable (GenerationEvent) -> Void {
                { event in
                    guard case let .reasoning(delta) = event else { return }
                    self.lock.lock(); defer { self.lock.unlock() }
                    self.storage.append(delta)
                }
            }

            private let lock = NSLock()
            private var storage: [String] = []
        }

        private static func anthropic(
            maxToolRounds: Int = defaultMaxToolRounds,
            onEvent: (@Sendable (GenerationEvent) -> Void)? = nil
        ) -> AnthropicOAuthLanguageModel {
            AnthropicOAuthLanguageModel(
                tokenProvider: { "token" },
                model: "claude-opus-5",
                maxToolRounds: maxToolRounds,
                onEvent: onEvent
            )
        }

        private static func codex(
            maxToolRounds: Int = defaultMaxToolRounds,
            onEvent: (@Sendable (GenerationEvent) -> Void)? = nil
        ) -> CodexLanguageModel {
            CodexLanguageModel(
                tokenProvider: { CodexToken(accessToken: "token", accountID: "account") },
                model: "gpt-5",
                maxToolRounds: maxToolRounds,
                onEvent: onEvent
            )
        }

        private static func expectOneContinuousAnswer(_ snapshots: [String]) {
            // Text from both sides of the tool call, in one continuous answer: the snapshots
            // stay cumulative across the round trip rather than restarting after it.
            #expect(snapshots.last == "Let me check. It is 17C in Athens.")
            #expect(snapshots.contains("Let me check. "))
            // The fragments the arguments arrived in are not text and must never be shown as
            // such, so every snapshot is a prefix of the finished answer.
            #expect(snapshots.allSatisfy { snapshots.last?.hasPrefix($0) == true })
        }
    }
}
