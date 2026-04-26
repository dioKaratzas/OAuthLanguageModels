import AnyLanguageModel
import Foundation

public extension CodexLanguageModel {
    /// Per-call options for `CodexLanguageModel`.
    ///
    /// Set via the standard AnyLanguageModel subscript:
    ///
    /// ```swift
    /// var options = GenerationOptions(temperature: 0.7)
    /// options[custom: CodexLanguageModel.self] = .init(
    ///     reasoning: .init(effort: .high, summary: .auto),
    ///     verbosity: .low
    /// )
    /// ```
    ///
    /// > Important: every field set here lands in the request body and
    /// > therefore in the prefix that OpenAI's prompt cache keys on
    /// > (via `prompt_cache_key`). Toggling these between calls in the
    /// > same `LanguageModelSession` will cause a cache miss. Prefer
    /// > setting them once for the duration of a session.
    struct CustomGenerationOptions: AnyLanguageModel.CustomGenerationOptions {
        // MARK: Lifecycle

        public init(
            topP: Double? = nil,
            parallelToolCalls: Bool? = nil,
            maxToolCalls: Int? = nil,
            reasoning: ReasoningConfiguration? = nil,
            verbosity: Verbosity? = nil,
            maxOutputTokens: Int? = nil,
            toolChoice: ToolChoice? = nil,
            extraBody: [String: JSONValue]? = nil
        ) {
            self.topP = topP
            self.parallelToolCalls = parallelToolCalls
            self.maxToolCalls = maxToolCalls
            self.reasoning = reasoning
            self.verbosity = verbosity
            self.maxOutputTokens = maxOutputTokens
            self.toolChoice = toolChoice
            self.extraBody = extraBody
        }

        // MARK: Public

        public enum ReasoningEffort: String, Hashable, Sendable {
            case minimal, low, medium, high
        }

        /// Configuration for the [reasoning](https://platform.openai.com/docs/guides/reasoning)
        /// behavior of o-series and gpt-5 models.
        public struct ReasoningConfiguration: Hashable, Sendable {
            // MARK: Lifecycle

            public init(effort: ReasoningEffort? = nil, summary: Summary? = nil) {
                self.effort = effort
                self.summary = summary
            }

            // MARK: Public

            public enum Summary: String, Hashable, Sendable {
                case auto, concise, detailed
            }

            public var effort: ReasoningEffort?
            public var summary: Summary?
        }

        public enum Verbosity: String, Hashable, Sendable {
            case low, medium, high
        }

        public enum ToolChoice: Hashable, Sendable {
            case none
            case auto
            case required
            case function(name: String)
            case allowedTools(names: [String], mode: AllowedMode)

            // MARK: Public

            public enum AllowedMode: String, Hashable, Sendable {
                case auto, required
            }
        }

        public var topP: Double?
        public var parallelToolCalls: Bool?
        public var maxToolCalls: Int?
        public var reasoning: ReasoningConfiguration?
        public var verbosity: Verbosity?
        public var maxOutputTokens: Int?
        public var toolChoice: ToolChoice?
        /// Additional keys to merge into the top-level request body.
        ///
        /// Reserved keys (`model`, `input`, `instructions`, `tools`,
        /// `prompt_cache_key`, `store`, `stream`, `include`) are dropped
        /// to preserve correctness of the Codex/OAuth request shape and
        /// the cache contract.
        public var extraBody: [String: JSONValue]?
    }
}
