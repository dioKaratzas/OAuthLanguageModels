import AnyLanguageModel
import Foundation

public extension AnthropicOAuthLanguageModel {
    /// Per-call options for `AnthropicOAuthLanguageModel`.
    ///
    /// Set via the standard AnyLanguageModel subscript:
    ///
    /// ```swift
    /// var options = GenerationOptions(temperature: 0.7)
    /// options[custom: AnthropicOAuthLanguageModel.self] = .init(
    ///     thinking: .init(budgetTokens: 4096),
    ///     toolChoice: .auto
    /// )
    /// ```
    ///
    /// > Important: every field set here lands in the request body and
    /// > therefore in the prefix that Anthropic's prompt cache keys on.
    /// > Toggling these between calls in the same `LanguageModelSession`
    /// > will cause a cache miss. Prefer setting them once for the
    /// > duration of a session.
    struct CustomGenerationOptions: AnyLanguageModel.CustomGenerationOptions {
        // MARK: Lifecycle

        public init(
            topP: Double? = nil,
            topK: Int? = nil,
            stopSequences: [String]? = nil,
            toolChoice: ToolChoice? = nil,
            thinking: Thinking? = nil,
            extraBody: [String: JSONValue]? = nil
        ) {
            self.topP = topP
            self.topK = topK
            self.stopSequences = stopSequences
            self.toolChoice = toolChoice
            self.thinking = thinking
            self.extraBody = extraBody
        }

        // MARK: Public

        public enum ToolChoice: Hashable, Sendable {
            case auto
            case any
            case tool(name: String)
            case disabled
        }

        /// Configuration for Claude's [extended thinking](https://docs.claude.com/en/docs/build-with-claude/extended-thinking).
        ///
        /// Enabling thinking forces `temperature` to `1` (Anthropic
        /// requirement) and adds a `thinking` block to the request body.
        public struct Thinking: Hashable, Sendable {
            // MARK: Lifecycle

            public init(budgetTokens: Int) {
                type = .enabled
                self.budgetTokens = budgetTokens
            }

            // MARK: Public

            public enum Mode: String, Sendable {
                case enabled
            }

            public var type: Mode
            public var budgetTokens: Int
        }

        public var topP: Double?
        public var topK: Int?
        public var stopSequences: [String]?
        public var toolChoice: ToolChoice?
        public var thinking: Thinking?
        /// Additional keys to merge into the top-level request body.
        ///
        /// An object under a key the package already writes — `thinking`,
        /// for one — merges into it field by field rather than replacing
        /// it, so one field can be added without restating the rest.
        /// Every other kind of value replaces outright.
        ///
        /// Reserved keys (`model`, `system`, `messages`, `tools`) are
        /// dropped to preserve the integrity of the OAuth/Claude Code
        /// request shape. A dropped key is logged at notice level.
        public var extraBody: [String: JSONValue]?
    }
}
