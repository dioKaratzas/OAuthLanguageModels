import Foundation

/// How many tool rounds an exchange gets before the package stops feeding it.
///
/// Well above what an ordinary question needs, and far below the point where a model
/// stuck in a loop has spent real money.
public let defaultMaxToolRounds = 8

// MARK: - TokenUsage

/// What a turn cost, normalized across providers.
///
/// `inputTokens` counts only the prompt the provider actually read at full price;
/// anything served from the cache is in ``cacheReadTokens`` instead. Anthropic reports
/// the two separately already, and the Responses API's `input_tokens` is reduced by its
/// `cached_tokens` to match, so `inputTokens + cacheReadTokens` is the whole prompt on
/// either provider.
public struct TokenUsage: Hashable, Sendable {
    // MARK: Lifecycle

    public init(
        inputTokens: Int = 0,
        outputTokens: Int = 0,
        cacheWriteTokens: Int = 0,
        cacheReadTokens: Int = 0
    ) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.cacheWriteTokens = cacheWriteTokens
        self.cacheReadTokens = cacheReadTokens
    }

    // MARK: Public

    /// Prompt tokens read at full price.
    public var inputTokens: Int

    /// Tokens generated, including any spent on extended thinking.
    public var outputTokens: Int

    /// Prompt tokens written into the cache, billed at a premium once. Always zero on
    /// Codex, which does not report cache writes separately.
    public var cacheWriteTokens: Int

    /// Prompt tokens served from the cache at a fraction of the input price. A turn that
    /// re-reads a long brief in full leaves this at zero.
    public var cacheReadTokens: Int

    /// Every prompt token the turn was charged for, cached or not.
    public var totalInputTokens: Int {
        inputTokens + cacheWriteTokens + cacheReadTokens
    }
}

// MARK: - StopReason

/// Why the provider stopped generating.
public enum StopReason: Hashable, Sendable {
    /// The model finished what it had to say.
    case endTurn
    /// The token ceiling was reached first, so the answer is cut off mid-thought.
    case maxTokens
    /// One of the caller's stop sequences was produced.
    case stopSequence
    /// The model is waiting on tool results before it continues.
    case toolUse
    /// The model declined to answer.
    case refusal
    /// A reason this package has no case for, in the provider's own wording.
    case other(String)

    // MARK: Lifecycle

    /// Maps Anthropic's `stop_reason` onto the shared vocabulary.
    init(anthropic raw: String) {
        switch raw {
        case "end_turn": self = .endTurn
        case "max_tokens": self = .maxTokens
        case "stop_sequence": self = .stopSequence
        case "tool_use": self = .toolUse
        case "refusal": self = .refusal
        default: self = .other(raw)
        }
    }
}

// MARK: - TurnReport

/// What a single request to the provider cost and how it ended.
///
/// A turn is one request/response pair, so a tool-using exchange reports once per round
/// trip rather than once for the whole exchange.
public struct TurnReport: Hashable, Sendable {
    // MARK: Lifecycle

    public init(usage: TokenUsage = .init(), stopReason: StopReason? = nil) {
        self.usage = usage
        self.stopReason = stopReason
    }

    // MARK: Public

    public var usage: TokenUsage

    /// Nil when the stream ended without the provider saying why, which is itself worth
    /// knowing: the connection dropped rather than the answer finishing.
    public var stopReason: StopReason?
}

// MARK: - GenerationEvent

/// Everything a turn produces that is not the answer itself — the reasoning behind it,
/// the tools it asked for, and what it cost.
///
/// Neither `AnyLanguageModel.LanguageModelSession.Response` nor its streaming `Snapshot`
/// has a slot to carry this, and the same model also serves FoundationModels, whose
/// channel has no slot either. A callback on the model is therefore the one channel both
/// adapters can feed from both the streaming and the non-streaming path.
public enum GenerationEvent: Hashable, Sendable {
    /// A piece of the model's reasoning, as it is written. Kept apart from the answer
    /// text on purpose — concatenating the two puts the model's scratch work in front of
    /// the reader as if it were the reply.
    case reasoning(String)

    /// A tool the model asked for, once its arguments are whole.
    ///
    /// Reported before the tool runs and once per call, so a caller can say what is being
    /// done while it is being done. The arguments are the JSON object the model wrote;
    /// they are never handed over half-written, since a fragment of one parses as
    /// nothing.
    case toolCall(name: String, arguments: String)

    /// The turn is over. Arrives once per request, after the last content.
    case turnFinished(TurnReport)
}
