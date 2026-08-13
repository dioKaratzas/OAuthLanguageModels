import AnyLanguageModel
import Foundation
import PartialJSONDecoder

/// The JSON Schema that makes a provider write `Content` as JSON, or nothing where the
/// content is plain text and no schema applies.
///
/// Both providers deliver a structured answer as ordinary assistant text — Anthropic's
/// `text_delta`, the Responses API's `response.output_text.delta` — so the schema is
/// about what the model writes, never about where it arrives.
func structuredSchema<Content: Generable>(for type: Content.Type) throws -> JSONValue? {
    guard type != String.self else { return nil }
    return try providerToolSchemaJSONValue(forEncodableSchema: Content.generationSchema)
}

/// The schema as an instruction, for the callers who ask for it in the prompt.
///
/// A response format already constrains what the model writes, so this is an addition to
/// it rather than the mechanism: `includeSchemaInPrompt` asks for the schema to be
/// visible to the model in words, and some models follow a schema they can read more
/// closely than one they cannot.
func schemaInstruction(_ schema: JSONValue) throws -> String {
    let data = try JSONEncoder.deterministic.encode(schema)
    guard let json = String(data: data, encoding: .utf8) else {
        throw StructuredOutputError.schemaNotEncodable
    }
    return "Respond with a single JSON value matching this schema, and nothing else:\n\(json)"
}

// MARK: - StructuredOutputError

/// Failures in getting a structured answer asked for, or read back.
public enum StructuredOutputError: LocalizedError, Sendable {
    /// The schema for the requested type could not be written as JSON, so the request
    /// would have gone out asking for nothing in particular.
    case schemaNotEncodable
    /// The answer arrived, and is not what was asked for.
    case answerNotDecodable(String)

    // MARK: Public

    public var errorDescription: String? {
        switch self {
        case .schemaNotEncodable:
            "The schema for the requested type could not be encoded as JSON."
        case let .answerNotDecodable(text):
            "The model answered with something other than the requested type: \(text)"
        }
    }
}

/// The finished answer, decoded.
///
/// Plain text is itself the answer; anything else is the JSON the model wrote, which has
/// to parse — a provider that ignored the response format and answered in prose is a
/// failure worth reporting rather than an empty result to puzzle over.
func finishedContent<Content: Generable>(
    _ text: String,
    as type: Content.Type
) throws -> (content: Content, rawContent: GeneratedContent) {
    if type == String.self {
        // Plain text is the answer verbatim, not a JSON document to be read out of it.
        let raw = GeneratedContent(text)
        return (try Content(raw), raw)
    }
    guard let document = jsonDocument(in: text) else {
        throw StructuredOutputError.answerNotDecodable(text)
    }
    let raw = try GeneratedContent(json: document)
    return (try Content(raw), raw)
}

/// The JSON a model wrote, out of whatever it wrapped it in.
///
/// With a response format in the request there is no wrapping and this is the whole
/// string. Without one — a model asked in the prompt alone — a fenced block or a
/// sentence in front of the object is what actually turns up.
func jsonDocument(in text: String) -> String? {
    var trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.hasPrefix("```") {
        trimmed = trimmed.drop(while: { !$0.isNewline }).trimmingCharacters(in: .whitespacesAndNewlines)
        if let fence = trimmed.range(of: "```", options: .backwards) {
            trimmed = String(trimmed[trimmed.startIndex..<fence.lowerBound])
        }
    }
    if let start = trimmed.firstIndex(where: { $0 == "{" || $0 == "[" }) {
        return String(trimmed[start...])
    }
    return trimmed.isEmpty ? nil : trimmed
}

// MARK: - StreamedAnswer

/// The answer as it is streamed, whichever kind it is.
///
/// Plain text grows across the whole exchange, so a caller appending the tail of each
/// snapshot gets one continuous answer with any tool rounds invisible in the middle of
/// it. A structured answer starts over at each turn instead: the JSON is whatever the
/// turn that finishes writes, and text from a turn that broke off to call a tool is not
/// the first half of it.
struct StreamedAnswer<Content: Generable> where Content.PartiallyGenerated: Sendable {
    // MARK: Lifecycle

    init(isStructured: Bool) {
        self.isStructured = isStructured
    }

    // MARK: Internal

    mutating func startTurn() {
        guard isStructured else { return }
        partial.startTurn()
    }

    mutating func append(_ delta: String) throws -> LanguageModelSession.ResponseStream<Content>.Snapshot? {
        guard isStructured else {
            text += delta
            let raw = GeneratedContent(text)
            return .init(content: try Content.PartiallyGenerated(raw), rawContent: raw)
        }
        guard let snapshot = partial.append(delta) else { return nil }
        didYield = true
        return snapshot
    }

    /// Reports a structured answer that never parsed.
    ///
    /// A provider that ignored the response format and replied in prose leaves the caller
    /// with an empty stream and nothing to explain it; the decode error names what came
    /// back instead. A turn that wrote nothing at all — stopped before it began — is
    /// silence rather than a failure.
    func checkFinished() throws {
        guard isStructured, !didYield, !partial.text.isEmpty else { return }
        throw StructuredOutputError.answerNotDecodable(partial.text)
    }

    // MARK: Private

    private let isStructured: Bool
    private var text = ""
    private var partial = PartialContent<Content>()
    private var didYield = false
}

// MARK: - PartialContent

/// Turns the JSON a model is halfway through writing into snapshots of `Content`.
///
/// A growing JSON document is decodable at every step: what has arrived is completed to a
/// well-formed value and the fields that have not arrived stay nil, which is what
/// `PartiallyGenerated` is for. Input that does not parse yet is the ordinary case, not a
/// failure — it means the next delta has not landed.
struct PartialContent<Content: Generable> where Content.PartiallyGenerated: Sendable {
    // MARK: Internal

    /// The document so far, for the final decode.
    private(set) var text = ""

    /// Adds what just arrived and hands back a snapshot, or nothing where the document has
    /// not yet reached a point that decodes.
    mutating func append(_ delta: String) -> LanguageModelSession.ResponseStream<Content>.Snapshot? {
        text += delta
        return snapshot()
    }

    /// Starts the document over for a new turn.
    ///
    /// The answer is whatever the turn that finishes writes; text from a turn that broke
    /// off to call a tool is not the first half of it, and concatenating the two would
    /// make a document that never parses.
    mutating func startTurn() {
        text = ""
        // The fields belong to the document, not to the exchange: a turn that wrote half
        // an object before breaking off to call a tool would otherwise hold the next turn
        // silent until it had written past the same fields again.
        arrived = []
    }

    // MARK: Private

    private let completer = JSONCompleter()

    /// The fields the last accepted snapshot carried. A snapshot that has lost one is
    /// dropped rather than yielded: a caller reading fields off successive snapshots must
    /// never see one it has already been given go back to nil.
    private var arrived: Set<String> = []

    private mutating func snapshot() -> LanguageModelSession.ResponseStream<Content>.Snapshot? {
        guard let document = jsonDocument(in: text) else { return nil }
        guard let completed = try? completer.complete(document), !completed.isEmpty else { return nil }
        guard let raw = try? GeneratedContent(json: completed) else { return nil }
        let content: Content.PartiallyGenerated
        do {
            content = try Content.PartiallyGenerated(raw)
        } catch {
            return nil
        }

        let fields = Self.fields(of: raw)
        guard arrived.isSubset(of: fields) else { return nil }
        arrived = fields
        return .init(content: content, rawContent: raw)
    }

    /// The fields that have actually arrived.
    ///
    /// A key whose value has not been written yet completes to null, and the field name
    /// itself may still be half-typed — `"degre` becomes `"degre": null` one delta before
    /// it becomes `"degrees": 17`. Counting only the fields that carry a value keeps a
    /// name that was never finished from looking like a field that has been lost.
    private static func fields(of content: GeneratedContent) -> Set<String> {
        guard case let .structure(properties, _) = content.kind else { return [] }
        return Set(properties.filter { $0.value.kind != .null }.keys)
    }
}
