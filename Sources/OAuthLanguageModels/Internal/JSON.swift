import AnyLanguageModel
import Foundation

/// Shared HTTP-body encoders/decoders for provider clients. Both Codex and
/// Anthropic use snake_case wire formats, so defaulting to the snake-case
/// conversion strategies keeps per-type `CodingKeys` to a minimum.
///
/// `.sortedKeys` is required: provider prompt caches key on the request
/// prefix bytes, and Swift `Dictionary` iteration order is not stable across
/// instances, so without it two otherwise-identical requests in the same
/// session could serialize differently and miss the cache.
extension JSONEncoder {
    static let snakeCase: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }()

    /// Deterministic encoder that preserves key names verbatim (no
    /// snake_case conversion). Used for JSON Schema and tool-call argument
    /// blobs where keys like `additionalProperties` must not be mangled.
    static let deterministic: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }()
}

extension JSONDecoder {
    static let snakeCase: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()
}

extension JSONValue {
    var stringValue: String? {
        if case let .string(string) = self { string } else { nil }
    }

    var intValue: Int? {
        switch self {
        case let .int(int): int
        case let .double(double): Int(double)
        default: nil
        }
    }

    var objectValue: [String: JSONValue]? {
        if case let .object(object) = self { object } else { nil }
    }

    var arrayValue: [JSONValue]? {
        if case let .array(array) = self { array } else { nil }
    }
}

// MARK: - DynamicCodingKey

/// `CodingKey` for containers whose keys aren't known at compile time
/// (e.g. Anthropic's `capabilities.effort` dictionary).
struct DynamicCodingKey: CodingKey {
    // MARK: Lifecycle

    init(stringValue: String) {
        self.stringValue = stringValue
    }

    init?(intValue _: Int) {
        nil
    }

    // MARK: Internal

    let stringValue: String

    var intValue: Int? {
        nil
    }
}
