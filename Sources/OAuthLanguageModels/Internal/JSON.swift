import Foundation

/// Shared HTTP-body encoders/decoders for provider clients. Both Codex and
/// Anthropic use snake_case wire formats, so defaulting to the snake-case
/// conversion strategies keeps per-type `CodingKeys` to a minimum.
extension JSONEncoder {
    static let snakeCase: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
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
