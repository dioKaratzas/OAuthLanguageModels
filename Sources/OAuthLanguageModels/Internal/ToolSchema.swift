import AnyLanguageModel
import Foundation

/// Encodes any `GenerationSchema`-like value (from either AnyLanguageModel
/// or FoundationModels) into the provider JSON Schema shape. Generic over
/// `Encodable` so both framework variants can share one implementation.
func providerToolSchemaJSONValue(forEncodableSchema schema: some Encodable) throws -> JSONValue {
    let data = try JSONEncoder.deterministic.encode(schema)
    let value = try JSONDecoder().decode(JSONValue.self, from: data)
    return resolveRootSchema(value)
}

private func resolveRootSchema(_ value: JSONValue) -> JSONValue {
    guard case let .object(object) = value else { return value }

    let defs: [String: JSONValue] = if case let .object(defsObject)? = object["$defs"] {
        defsObject
    } else {
        [:]
    }

    let root: JSONValue
    if case let .string(ref)? = object["$ref"],
       ref.hasPrefix("#/$defs/") {
        let name = String(ref.dropFirst("#/$defs/".count))
        root = defs[name] ?? value
    } else {
        root = value
    }

    guard case var .object(rootObject) = root else { return root }

    if rootObject["$defs"] == nil, !defs.isEmpty {
        rootObject["$defs"] = .object(defs)
    }

    if rootObject["additionalProperties"] == nil {
        rootObject["additionalProperties"] = .bool(false)
    }

    if rootObject["required"] == nil,
       case let .object(properties)? = rootObject["properties"],
       !properties.isEmpty {
        rootObject["required"] = .array(properties.keys.sorted().map(JSONValue.string))
    }

    return .object(rootObject)
}
