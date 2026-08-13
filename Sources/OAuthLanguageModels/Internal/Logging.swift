import Foundation
#if canImport(os)
import os
#endif

#if canImport(os)
private let logger = Logger(subsystem: "OAuthLanguageModels", category: "request")
#endif

/// Reports a caller-supplied key that was thrown away rather than sent.
///
/// The dropping is deliberate — these are the fields a subscription token is honoured
/// for — but silently ignoring a caller's input sends them looking for the fault in
/// their own code.
func logDropped(_ kind: String, name: String, from source: String) {
    #if canImport(os)
    logger.notice("Dropped reserved \(kind, privacy: .public) \"\(name, privacy: .public)\" from \(source, privacy: .public): it is part of the OAuth request shape and cannot be overridden.")
    #endif
}
