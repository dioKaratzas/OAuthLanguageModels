import Foundation
#if canImport(os)
import os
#endif

#if canImport(os)
private let logger = Logger(subsystem: "OAuthLanguageModels", category: "request")
#endif

/// Where ``logDropped`` reports to besides the log, so that a test can assert the
/// reporting happens at all — `os.Logger` output cannot be read back in process.
///
/// Set from one test at a time and cleared afterwards.
nonisolated(unsafe) var droppedKeyObserver: (@Sendable (_ kind: String, _ name: String, _ source: String) -> Void)?

/// Reports a caller-supplied key that was thrown away rather than sent.
///
/// The dropping is deliberate — these are the fields a subscription token is honoured
/// for — but silently ignoring a caller's input sends them looking for the fault in
/// their own code.
func logDropped(_ kind: String, name: String, from source: String) {
    droppedKeyObserver?(kind, name, source)
    #if canImport(os)
    logger.notice("Dropped reserved \(kind, privacy: .public) \"\(name, privacy: .public)\" from \(source, privacy: .public): it is part of the OAuth request shape and cannot be overridden.")
    #endif
}
