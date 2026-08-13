import AnyLanguageModel
import Foundation

/// What a streamed tool-using turn actually put on the wire, kept per session.
///
/// A streamed exchange sends the assistant's tool calls and the user's tool results as
/// their own messages, and the provider caches that exact sequence. The session's
/// transcript records none of it — a `ResponseStream` has nowhere to put transcript
/// entries — so the next turn would rebuild that exchange as one plain assistant message,
/// the wire prefix would diverge at the first tool-using turn, and everything behind it
/// would be re-read at full price.
///
/// Keeping the messages here and putting them back on the next turn is what makes the
/// second request byte-identical to the first as far as it went. The root fix is a
/// `ResponseStream` that can carry transcript entries, which is upstream; this needs no
/// upstream change.
///
/// Keyed by session identity and held weakly: a session that goes away takes its record
/// with it, and nothing here keeps one alive.
final class StreamedTurnStore<Message>: @unchecked Sendable {
    // MARK: Internal

    /// Records the messages a streamed turn added beyond what the transcript will
    /// remember, against the response entry that turn is about to become.
    func record(_ messages: [Message], for session: LanguageModelSession, at responseIndex: Int) {
        guard !messages.isEmpty else { return }
        lock.withLock {
            reapLocked()
            let id = ObjectIdentifier(session)
            var bucket = buckets[id] ?? Bucket(reference: WeakSession(session), turns: [:])
            bucket.turns[responseIndex] = messages
            buckets[id] = bucket
        }
    }

    /// What the response at `responseIndex` sent on the wire, where it was streamed with
    /// tools in play.
    func messages(for session: LanguageModelSession, at responseIndex: Int) -> [Message]? {
        lock.withLock {
            reapLocked()
            return buckets[ObjectIdentifier(session)]?.turns[responseIndex]
        }
    }

    // MARK: Private

    private final class WeakSession: @unchecked Sendable {
        // MARK: Lifecycle

        init(_ session: LanguageModelSession) {
            self.session = session
        }

        // MARK: Internal

        weak var session: LanguageModelSession?
    }

    private struct Bucket {
        let reference: WeakSession
        var turns: [Int: [Message]]
    }

    private let lock = NSLock()
    private var buckets: [ObjectIdentifier: Bucket] = [:]

    private func reapLocked() {
        buckets = buckets.filter { $0.value.reference.session != nil }
    }
}

extension Transcript {
    /// How many responses the transcript already holds, which is the index the response
    /// being generated now will take.
    var responseCount: Int {
        reduce(into: 0) { count, entry in
            if case .response = entry { count += 1 }
        }
    }
}
