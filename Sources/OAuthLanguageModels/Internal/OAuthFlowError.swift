import Foundation

/// Failures in getting a login started, before the user has been sent anywhere.
public enum OAuthFlowError: LocalizedError, Sendable {
    /// The authorization URL could not be assembled from its parts, so there is nowhere
    /// to send the user. A wrong URL opened in a browser is worse than none: the provider
    /// answers it with a page about the wrong thing.
    case authorizationURLNotBuildable

    // MARK: Public

    public var errorDescription: String? {
        switch self {
        case .authorizationURLNotBuildable:
            "The authorization URL could not be built."
        }
    }
}
