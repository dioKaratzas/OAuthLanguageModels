import Foundation

// Defaults match the Anthropic SDK: 10-minute per-request timeout, up to 2
// retries (3 total attempts), 60s cap on backoff, retrying on transient
// network errors and HTTP 408/409/429/5xx.

let defaultLLMRequestTimeout: TimeInterval = 600 // 10 minutes
let defaultLLMMaxRetries = 2 // i.e. 3 attempts
let defaultLLMInitialRetryDelay: Duration = .milliseconds(500)
let defaultLLMMaxRetryDelay: Duration = .seconds(60)

// MARK: - RetryableServerError

/// Throw this from inside `withNetworkRetry` to force a retry (e.g. after
/// observing a retryable HTTP status code from the server).
struct RetryableServerError: Error {
    let statusCode: Int
    let message: String
}

/// Returns true for errors that should trigger a retry inside `withNetworkRetry`.
func isTransientNetworkError(_ error: any Error) -> Bool {
    if error is RetryableServerError { return true }
    let nsError = error as NSError
    guard nsError.domain == NSURLErrorDomain else { return false }
    switch nsError.code {
    case NSURLErrorTimedOut,
         NSURLErrorCannotConnectToHost,
         NSURLErrorNetworkConnectionLost,
         NSURLErrorNotConnectedToInternet,
         NSURLErrorDNSLookupFailed,
         NSURLErrorResourceUnavailable,
         NSURLErrorCannotFindHost:
        return true
    default:
        return false
    }
}

/// Retryable HTTP status codes, matching the Anthropic SDK's policy.
func isRetryableHTTPStatus(_ status: Int) -> Bool {
    status == 408 || status == 409 || status == 429 || (500..<600).contains(status)
}

/// Runs `operation` with retries on transient errors. The closure should throw
/// on transient failure (network error, or after observing a retryable HTTP
/// status). Backoff is exponential with `initialDelay`, capped at `maxDelay`.
func withNetworkRetry<T>(
    maxRetries: Int = defaultLLMMaxRetries,
    initialDelay: Duration = defaultLLMInitialRetryDelay,
    maxDelay: Duration = defaultLLMMaxRetryDelay,
    operation: () async throws -> T
) async throws -> T {
    var attempt = 0
    var delay = initialDelay
    while true {
        do {
            return try await operation()
        } catch {
            if attempt >= maxRetries || !isTransientNetworkError(error) {
                throw error
            }
            try await Task.sleep(for: delay)
            delay = min(delay * 2, maxDelay)
            attempt += 1
        }
    }
}
