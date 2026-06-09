# OAuthLanguageModels

Language-model backends that talk to **Anthropic Claude** and **OpenAI ChatGPT / Codex** using a user's existing Claude or ChatGPT subscription, instead of a paid API key.

`AnthropicOAuthLanguageModel` and `CodexLanguageModel` each conform to **both**:

- [`AnyLanguageModel.LanguageModel`](https://github.com/huggingface/AnyLanguageModel) — works on macOS 14 / iOS 17 / visionOS 1 and up.
- `FoundationModels.LanguageModel` (Apple's framework) — additionally available on iOS 27 / macOS 27 / visionOS 27+.

Use whichever framework's `LanguageModelSession` you like; the same model value works with both.

Two pieces, used independently:

1. **Language models** — `AnthropicOAuthLanguageModel` and `CodexLanguageModel`. Drop them into any `LanguageModelSession`. They take a `tokenProvider` closure; you decide where the token comes from.
2. **OAuth flow (optional)** — `AnthropicOAuthFlow` / `CodexOAuthFlow` plus refresh helpers. Use these if you want this package to obtain and refresh tokens for you. Skip them entirely if you already have a token from somewhere else (your own backend, a Claude Code / Codex CLI install on disk, a test fixture, …).

## Installation

```swift
.package(url: "https://github.com/finnvoor/OAuthLanguageModels", from: "0.1.0")
```

```swift
.target(
    name: "MyApp",
    dependencies: [
        .product(name: "OAuthLanguageModels", package: "OAuthLanguageModels"),
    ]
)
```

Requires Swift 6.1+. Matches AnyLanguageModel's platforms: macOS 14+, Mac Catalyst 17+, iOS 17+, tvOS 17+, watchOS 10+, visionOS 1+. The `FoundationModels` conformance is gated to the 27-series OSes (iOS 27 / macOS 27 / visionOS 27 / watchOS 27, tvOS unavailable); on older OS versions use the `AnyLanguageModel` conformance.

## Using the language models

Both models work with structured generation (`Generable`), tool calls, and streaming through either framework's `LanguageModelSession`.

### With AnyLanguageModel (any supported OS)

```swift
import AnyLanguageModel
import OAuthLanguageModels

let model = AnthropicOAuthLanguageModel(
    tokenProvider: { "sk-ant-oat01-…" },
    model: "claude-sonnet-4-5"
)
let session = LanguageModelSession(model: model)   // AnyLanguageModel.LanguageModelSession
let response = try await session.respond(to: "Write a haiku about Swift.")
print(response.content)
```

Add the `AnyLanguageModel` product to your target when you use this path:

```swift
.product(name: "AnyLanguageModel", package: "AnyLanguageModel"),
```

### With Apple's FoundationModels (iOS/macOS/visionOS 27+)

The examples below import `FoundationModels`. The same model values are used; just pick the import that matches the `LanguageModelSession` you want.

### Anthropic

```swift
import FoundationModels
import OAuthLanguageModels

let model = AnthropicOAuthLanguageModel(
    tokenProvider: { "sk-ant-oat01-…" },   // your access token
    model: "claude-sonnet-4-5"
)

let session = LanguageModelSession(model: model)
let response = try await session.respond(to: "Write a haiku about Swift.")
print(response.content)
```

> The Claude OAuth scope only authorizes requests that look like Claude Code. `AnthropicOAuthLanguageModel` automatically prepends the required Claude Code system preamble and sets the `user-agent` / `x-app` headers for you.

### Codex / ChatGPT

```swift
import FoundationModels
import OAuthLanguageModels

let model = CodexLanguageModel(
    tokenProvider: {
        CodexToken(accessToken: "eyJhbGciOi…", accountID: "acc_…")
    },
    model: "gpt-5"
)

let session = LanguageModelSession(model: model)
for try await snapshot in session.streamResponse(to: "Stream me some prose.") {
    print(snapshot.content, terminator: "")
}
```

`CodexLanguageModel` talks to ChatGPT's `backend-api/codex/responses` SSE endpoint, so the access token must come from a ChatGPT account (free / Plus / Pro / Team), not an OpenAI Platform API key. The `accountID` is the ChatGPT account UUID — `CodexAuth.toToken()` will extract it for you from the JWT if you don't have it handy.

The `tokenProvider` closure is called for **every request**, so the recommended pattern is to wire it through your refresh logic (see below) rather than capturing a token by value.

## Optional: the OAuth login flow

If you want this library to obtain tokens, use `AnthropicOAuthFlow` / `CodexOAuthFlow`. Both follow the same shape:

1. Call `login()` to start a loopback callback server and get a `PendingLogin`.
2. Present `pending.authorizationURL` to the user however you like — `NSWorkspace.shared.open(_:)`, `UIApplication.shared.open(_:)`, a SwiftUI `Link`, a `Button`, a copy-to-clipboard prompt, a QR code, …
3. `await pending.waitForCallback()` to complete the exchange and receive a credential bundle.

```swift
let pending = try await AnthropicOAuthFlow.login()

// Hand the URL to the user however you want.
NSWorkspace.shared.open(pending.authorizationURL)

let auth: AnthropicAuth = try await pending.waitForCallback()
try persist(auth)   // your storage; Keychain, a JSON file, etc.
```

```swift
let pending = try await CodexOAuthFlow.login()
NSWorkspace.shared.open(pending.authorizationURL)

let auth: CodexAuth = try await pending.waitForCallback()
try persist(auth)
```

Cancel a pending flow if the user backs out:

```swift
await pending.cancel()
```

The callback server listens on a fixed loopback port (`53692` for Anthropic, `1455` for Codex — matching the upstream Claude Code and Codex CLI redirect URIs). Times out after 5 minutes.

### Refresh + persistence

This package is **stateless about storage**. You provide `load` / `save` closures and `AuthService` handles freshness checks and refresh:

```swift
let token = try await AnthropicAuthService.validAccessToken(
    load: { try Keychain.loadAnthropicAuth() },
    save: { try Keychain.saveAnthropicAuth($0) }
)

let model = AnthropicOAuthLanguageModel(
    tokenProvider: {
        try await AnthropicAuthService.validAccessToken(
            load: { try Keychain.loadAnthropicAuth() },
            save: { try Keychain.saveAnthropicAuth($0) }
        )
    },
    model: "claude-sonnet-4-5"
)
```

```swift
let model = CodexLanguageModel(
    tokenProvider: {
        let auth = try await CodexAuthService.validAuth(
            load: { try Keychain.loadCodexAuth() },
            save: { try Keychain.saveCodexAuth($0) }
        )
        return try auth.toToken()
    },
    model: "gpt-5"
)
```

`validAccessToken` / `validAuth` returns the cached token immediately if it's still fresh, otherwise refreshes (and `save`s the rotated credentials) before returning.

## Skipping the login flow

You don't need `*OAuthFlow` at all if you already have credentials. A few common cases:

- **Reusing Claude Code's stored token** — read `~/.claude/.credentials.json`, decode it as `AnthropicAuth`, and feed it through `AnthropicAuthService`.
- **Reusing Codex CLI's stored token** — read `~/.codex/auth.json`, decode it as `CodexAuth`, and feed it through `CodexAuthService`.
- **Server-issued tokens** — your backend hands you a token; just return it from `tokenProvider`.
- **Tests** — return a stub token synchronously.
