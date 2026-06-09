import FoundationModels
import OAuthLanguageModels
import SwiftUI
import UIKit

struct ContentView: View {
    // MARK: Internal

    enum Provider: String, CaseIterable, Identifiable {
        case anthropic = "Claude"
        case codex = "Codex"

        // MARK: Internal

        var id: String {
            rawValue
        }
    }

    var body: some View {
        NavigationStack {
            Form {
                Picker("Provider", selection: $provider) {
                    ForEach(Provider.allCases) { Text($0.rawValue).tag($0) }
                }
                .onChange(of: provider) { _, provider in
                    model = provider == .anthropic ? "claude-sonnet-4-5" : "gpt-5"
                }

                Section("Credentials") {
                    SecureField(provider == .anthropic ? "Claude OAuth access token" : "ChatGPT access token", text: $token)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    if provider == .codex {
                        TextField("ChatGPT account ID", text: $accountID)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                    }
                    Button("Sign in with \(provider.rawValue)") {
                        Task { await signIn() }
                    }
                    .disabled(isRunning)
                    TextField("Model", text: $model)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }

                Section("Prompt") {
                    TextEditor(text: $prompt)
                        .frame(minHeight: 90)
                    Button(isRunning ? "Running…" : "Send") {
                        Task { await run() }
                    }
                    .disabled(isRunning || token.isEmpty || (provider == .codex && accountID.isEmpty))
                }

                Section("Output") {
                    Text(output.isEmpty ? "No response yet" : output)
                        .textSelection(.enabled)
                }
            }
            .navigationTitle("OAuth LM Demo")
        }
    }

    // MARK: Private

    @State private var provider: Provider = .anthropic
    @State private var token = ""
    @State private var accountID = ""
    @State private var model = "claude-sonnet-4-5"
    @State private var prompt = "Reply with one sentence confirming this works."
    @State private var output = ""
    @State private var isRunning = false

    @MainActor private func signIn() async {
        isRunning = true
        output = "Starting \(provider.rawValue) sign-in…"
        defer { isRunning = false }

        do {
            switch provider {
            case .anthropic:
                let pending = try await AnthropicOAuthFlow.login()
                output = "Opening Claude sign-in…"
                await UIApplication.shared.open(pending.authorizationURL)
                let auth = try await pending.waitForCallback()
                token = auth.accessToken
                output = "Claude sign-in complete. Access token filled in."
            case .codex:
                let pending = try await CodexOAuthFlow.login()
                output = "Opening ChatGPT sign-in…"
                await UIApplication.shared.open(pending.authorizationURL)
                let auth = try await pending.waitForCallback()
                let codexToken = try auth.toToken()
                token = codexToken.accessToken
                accountID = codexToken.accountID
                output = "Codex sign-in complete. Access token and account ID filled in."
            }
        } catch {
            output = "Sign-in error: \(error.localizedDescription)\n\n\(String(describing: error))"
        }
    }

    @MainActor private func run() async {
        isRunning = true
        output = ""
        defer { isRunning = false }

        let provider = provider
        let token = token
        let accountID = accountID
        let modelName = model
        let prompt = prompt

        do {
            switch provider {
            case .anthropic:
                let model = AnthropicOAuthLanguageModel(tokenProvider: { token }, model: modelName)
                let session = LanguageModelSession(model: model)
                let response = try await session.respond(to: prompt)
                output = response.content
            case .codex:
                let model = CodexLanguageModel(
                    tokenProvider: { CodexToken(accessToken: token, accountID: accountID) },
                    model: modelName
                )
                let session = LanguageModelSession(model: model)
                let response = try await session.respond(to: prompt)
                output = response.content
            }
        } catch {
            output = "Error: \(error.localizedDescription)\n\n\(String(describing: error))"
        }
    }
}
