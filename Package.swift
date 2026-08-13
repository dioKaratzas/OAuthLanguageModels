// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "OAuthLanguageModels",
    platforms: [
        .macOS(.v14),
        .macCatalyst(.v17),
        .iOS(.v17),
        .tvOS(.v17),
        .watchOS(.v10),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "OAuthLanguageModels",
            targets: ["OAuthLanguageModels"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/AnyLanguageModel", from: "0.8.0"),
        // Completes the JSON a structured answer is halfway through writing, so each
        // delta can be decoded instead of only the last one. Already in the tree by way
        // of AnyLanguageModel; named here so it can be imported.
        .package(url: "https://github.com/mattt/PartialJSONDecoder", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "OAuthLanguageModels",
            dependencies: [
                .product(name: "AnyLanguageModel", package: "AnyLanguageModel"),
                .product(name: "PartialJSONDecoder", package: "PartialJSONDecoder")
            ]
        ),
        .testTarget(
            name: "OAuthLanguageModelsTests",
            dependencies: ["OAuthLanguageModels"]
        )
    ],
    swiftLanguageModes: [.v6]
)
