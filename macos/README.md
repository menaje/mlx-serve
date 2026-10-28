# MLX Serve for macOS

This is an early native client for Apple Silicon Macs running macOS 26 or later.
The pinned MLX and Metal wheels in this runtime require macOS 26; the minimum is
not just the SwiftUI deployment target.
The app starts a local management process and can either start its own engine or
connect to an existing server. The management process only listens on loopback,
requires a per-launch secret, and never stops a server it did not start. Closing
the window leaves the menu-bar app running; quitting the app cancels its model
preparation jobs and stops its own server.

The first supported install flow is LiquidAI's original d1 checkpoint or its
official MLX FP16 checkpoint. Preparation is the same verified backend path as
the CLI. Other model types may be tested if already installed, but adding other
repositories is intentionally disabled until their conversion and validation
paths meet the same guarantees. Decision models can be removed without deleting
shared source caches; deletion of other types is disabled. Generation-worker
management, storage migration, general Hugging Face cache cleanup, and general
background-service ownership are not implemented yet. The app can preview and
remove unreferenced retained d1 originals; it rechecks the registry, files and
preparation lock before deletion, and leaves referenced originals in place.

An installed decision, embedding, or reranker model can be selected as the
default for its type. Requests that omit the model use that selection; an
explicit model ID takes precedence. The selection is not used if its registered
identity or files no longer match. Other model types do not yet support defaults.

The app can optionally start its own server when opened, and can retry an
unexpected server exit up to three times while the app remains open. It never
restarts a server the user stopped or takes control of an existing service.
The retry budget resets after a minute of healthy operation.
Notifications for unexpected server exits and model preparation outcomes are
optional and depend on the actual macOS notification permission. Notifications
are emitted only while the app is running and do not include request contents.

Configuration, model files, and job records are kept outside the app in
`~/.mlx-serve`. A fresh app install creates a local-only server configuration;
an existing CLI installation is left unchanged. Settings show saved, effective
on restart, and currently applied values separately. Existing external servers
are not reconfigured by the app. Hugging Face credentials are saved in macOS
Keychain and passed only to the preparation process for a job.
Paths under `/Volumes/<name>` must refer to a mounted volume. This prevents
starting the engine or creating a replacement directory after that volume is
disconnected. Storage location changes are blocked while existing models or
preparation jobs would be stranded; moving existing models is not yet supported.

## Building

Run `./macos/build-app.sh` on Apple Silicon with Xcode command-line tools and
network access. It fetches a checksum-verified portable Python runtime, installs
the versions in `requirements-runtime.txt`, packages the current source, and
creates `macos/.build/MLX Serve.app`. The resulting bundle uses an ad-hoc
signature and is **not notarized**. Developer ID signing, notarization, DMG
distribution, and verification on a separate Mac without developer tools still
require a release environment and credentials.

See [VALIDATION.md](VALIDATION.md) for the tested flows and known gaps.

For local UI development only, `./macos/build-app.sh --development` creates a
small bundle without a runtime. Launch its executable with `MLX_SERVE_PYTHON`
set to the project's Python executable; that development bundle is not portable.
