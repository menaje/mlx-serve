# MLX Serve for macOS

This is an early native client for Apple Silicon Macs running macOS 26 or later.
The pinned MLX and Metal wheels in this runtime require macOS 26; the minimum is
not just the SwiftUI deployment target.
The app starts a local management process and can either start its own engine or
connect to an existing server. The management process only listens on loopback,
uses an encrypted channel authenticated with a per-launch secret, and never
stops a server it did not start. Closing the window leaves the menu-bar app
running; by default, quitting cancels its preparation jobs and stops its own
server. The user can opt in to retaining the app-owned agent, jobs and server
after quitting; the connection secret is then kept in Keychain for reconnection.
When preparation requests automatic loading, the agent records and performs
that step even while the app is closed. A loading failure is recorded separately
from successful preparation so the user can load the model manually.
The app also inspects known CLI paths and the user's launchd service without
changing those services.

Supported installs are LiquidAI's original d1 checkpoint, its official MLX
FP16 checkpoint, `vllm-sr/d3-lite`,
`mlx-community/all-MiniLM-L6-v2-bf16` (embedding), and
`mlx-community/Qwen3-Reranker-0.6B-4bit` (reranker). The d3 model's BF16
weights and FP32 readout and the two retrieval models' MLX weights are used
directly after file-hash and inference checks. Preparation runs in the backend.
Other model types may be tested if already installed, but adding other
repositories is disabled until their validation paths are supported.

Generation-worker load and unload commands are routed to the appropriate
model- or type-scoped worker. A running server can remove decision models by
using the worker-aware operation. With the server and preparation jobs stopped,
registered models of the supported types can also be previewed and removed from
storage. The offline operation checks the registry, references, target files and
name; linked or otherwise unsafe directories cannot be removed this way. Stop
other services that might use those files before removing them. Originals and
shared Hugging Face caches are not removed by this operation.

The app can preview and remove unreferenced retained d1 originals; it rechecks
the registry, files and preparation lock before deletion. The Hub cache inventory
can also preview and remove individual detached revisions that are no longer
referenced by Hub refs or registered app models, retaining another revision in
the repository and preserving shared blobs. The estimated reclaimable space is
based on the Hub cache metadata. Stop other applications that use the cache
before cleaning it: their model references cannot be discovered by the app. This
is not general cache cleanup, and it does not include the Xet cache. Shared Hub
caches are not deleted automatically by any conversion path.

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
Keychain and passed only to the preparation process for a job. Keychain is built
into macOS; no separate account or Developer ID is required. A token is optional
for public models, and the default quit-and-stop mode needs no background
reconnection secret.
Paths under `/Volumes/<name>` must refer to a mounted volume. This prevents
starting the engine or creating a replacement directory after that volume is
disconnected. With the server and preparations stopped, existing model storage
can be copied to a new parent directory, checked, and switched in the saved
configuration. The original files remain in place, including after a failed or
interrupted copy. The user must stop other services that could modify the source;
linked or nested mounted files are not migrated.

## Building

Run `./macos/build-app.sh` on Apple Silicon with Xcode command-line tools and
network access. It fetches a checksum-verified portable Python runtime, installs
the versions in `requirements-runtime.txt`, packages the current source, and
creates `macos/.build/MLX Serve.app`. The resulting bundle uses an ad-hoc
signature and is **not notarized**. Developer ID signing, notarization, a
distributable DMG, and verification on a separate Mac without developer tools
are outside the current issue scope and require a release environment and
credentials if pursued later.

`./macos/package-dmg.sh` packages an already built app into
`macos/.build/MLX-Serve-experimental.dmg` for local testing. This image contains
the same ad-hoc-signed app and is not a notarized distribution.

See [VALIDATION.md](VALIDATION.md) for the tested flows and known gaps.

For local UI development only, `./macos/build-app.sh --development` creates a
small bundle without a runtime. Launch its executable with `MLX_SERVE_PYTHON`
set to the project's Python executable; that development bundle is not portable.
