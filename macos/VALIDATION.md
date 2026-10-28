# macOS app validation

The app is still under development. Issue #12 excludes Developer ID signing,
notarization, signed updates, a separate development-tools-free Mac, actual
system sleep/wake, and login execution after a reboot from its current
completion criteria. Other unverified flows are listed below.

## Tested environment

- Apple M2, 16 GB RAM, macOS 27.0.1 (26A434).
- Development environment: Python 3.14.3, MLX 0.32.4, mlx-vlm 0.7.10,
  huggingface-hub 1.33.0, transformers 5.21.0.
- Bundled environment: Python 3.14.10 with the same package versions above;
  the complete pinned dependency list is in `requirements-runtime.txt`.
- The Python test suite passed 397 tests with real d3-lite and Qwen3-Reranker
  checkpoints supplied; no tests were skipped. The Starlette test-client
  deprecation was fixed with the HTTPX2 development dependency, and this run
  treated that warning as an error. Set `MLX_SERVE_D3_TEST_MODEL_DIR` and
  `MLX_SERVE_RERANK_TEST_MODEL_DIR` to local checkpoint directories to include
  those optional real-model tests in a future run.

## Observed flows

The current public-only, quit-to-stop build was exercised through the packaged
app UI in an isolated home after the Mac was unlocked. Earlier builds and
additional native and backend harness checks are identified below.

- In a clean test home, the model-add UI defaulted to
  `LiquidAI/d1-omni-600M-MLX-fp16` with automatic loading enabled. A fresh
  download of revision `05e99e49eb1799f1f9f98ec619b6fef2b0dd0960` passed
  compatibility and inference validation, registered and loaded without local
  conversion. Bool, choice and score tests succeeded in the app. After making
  it the default, a separate client successfully sent a decision request with
  no model ID. Synthetic Hugging Face credentials were present in the app's
  environment and test token file during this public download.
- A fresh Qwen3-Reranker download was in progress when the app was quit. The
  app-owned server, agent and job runner exited, the job was recorded as
  cancelled, and its staging directory was removed. After relaunch, the model
  and job history remained available. Retrying the cancelled job from the UI
  downloaded revision `5f324548f1d20c2b5a450f126fc6ef2fb1126524`, verified it,
  loaded it without conversion, and ranked the matching test text higher.
- Closing the main window left the app and its owned server running; reopening
  it showed the same model and server state. An independently started server
  was identified as external, could not be stopped through the app, and kept
  responding after the app quit. Only the isolated test server was stopped at
  the end of that check.
- In the final packaged app, the isolated owned server was suspended for 25
  seconds and then resumed. The UI changed from healthy to degraded while the
  server did not respond, returned to healthy after it resumed, and retained
  the completed and cancelled job history. Quitting stopped the test server and
  workers. This exercises a temporary outage, not an actual system sleep/wake.
- Queue size, request timeout, memory thresholds and server-log retention were
  changed through the current settings UI. Saved and applied values were shown
  separately until restart, after which the server reported the new values. An
  invalid timeout was rejected. On restart, only the expired completed test
  log was removed; recent, unconfirmed, unrelated and legacy logs remained.
  The packaged app's own login-item control was registered and unregistered,
  with the displayed status changing accordingly; it was left unregistered.
- An ad-hoc-signed test copy launched from a temporary folder could not request
  notification permission on this Mac. The app now leaves notifications off
  and displays the failure and suggested next steps; this was checked in its
  settings UI. A second test copy in the user's Applications folder did produce
  a macOS authorization request. These observations do not establish how every
  macOS version or installation location behaves.
- In another installed test copy, the user allowed notifications. A deliberate
  unexpected exit of its isolated server, a model job with a nonexistent
  revision, and a successful job reusing a verified model each produced a
  separate notification. The macOS notification service recorded delivery to
  Notification Center for all three. It also reported silencing interruptions
  because the display was being shared; a visible banner was not independently
  observed. The failed job showed its error and suggested checking the revision.
- The user reported a sleep and wake while the installed test app was running.
  The same app, management agent and server processes were present afterward,
  and the server was healthy. The model had become unloaded after more than its
  30-minute idle cache lifetime; reloading it and running a bool inference in
  the UI succeeded. The power log recorded the display turning off and on but
  did not record a system sleep/wake transition in that interval, so this check
  establishes display sleep/wake recovery rather than full system sleep/wake.

- A native harness compiled from the current `AppStore` started the actual
  agent and its server in an isolated home. After the harness exited normally
  or was killed, the agent, server and observed worker processes stopped. Both
  cases were repeated with a model-preparation job waiting on an isolated test
  Hub: the runner stopped, the job was recorded as cancelled, and its staging
  directory was absent. A separate test used an independently started server
  and confirmed that it kept running after the harness exited. The lifecycle
  cases were exercised with the development and bundled Python runtimes. Tests
  using both runtimes also killed the management agent itself; the app-owned
  server stopped and a job blocked on the test Hub was bounded and subsequently
  recovered as interrupted. A log whose process group could not be confirmed
  stopped remains ineligible for automatic log cleanup.
- Public-model jobs reject token input, clear inherited Hugging Face token
  variables, and disable implicit cached credentials in their child processes.
  Tests placed synthetic credentials in both the environment and token file,
  then verified that the actual Hugging Face header builder sent no
  authorization header. With synthetic credentials present, fresh public
  embedding and reranker downloads, validation and inference also succeeded.
  The app-owned server uses the same credential policy; existing CLI
  configurations and credentials are not modified.
- With both the development and bundled runtimes, the official d1 MLX FP16
  checkpoint was copied from an existing local installation, validated, and
  used for choice, Boolean and score inference without invoking a converter.
  Regression tests also check that only the original d1 checkpoint takes the
  conversion path and that older non-converted records do not report a local
  converter.
- Queue and memory limit settings were checked for optional values, valid
  boundaries, atomic saves and rejection of invalid values. A native, non-UI
  harness checked the current Swift input and display helpers. Tests also check
  server-log completion and retention: only expired completed app-owned logs
  are eligible, and active, legacy, unrelated and linked files are preserved.

- In an isolated, initially empty home directory, the management agent using
  the development runtime downloaded `LiquidAI/d1-omni-600M` revision
  `02b55d7076f15129e59ab3f94783f32c4b088674`, converted it to MLX FP16,
  verified and registered it, then served bool, choice and score requests.
  The worker was unloaded and the app-owned server was stopped. This was a
  real download and conversion, but it was driven through the management API,
  not through the packaged app UI.
- In another isolated home, a real `vllm-sr/d3-lite` download at revision
  `03560c26f5199c3c04129a2ccb38960c913635e0` ran through the persistent job
  manager. It checked the published file hashes and tensor formats, ran choice,
  Boolean and score inference with MLX, and registered the verified artifact.
  An independent real-model test also matched the recorded Ollama reference.
  A second real d3 download was cancelled after it started: the worker and its
  staging directory stopped and an unrelated fixture remained intact.
- The rebuilt packaged app was launched with a separate test home containing
  that verified d3 artifact. Its UI loaded and tested the model, then added a
  second model by reusing the verified revision and automatically loaded it.
  A separate local API client successfully exercised choice, Boolean and score
  requests against that model; the UI then unloaded it. This part reused a local
  checkpoint. The model detail also showed the recorded license, preparation
  options and validation; these are recorded results, not a fresh checksum on
  every refresh.
- In a separate empty home, the packaged app started a fresh d3 download. The
  UI cancelled it during transfer, confirmed the cancelled state after the
  worker and staging directory were gone, and retried it as a new job. The
  retry downloaded and verified the checkpoint, automatically loaded it, and
  returned a bool result in the app. A separate local client successfully sent
  choice, bool and score questions before the model was unloaded. The test used
  an environment override for the loopback port and disabled Xet transfers.
- The packaged app loaded, tested and unloaded pre-existing
  `mlx-community/all-MiniLM-L6-v2-bf16` and
  `mlx-community/Qwen3-Reranker-0.6B-4bit` models in the isolated home. The
  first reranker test exposed incorrect ranking; after applying Qwen's input
  framing and FP32 probability
  calculation, a repeated UI test scored the matching text about 0.999 and the
  unrelated text about 0.0003. An independent real-model test also checked a
  relevant document against an unrelated one and compared single and batched
  scores. These pre-existing models were installed by the development CLI, not
  by the app's model-add UI.
- In an isolated home, the persistent job manager downloaded and prepared
  `mlx-community/all-MiniLM-L6-v2-bf16` revision
  `b6691709eacd8f0afcc3faace288cf50e611f3aa` and
  `mlx-community/Qwen3-Reranker-0.6B-4bit` revision
  `5f324548f1d20c2b5a450f126fc6ef2fb1126524`. It checked the published file
  set, sizes and hashes, verified the model architecture and precision, ran MLX
  inference, and registered both models. Independent inference returned a
  384-dimensional embedding and ranked the matching reranker example higher.
  The same two fresh downloads and inference checks also succeeded using the
  packaged Python runtime and a separate empty storage directory; that run
  invoked the job manager directly.
  The management agent using the packaged runtime then reused verified copies,
  automatically loaded and tested both through an owned server, and removed
  one copy while the server was stopped without removing its source. The new
  model-add UI itself was not exercised during this run. With the app's
  bytecode-writing protection applied, the rebuilt bundle's ad-hoc signature
  still verified after these operations.
- In an earlier packaged app build with its server stopped, the Hub cache
  inventory showed
  an isolated synthetic repository with two revisions sharing one 23-byte file.
  Automated tests also cover a missing cache, corrupt entries, authorization and
  preservation of shared cache files during other model conversions. The newer
  management endpoint previewed and removed one detached revision in an isolated
  cache while preserving a shared blob, the current revision and an unrelated
  model. Tests cover changed references and files, corrupt metadata, shared
  files, symlinked paths, authorization and server state. Xet is not included.
- On the same Mac, the packaged app was also launched with a separate empty
  home directory. Through its UI, the original d1 repository was downloaded,
  converted to MLX FP16, validated, registered and automatically loaded. A
  bool test succeeded in the app, an independent API client received the same
  answer, and the model was unloaded. The bundle's ad-hoc signature remained
  valid after preparation and inference. The server port was supplied through
  an environment override to avoid other services on the test machine.
- On the same Mac, the ad-hoc-signed bundle was launched with a clean process
  environment and an isolated home directory. Its bundled Python started the
  server. A previously verified d1 model was copied into the isolated home;
  the UI loaded it, sent a bool test, unloaded it, and prepared a second
  registration by reusing the verified model. Quitting stopped the app-owned
  server. This is not a test on a separate Mac without development tools.
- The bundle was subsequently rebuilt at the same path and relaunched using
  that same isolated home. The UI still showed the two registered models, the
  completed job, and the previously saved performance settings. This checks
  data persistence across a local rebuild, not a signed installer update.
- In the development UI, changing the port while the server was running
  displayed the saved and applied values separately and marked restart as
  required. Closing the window kept the server available; quitting stopped it.
  The model and job lists were checked with multiple models and a saved job;
  editable choice and score tests returned results from the loaded d1 worker.
- An app-owned server bound to IPv6 loopback also started, reported worker
  state through the management channel, and stopped successfully.
- Automated tests cover local management authorization, port conflicts,
  proxy and redirect isolation, settings validation, job persistence,
  cancellation and interrupted-job cleanup. A separate M2 smoke test also
  exercised reuse and cancellation.
- A locally created disk image was detached while the packaged app's server
  was running. The app showed storage unavailable and blocked model addition;
  after reattachment, the registered test model reappeared. Starting the server
  with the image detached was blocked without recreating its mount point, and
  the image's data remained intact on return. This simulates a disconnected
  volume; it is not a physical-drive test.
- After a separate test terminated the app-owned server, the UI showed failure,
  allowed it to be restarted, and still showed the model and job history. This
  used a graceful termination signal, not a sudden power loss.
- With automatic start and recovery enabled through a test-only process
  argument, the packaged app started its server, recovered after a graceful
  termination signal, and did not restart after the server was stopped through
  the UI. The policy's bounded retries and external-server behavior were also
  tested independently.
- In the packaged app, a d1 model was selected as the default and a separate
  client successfully sent a real decision request with no model ID. The
  selection remained visible after a local rebuild and app restart. The
  notification settings displayed the current macOS authorization state.
- In the packaged app, a preview with isolated synthetic source directories
  showed an unreferenced original and excluded another referenced by the model
  registry; the destructive confirmation was inspected and cancelled. Automated
  tests performed cleanup on temporary files, checked shared references, links,
  changed targets and active preparation, and simulated move and deletion failures.
  The final bundled runtime also cleaned a disposable source directory and left
  a referenced directory intact in an isolated temporary fixture.
- A compressed experimental DMG was created and mounted read-only on this Mac;
  it contained the bundled runtime, an Applications link and an app whose ad-hoc
  signature still verified. This did not test installation or Gatekeeper.
- With the development runtime, a real SmolLM2 135M MLX model was loaded,
  queried and unloaded through both generation-worker isolation modes. Tests
  also covered worker startup races, active streams and unavailable workers.
- On the same Mac, an isolated storage migration copied that model, verified
  the files and saved settings, then loaded it and served an inference from the
  new location. The source remained intact. The packaged app also previewed
  and copied a disposable synthetic model through its settings UI. Automated
  tests cover interrupted copies, changed files, concurrent configuration,
  links, insufficient space and failed saves.
- Existing CLI and launchd detection is read-only; tests cover running, loaded,
  absent and uninspectable user services. System service files are identified,
  but their running state is not determined.
- After the encrypted management channel and launchd detection were added, the
  rebuilt packaged app was launched in a separate test home. Its UI displayed
  the test port and service state, and started and stopped its own server. Its
  ad-hoc signature was checked after use.
- A separate temporary ad-hoc test app registered and unregistered a real
  `SMAppService` login item. It reported enabled after registration and
  unregistered afterward. This did not test the packaged app's UI or login
  behavior after a reboot.
## Remaining validation and limitations

- One development UI session terminated with an AppKit layout-constraint
  exception during testing. The model and job layouts were later simplified
  after a separate list-clipping issue was found. Subsequent UI sessions
  completed the flows above, but the original exception has not been reliably
  reproduced or definitively resolved.
- System sleep/wake and login execution after a reboot are excluded from the
  completion criteria and were not tested end to end. The packaged app's
  login item was registered and unregistered on this Mac. Physically disconnected
  external storage and the minimum macOS 26 deployment target also remain
  unverified end to end.
- The bundle has an ad-hoc signature only. Developer ID signing, notarization,
  distribution of a release DMG, and installation on a separate Mac without
  development tools and signed installer updates are outside the current issue
  scope and remain unverified.
- General Hugging Face cache cleanup is not implemented; cleanup is restricted
  to individually reviewed detached revisions as described in `README.md`.
  Defaults are limited to decision, embedding and reranker models, and new model
  installation is limited to the documented profiles. The cleanup and
  offline-removal controls have backend and earlier UI coverage as described
  above, but were not re-exercised in the current full UI pass.
- Notification transition filtering and macOS delivery for all three event
  types were verified. A visible banner was not independently observed while
  display sharing caused macOS to silence interruptions.
