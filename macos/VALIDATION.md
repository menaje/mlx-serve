# macOS app validation

The app is still under development. These results do not meet all acceptance
criteria of issue #12.

## Tested environment

- Apple M2, 16 GB RAM, macOS 27.0.1 (26A434).
- Development environment: Python 3.14.3, MLX 0.32.4, mlx-vlm 0.7.10,
  huggingface-hub 1.33.0, transformers 5.21.0.
- Bundled environment: Python 3.14.10 with the same package versions above;
  the complete pinned dependency list is in `requirements-runtime.txt`.
- The Python test suite passed 362 tests with real d3-lite and Qwen3-Reranker
  checkpoints supplied; no tests were skipped. The run emitted one third-party
  Starlette deprecation warning. Set `MLX_SERVE_D3_TEST_MODEL_DIR` and
  `MLX_SERVE_RERANK_TEST_MODEL_DIR` to local checkpoint directories to include
  those optional real-model tests in a future run.

## Observed flows

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
- The app's native Keychain code stored, updated, reopened and removed disposable
  Hugging Face and agent secrets across two processes using a temporary Keychain.
  The test did not add it to the user's Keychain search list. This does not yet
  exercise the app's complete background reconnection flow with the login Keychain.
- With the packaged app's server stopped, its read-only Hub cache inventory showed
  an isolated synthetic repository with two revisions sharing one 23-byte file.
  Automated tests also cover a missing cache, corrupt entries, authorization and
  preservation of shared cache files during other model conversions. Xet cache
  size and reclaimable space are not included in the inventory.
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
- In an isolated home, the development agent and app-owned server continued
  after their parent process exited while a real preparation job was running.
  An authenticated encrypted client reattached, cancelled the job, and stopped
  and restarted the owned server. After disabling the option and ending the
  attached parent, the agent and server exited. The protocol tests cover replay,
  incorrect keys, remote and browser-origin requests, and discovery records.
- In separate isolated homes, the agent also completed an official MLX FP16 d1
  preparation and automatically loaded it after the app's parent process exited.
  Repeated runs reused a verified artifact and exercised the current asynchronous
  load request, including one using the rebuilt app's bundled runtime. Tests cover
  load failure, cancellation during loading, and recovery of a saved load request.
  These process tests do not exercise macOS Keychain.
- Existing CLI and launchd detection is read-only; tests cover running, loaded,
  absent and uninspectable user services. System service files are identified,
  but their running state is not determined.
- After the encrypted management channel and launchd detection were added, the
  rebuilt packaged app was launched in a separate test home. Its UI displayed
  the test port and service state, started and stopped its own server, and showed
  the background option disabled. Its ad-hoc signature was checked after use.

## Remaining validation and limitations

- One development UI session terminated with an AppKit layout-constraint
  exception during testing. The model and job layouts were later simplified
  after a separate list-clipping issue was found. Subsequent UI sessions
  completed the flows above, but the original exception has not been reliably
  reproduced or definitively resolved.
- Sleep/wake, real login-item registration and approval, physically disconnected
  external storage, signed installer updates, and the minimum macOS 26
  deployment target have not been tested end to end.
- The bundle has an ad-hoc signature only. Developer ID signing, notarization,
  distribution of a release DMG, and installation on a separate Mac without
  development tools remain unverified.
- General Hugging Face cache cleanup remains unimplemented. Defaults are limited
  to decision, embedding and reranker models. See `README.md` for the supported
  model scope. The optional background mode's actual Keychain and app relaunch
  flow has not been exercised end to end.
- Notification transition filtering is tested, but the macOS permission prompt
  and visible delivery have not been exercised end to end.
