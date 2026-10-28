# Decision model validation on Apple M2

Measured on 2026-10-28 with an Apple M2 and 16 GiB unified memory. These are
smoke checks and six hand-written Korean examples, not a production benchmark.
The complete answers, option probabilities, revisions, usage and measurements
are in [decision-evaluation-2026-10-28.json](decision-evaluation-2026-10-28.json).
The examples and expected answers are in
[`scripts/evaluate_decisions.py`](../scripts/evaluate_decisions.py).

## d1 preparation and server

The server downloaded `LiquidAI/d1-omni-600M` at commit
`02b55d7076f15129e59ab3f94783f32c4b088674`, converted it to FP16 in a
separate process, and loaded both checkpoints through `mlx-vlm`'s official
`load` and `predict`. The original-versus-converted maximum probability
difference for the fixed English choice, Boolean and score questions was
0.0001, below the configured 0.02 tolerance. The selected option, Boolean
result and score scale matched. The server response exactly matched the
published `LiquidAI/d1-omni-600M-MLX-fp16` checkpoint at commit
`05e99e49eb1799f1f9f98ec619b6fef2b0dd0960` for those questions.

The original weight file was 2,348,774,500 bytes; the converted model weight
file was approximately 1.17 GB. They are disk sizes, not runtime memory. For
six short questions in a standalone process, peak process RSS was 1,647 MiB
and peak MLX allocation was 1,211 MiB. These measurements overlap and must not
be added. MLX allocation and RSS do not describe all system-wide memory use.
A separate isolated gateway test with three short questions took 4.572 s for
the first request, including model loading and verification, and 0.061 s for
the repeated request. The standalone six-question check reported 0.933 s to
load and 0.076 s for its first inference. Timings vary with system load.

The original and official FP16 installation paths were both tested; the latter
was loaded and checked without conversion. The original download, conversion,
validation and registration were repeated from an empty model directory; the
source and output hashes matched the installed artifact. Retaining the source
was verified in a separate installation, as were repeating the same revision,
rejecting incompatible or damaged files, preserving an existing model on failure,
concurrent requests and cancelling preparation. The manifest records the source
and output hashes, converter options, full revision, calibration, license and
versions. Tested d1 environment: Python 3.14.3, MLX 0.32.4, `mlx-vlm` 0.7.10,
Transformers 5.21.0, `huggingface-hub` 1.33.0 and safetensors 0.8.0.

Reordering the two server choice options preserved their ID mapping; the
billing probability changed from 0.9828 to 0.9784. A request that would have
been truncated returned HTTP 422. Decision, embedding and reranking requests
were sent concurrently through isolated workers and each returned HTTP 200.
After deliberately stopping the decision worker, decision requests returned
HTTP 503 while embedding, reranking and gateway health continued to respond.
The current supervisor does not automatically restart that worker; restart
the server to restore it. The tests exercise error handling and availability,
not the relevance quality of the separate embedding or reranking models.
After restart, the gateway showed the loaded decision model in its cache,
unloaded it in its worker, and successfully loaded it again. An unload during
an active request returned HTTP 409. In a separate worker-role check, MLX
active allocation was 0 bytes before loading, 1,174,323,218 bytes after
inference, and 1,040 bytes after unloading. These are MLX allocation counters,
not total system memory; the recorded peak remains a historical maximum.

To check the changed shared `mlx-vlm` dependency, the
`mlx-community/SmolVLM-256M-Instruct-4bit` checkpoint at commit
`69cb5195f414ceb6398c5581254673d2c6c8d0d8` was loaded and converted with the
installed converter. The server's image chat endpoint and the converted
checkpoint both described a generated red test image correctly. This checks
one small VLM and does not establish compatibility with every VLM architecture.

## Small Korean comparison

The expected score level is compared to the highest-probability level, rather
than rounding the expected score. This is just an initial candidate screen.

| Example | Expected | d1 FP16 | Laya-ko MLX | Kev-0.8B MLX |
| --- | --- | --- | --- | --- |
| Document type | customer support | receipt | customer support | customer support |
| Note topic | software development | software development | software development | software development |
| Request routing | technical | technical | technical | technical |
| Renewal condition | true | true | true | true |
| Refund requested | false | true | false | false |
| Urgency level | high | high | high | high |
| Correct | | **4/6** | **6/6** | **6/6** |

For the urgency example Laya assigned only 0.3713 to the highest-probability
level; Kev assigned 0.6927, and d1 assigned 0.3808. Kev assigned 0.5951 to
the true renewal condition and 0.4355 to the false refund condition. The
sample is too small to establish calibration or choose an operating model.
The d1 model card does not list Korean among its supported languages.

Laya-ko was tested with the official `mlx-vlm` Laya implementation and a
checkpoint layout created from its source files as described in the
[upstream Laya README](https://github.com/Blaizzy/mlx-vlm/blob/v0.7.10/mlx_vlm/models/laya/README.md).
The six answers were also checked with the original Laya 0.4.15 PyTorch/MPS
implementation; the largest option-probability difference was 0.0019.
Laya is an evaluation candidate, not an installed `mlx-serve` decision profile.
To reproduce its MLX layout from a local source snapshot:

```bash
python scripts/prepare_laya_evaluation.py /path/to/laya-ko /path/to/new-laya-mlx
python scripts/evaluate_decisions.py /path/to/new-laya-mlx laya-results.json
```

Kev was tested in a separate Python 3.12.13 environment using the upstream
[Kev repository](https://github.com/jaredpalmer/kev) at commit
`2074cc44256200bb37540103fa4572901faa10f3` and its official MLX serving path.
It loaded and merged the published LoRA adapter and pointer head with the
Qwen3.5-0.8B-Base backbone. The adapter revision was
`bf75a6a8848ea6960ff2ed108d9ed44c2941174f` and the resolved backbone revision
was `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. The isolated environment used
MLX 0.32.4, `mlx-lm` 0.31.3, PyTorch 2.8.0 and Transformers 5.21.0. Kev's
`/v1/systemone` endpoint was called with the same six examples; Kev is not an
integrated `mlx-serve` profile. Its published model card describes English as
its supported language, so these examples do not establish Korean support.

Larger models were not loaded for this initial comparison on the 16 GiB host.
