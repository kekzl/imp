# Changelog

All notable changes since v0.6. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

**This is a changelog, not a journal.** One to three lines per entry: what
changed, from the reader's side, plus the number or issue that makes it
checkable. The investigation behind a change — hypotheses, what was ruled out,
how it was measured — belongs in `docs/` (`quantization.md`, `roadmap.md`,
`AUDIT.md`, `docs/plans/`) or in `docs/MISSION_JOURNAL.md`, and the entry links
there instead of retelling it.

## [Unreleased]

### Fixed
- **Constrained decoding dropped every non-ASCII character** (#1197). With
  `response_format: json_schema` or `json_object`, "Die Bären hören" came back as
  "Die Baren horen" — German, and every other non-English language, was
  unusable. The FSM was never at fault: it compares through `unsigned char` and
  accepted umlauts all along. `classify_token()` did not, and `char` is signed,
  so every byte of a multi-byte UTF-8 sequence read as negative and lost
  `CAT_STRING_CHAR` — the category mask then dropped those tokens *before* the
  FSM was consulted, and the model spelled the nearest ASCII word it was allowed
  to emit. GBNF and regex constraining were never affected (they do not use that
  pre-filter). Pinned by `TokenCategory.NonAsciiCountsAsStringContent` in the CPU
  lane, and the json_object property batteries now generate non-ASCII instead of
  documenting its absence as deliberate.
- **An image sent to a model that cannot see it is refused, not ignored**
  (#1198). A multimodal SafeTensors checkpoint whose tower imp does not
  understand (Gemma-4 NVFP4, the Qwen3.5/3.6 MoE checkpoints) loads text-only and
  says so in the load log — but `image_url` parts were then accepted and answered
  from the text alone, so the caller got a confident description of a picture the
  model never received. Now `400` with code `vision_unavailable`, in every
  dialect. `docs/supported-models.md` states which checkpoints can see and which
  cannot.

## [0.20.1] - 2026-08-01

### Changed
- **CUDA 13.3.0 → 13.3.1** (nvcc V13.3.33 → V13.3.73), the newest toolkit there
  is — there is no 13.4 or 14.x. Same release string, so CI's nvcc check and the
  ccache keys are untouched, and the driver's UMD is 13.3 either way. Measured
  perf-neutral: decode tg128 287.95 tok/s (median of three) against 288.38 on
  13.3.0, both against the 287.19 baseline.
- **`imp-quantize --calib` says where it is validated and where it is not.** A
  model too big to run in BF16 can be calibrated off any quantization of itself
  (the statistics are keyed by layer and tensor kind, not by checkpoint) — which
  produced the first `--calib` result on Qwen3-14B, and it is negative: PPL
  **9.9252 uncalibrated vs 12.6016 / 12.2853** calibrated, from two independent
  twins. It still helps on Qwen3-0.6B/1.7B. Recipe and what was ruled out:
  `docs/quantization.md`.
- **A dead `docs/…` pointer in a code comment is now a gate failure.** Sixteen
  design memos were deleted in doc-consolidation PRs (#183, #273, #441) and the
  38 comments citing them stayed behind, so following one led nowhere. They now
  name what survived — or state the finding inline, where nothing did.

### Fixed
- **`--set` with a key that does not exist is now an error, not a warning.** A
  typo silently measured the default instead: `tools/analysis/awq_ppl_ab.sh`,
  the harness the AWQ result names for reproduction, passed
  `--set gemm.deterministic=true` — the key is `runtime.deterministic_gemm` —
  so its three scoring runs never got the determinism they asked for. (Re-run
  with the key fixed, the published numbers reproduce unchanged: BF16 24.0641,
  round-to-nearest 30.0979, AWQ 28.4782.) An unknown key in `imp.conf` stays a
  warning: a config file may outlive the build that understood every key in it.

## [0.20.0] - 2026-08-01

### Added
- **Several images in one request** (Qwen3-VL). Every `image_url` part is
  encoded, in prompt order, into one concatenated embedding; each placeholder
  expands to its own picture's token count. `imp-cli --image` repeats, `/image`
  stacks before a turn, and the C API gained `imp_add_image{,_from_memory}`.
  Previously the last image silently won. An `image_url` that cannot be read is
  now a 400: dropping one would slide every later picture onto the wrong
  placeholder. The mmproj tower (Gemma-3/4) still takes one and refuses more.
- **Vision: Qwen3-VL** (#1163-#1180). imp describes images end to end, from
  `imp-cli --image` and from `/v1/chat/completions`. Dynamic resolution (no
  fixed image size — a 1795x2397 photo becomes 972 image tokens), DeepStack
  taps injected at the LM's first layers, and three-axis M-RoPE. Text-only
  models and text-only prompts are bit-identical to before.
  Plan and inventory: `docs/plans/2026-07-31-qwen3-vl-vision.md`.
- **`POST /v1/rerank`** (also `/rerank`) — Cohere/Jina/vLLM-compatible, scoring
  query and document jointly in one forward. Supports `top_n`,
  `return_documents`, `instruction`, Cohere's object documents and vLLM's
  `texts`. Validated against llama.cpp on the same GGUF (top-1 agreement 3/3,
  median score delta 0.0014). Needs a reranker model; a general model is
  refused with a 400. Gate: `make test-rerank`.
- **GBNF grammar-constrained decoding** — `response_format: {"type":"grammar"}`,
  llama.cpp's `grammar` and vLLM's `guided_grammar`. Recursive formats a regex
  cannot express. Grammars the simulator cannot honour (left recursion,
  undefined rules, missing `root`, unbounded repetition) are refused rather
  than mis-enforced.
- **Regex-constrained decoding** — `response_format: {"type":"regex"}` and
  vLLM's `guided_regex`. Constructs a DFA cannot honour (lookaround, anchors,
  `\b`, backreferences) are refused.
- **`imp-quantize` (EXPERIMENTAL)**: first-party BF16/FP16 → NVFP4 checkpoint
  conversion, writing the layout the loader already reads. Sharded sources
  supported. With `--calib` (below) Qwen3-0.6B scores PPL 28.48 vs BF16 24.06;
  without it 30.10. Details and caveats in `docs/quantization.md`.
- **AWQ-class activation calibration** (`imp-cli --calibrate` +
  `imp-quantize --calib`). Measured over a 13.5k-token corpus: Qwen3-0.6B
  30.10 → **28.48**, Qwen3-1.7B 20.43 → **19.21**; `degen_suite.py` 45/45.
  `--calibrate` forces `runtime.deterministic_gemm` — without it two identical
  runs differ on 94% of recorded floats and the resulting checkpoints spread
  1.6% in perplexity. Refuses architectures whose norm applies `(1 + g)`
  (Gemma-class) rather than emitting a different model. Reproduce with
  `tools/analysis/awq_ppl_ab.sh`.
- **`evicted_tokens`**: the caller is told when StreamingLLM dropped context —
  `usage.prompt_tokens_details.evicted_tokens` (chat-completions),
  `usage.input_tokens_details.evicted_tokens` (`/v1/responses`),
  `usage.imp_evicted_tokens` (`/v1/messages`). The key is absent unless
  eviction fired, so its presence is the signal.
- **Model swapping on request** (`server.model_swap`, default on). A request
  naming another model in the models directory swaps to it instead of 404ing.
  In-flight generations drain first; a failed load restores the previous model.
  `/v1/models` lists the directory with `loaded: true|false`.
- **Web UI at `GET /`** — single-page chat client embedded in the binary,
  streaming over the existing SSE endpoint, with per-token latency bars and a
  separate thinking channel.
- **SWA window snapshots** (`kv_cache.swa_snapshot_mb`, default 0): prefix
  caching and SWA-aware KV sizing now combine. Opt-in — the sized attention
  route trades ~50-100 ms warm TTFT at 1-2K contexts for faster long prefill
  (+8% at 13K) plus the KV savings.
- **Qwen-Coder XML tool-call grammar** for templates teaching the
  `<function=NAME>`/`<parameter=KEY>` dialect (Qwen3-Coder, Qwen3.6). Measured
  0/3 → 3/3 compiling `write_file` contents on Coder-30B.
- **External agent gates** (`make test-agents-external`): aider over
  chat-completions, Claude Code over `/v1/messages`, and the OpenAI Agents SDK
  over `/v1/responses`. Each must land a real edit in a throwaway repo.
- **Generative property batteries for both constrained-decode FSMs**, in the
  CPU `unit` lane, checked against nlohmann/json as an independent oracle and
  validated by mutation.
- **Measurement tools**: `tools/analysis/agentic_compare.py` (cross-engine
  agentic reliability; results in `docs/BENCHMARKS.md`) and
  `tools/analysis/ctx_capacity_decode_sweep.sh` (decode throughput against
  configured context capacity — the regime the CI perf gate cannot see).

### Changed
- **SWA-aware KV sizing is tri-state `auto|on|off`** (default `auto`): the
  savings (gpt-oss ~2x, gemma-3/4 ~5-6x KV tokens) are taken only when prefix
  caching is off, so warm-prefix TTFT is untouched. Legacy bools still parse.
- **Auto `max_seq_len` ceiling raised 64K → 128K** (#1004).
- **Engine-persistent (T2) arena for `compute/` scratches** (A7 step 8). The
  CUTLASS grouped workspace reservation drops 512 MiB → 1 MiB: measured
  `get_workspace_size()` returns 152 320 B across every geometry tried, a
  property of the chip rather than the problem. Qwen3-30B-A3B-NVFP4 own peak
  VRAM 20932 → 20454 MiB; dense models and decode unaffected.
- **`--mem-report` names every `VRAMAllocator` charge** instead of estimating
  the executor's. Diagnostics only.
- **One `needs_constrained` flag** replaces `needs_json_mode` /
  `needs_schema_mode`; regex and GBNF now take the same pipelined path.
  Measured neutral on a 4B model, shipped for consistency rather than speed.
- **"Prefer a published Modelopt checkpoint" no longer stands unmeasured.**
  Against a bit-identical Modelopt export of Qwen3-14B, same corpus and engine:
  **Modelopt 10.0301, `imp-quantize` without `--calib` 9.9252**. One model, one
  corpus — enough to retire the blanket advice, not to reverse it.
- `tools/imp-server/tool_call.cpp` split (the Gemma-4 dialect parser moved to
  `tool_call_gemma.cpp`), clearing the repo's only hard-review file-size
  violation.
- **`scripts/check-release.sh` runs in CI** as `Release hygiene`, and now also
  pins the version to one value across `CMakeLists.txt`, `CHANGELOG.md` and
  `docs/BENCHMARKS.md`. It was wired into nothing before, so it ran only when
  someone remembered to: cutting this release found two maintainer paths that
  had been on `main` for days.

### Fixed
- **A MoE checkpoint whose experts imp cannot read now fails to load** instead
  of routing through null experts and generating garbage. Measured on
  `gpt-oss-20b` in BF16, whose experts are 3-D stacks only the MXFP4
  `_blocks`/`_scales` matcher looks for: every layer logged "unrecognised layer
  weight", the load succeeded, and generation produced `:!!!!!!!!!!`. The check
  fires only when the config declares experts and not one layer carries any
  expert representation, so a partially mapped MoE stays a warning.
- **The persisted prefix cache dropped the KV scales**, so a restored block
  decoded against whatever scales were in the pool — wrong attention, no error.
  Only on a KV dtype with a separate scale pool (NVFP4, INT8, INT4, MXFP4_KV),
  which is why the FP16/FP8 defaults never showed it; `--prefix-cache` plus
  `--kv-nvfp4` was enough. The file format now carries the scales
  (`kPrefixCacheVersion` 3, older files are discarded as before). Pinned by
  `PrefixPersistTest.QuantizedKvRestoresItsScales`.
- **An image that spanned a prefill chunk boundary got the wrong half of
  itself.** Both vision kernels find "the k-th image token" by scanning the span
  they are handed, which under chunked prefill is one chunk — so a second chunk
  restarted at the image's *first* embeddings. Reachable on defaults (chunks are
  2048 and Qwen3's FP8-KV path is chunk-eligible) as soon as enough text precedes
  the picture. Pinned by `tests/test_vision_chunk_offset.cu`.
- **`/image` in the interactive CLI loaded a picture the prompt never
  referenced.** The multi-turn path branched on the mmproj tower alone, so on
  Qwen3-VL it rendered a prompt with no image tokens: "Image loaded", then an
  answer given as if there were no image.
- **`imp-quantize` wrote a "quantized" checkpoint whose experts were untouched.**
  The 3-D refusal sat behind a `.weight` name test, and no real stacked
  checkpoint names its experts that way (`experts.gate_up_proj`,
  `..._blocks`), so they were copied through as BF16 with no message, no
  counter and no exclusion entry — while `hf_quant_config.json` announced
  NVFP4. Such a checkpoint is now refused before anything is written. The
  selection rule moved to `tools/imp-quantize/tensor_policy.cpp` and is covered
  by `tests/test_quantize_policy.cpp` in the CPU lane.
- **`imp-quantize` silently produced a broken MoE checkpoint.** "MoE is left
  unquantized" only ever applied to 3-D stacked tensors; the HF-standard
  per-expert 2-D layout was quantized and produced a model that loaded and then
  emitted garbage. The experts were not the cause — the **MLA latent
  projections** and the **MoE router** are, and both are now refused.
  DeepSeek-V2-Lite: 29.26 → 8.91 GiB (3.28x), `degen_suite.py` 3 FAIL/32
  against the BF16 source's 5 FAIL/32.
- **The prefix cache could serve one request's image to another.** It is
  addressed by token ids, and every image token carries the same id, so two
  requests with different pictures shared a prefix. Block hashes are now salted
  with the image content: a hit needs the same tokens *and* the same picture.
  Affected the mmproj path too, not only Qwen3-VL.
- **Decode paid for context capacity it never used — up to −38% on the served
  path** (#1100). The NVFP4 decode cache's reservation was subtracted from the
  budget it spends from. Same 280-token request at `max_seq_len` 1024 → 40960:
  **160.10 → 99.29 tok/s before, 162.77 → 163.24 after**. No effect on the
  pinned perf baseline, which never bound that budget.
- **The library reserve was measured through a window that missed most of it**,
  costing one model 4x its KV capacity. It is now charged across the whole
  init. Qwen3.6-35B-A3B-NVFP4's second start gets **16 384 tokens instead of
  4096**; attribution 98.3/96.6/89.7% → 99.9/99.9/100%. Related: the remembered
  reserve never survived a `docker run --rm` (the cache path is inside the
  container — `imp.conf.example` now spells out the invocation), a missing
  measurement is reported at plan time instead of after the first forward, and
  `MemAccount`'s per-pool ledger is no longer gated on `--mem-report`.
- **The first `response_format` request after a model load could come back
  unconstrained** (#1104). `JsonConstrainer` allocated its allow list lazily
  mid-decode; under VRAM pressure the allocation failed and `apply_mask`
  returned without masking and without logging. It now allocates at init like
  its three siblings.
- **`top_k` above 128 sampled from the previous decode step's candidates**
  (#1142). `cub::DeviceTopK::MaxPairs` writes nothing from its second call
  while returning `cudaSuccess`, and nothing checked the code. Replaced with a
  full-vocabulary sort; `degen_suite.py` gains `top_k` 129 and 2000 cases.
- **The sampler drew the same quantile on every token.** Seeds are
  `base_seed + step`, and an LCG's first output is affine in its seed, so the
  draw was effectively constant and the sampler kept picking the same rank.
  Fixed with a splitmix32 finalizer; same seed still gives the same token.
- **A CUDA graph could replay a kernel whose scratch pointer had been freed**
  (AUDIT B13). Six `compute/` statics grew with `cudaFree` + `cudaMalloc`; they
  now come from the T2 arena, which never frees. The lazy CUTLASS workspace
  growth path is removed outright — it was reachable under capture, where
  `cudaMalloc` is illegal, and no in-tree caller could reach it.
- **Streaming leaked the chain of thought as the answer whenever tools were
  present** — the half every real agent client sees. A tool request renders a
  pre-closed think block, so the model emits only the closer and the splitter
  waited for an opener that could never arrive. Found by pointing the real
  Claude Code binary at imp-server.
- **Streamed non-ASCII text was corrupted** (`"größer"` → `gr<?><?>ßer`). Two
  causes: a BPE token can end mid-character (now stitched at detokenization),
  and the stop-sequence holdback cut at a byte offset (now at a codepoint
  boundary). Every non-ASCII script was affected.
- **`json_object` accepted a trailing comma** (#1096) and the schema-less FSM
  emitted structurally invalid JSON (#1067). Both are container-stack bugs; the
  schema FSM was never affected.
- **Llama 3.2 tool calls were dropped** — 3.2 emits a bare JSON object where
  3.1 used the `<function=F>` envelope. Accepted now, but only for a name the
  request actually offered: a fabricated call is worse than a missed one.
  Llama-3.2-3B goes 4/6 → 6/6 on the agentic comparison.
- **An undersized `kv_cache.swa_snapshot_mb` silently disabled prefix caching** —
  strictly worse than `0`. It now warns with the required size and both ways out.
- Tool-call enforcement derives from the post-load template (it was collected
  before `ensure_model_loaded`); multi-turn tool replay re-renders in the XML
  shape the model emits; Qwen XML close tags are matched newline-anchored so
  raw values may contain bare close-tag text.
- `--mem-report` counted the FP8 SSM sidecar twice. Diagnostics only.

### Removed
- **`speculative.recycle_loop`** (verify-in-loop) and its ~1.5k LOC of support.
  A nine-class prompt sweep found no class where the loop beat the same
  configuration with it off; isolated against the eager drafter it cost a
  consistent 5.6-8.3%. The eager `speculative.token_recycling` drafter stays
  (measured neutral). A stale key in an existing `imp.conf` warns, not errors.

## [0.19.2] - 2026-07-17

Hardening release: a latent KV prefix-cache corruption fix plus a
diagnostics/robustness sweep. Decode measured neutral at every step
(Qwen3-Coder-30B NVFP4 spec-OFF tg256 402.6 ± 0.3 tok/s across all A/Bs;
`docs/audit/PERF_LOG.md`). Baseline snapshot and the claim-verification
matrix for the external hardening brief that seeded the campaign:
`docs/audit/DISPATCH_BASELINE_2026_07_17.md`.

### Fixed
- **KV prefix-cache block double-ownership after rollback** (PR #1044):
  `rollback()` and the partial-allocation rollback freed hash-registered
  blocks to ref 0 without erasing the `block_hash_to_id_` entries; a later
  same-prefix allocation hit the stale entry, took the "actively
  referenced — share it" branch and `inc_ref`'d a block sitting in the free
  list, so the next free pushed it into the free list twice
  (`num_free_blocks` exceeded the pool; silent cross-request KV corruption).
  Production trigger: KV-pool pressure during prefix-cache allocation
  followed by a same-prefix client retry — prefix caching is default-on.
  Both rollback paths now drop the hash entries when a free reaches ref 0,
  and the reuse path treats a ref==0 non-cached hash hit as a loud miss.
  Found by the new `LeakUnderSustainedChurn` regression test (200 sustained
  alloc/prefix/rollback/free/evict cycles with exact pool-baseline
  assertions).
- **Sticky CUDA error after the expected graph exec-update fallback**
  (PR #1048): a failed `cudaGraphExecUpdate` (topology changed) is a
  handled reinstantiate path, but the stale per-thread error lingered until
  engine teardown's leak net cleared it with a WARN. Cleared at the
  fallback site.
- Last two compiler warnings cleared — a full rebuild is now 0 warnings
  under `-Wall -Wextra -Wpedantic` (PR #1044).

### Added
- **Post-launch CUDA error checks at 399 kernel-launch sites** (PR #1044):
  new `IMP_CUDA_CHECK_LAUNCH()` (cudaPeekAtLastError-based — logs file:line
  at the launch site without clearing the sticky error, so downstream
  `IMP_CUDA_CHECK_*` propagation is unchanged). Launch-config failures now
  surface where they happen instead of at the next synchronizing call;
  coverage went from ~1% to 100% of `<<<>>>` sites in src/.
- **RAII owners for CUDA graph handles** (PR #1045): `CudaGraph` /
  `CudaGraphExec` move-only wrappers (`core/cuda_raii.h`), adopted by
  `CudaGraphCapture`, `CudaGraphConditionalRunner`, the per-bucket
  spec-verify graphs and the spec-capture locals — every manual
  destroy+null pair replaced, error/throw paths structurally leak-safe,
  semantics preserved 1:1.
- **`cache_control` per-breakpoint pin boundary** (#1046, PR #1049): the
  LAST marked system/message block now bounds the prompt-KV pin instead of
  always pinning the whole prompt (internal `cache_prefix_messages` →
  truncated re-render → token boundary, rounded down to full blocks).
  Tighter pins reduce pin-budget pressure in multi-turn agent loops. A
  marker on tools keeps the whole-prompt pin; TTL tiers are accepted but
  not modeled. Additive — unmarked requests unchanged. 7 new contract
  tests.
- **`make asan`** (#1047, PR #1049): reproducible host-code ASan+UBSan run
  over the CPU test binaries, WSL2-capable (unlike compute-sanitizer).
  Suppressions in `tools/sanitizers/` cover vendored-stb intentional
  unaligned stores and NVIDIA driver one-time allocations. Baseline: 0
  imp-code findings.

## [0.19.1] - 2026-07-17

### Added
- **Per-layer attention routing for heterogeneous models (Gemma-4 dual
  head_dim 256/512)** (PR #1042): the coarse model-level `force_cublas`
  gate is replaced by per-layer dispatch — the hd=256 SWA majority (5:1)
  rides FA2 f16-QK, hd=512 global layers stay on the faster materialized
  cuBLAS path, with a new fused WMMA FMHA hd=512 instantiation (Bq=16,
  Bkv=32, ~82 KB SMEM) as the O(n)-memory terminal fallback. New
  `attention_cublas_prefill_sliced` serves hd=512 at S-matrix overflow in
  workspace-sized q-row slices (FP32-S accuracy preserved) — measured
  3.4–3.9× faster than the whole-chunk fused kernel at Skv 8k/16k — and
  `max_safe_prefill_chunk` no longer clamps the global chunk size on
  fused-servable heterogeneous models (Gemma-4 keeps full 2048-row chunks
  at any context; was ~190-row chunks at 64k). Gemma-4-12B pp16384 +5.3%
  end-to-end; audit trail in `docs/audit/gemma4_attn_routing_2026_07_16/`.
- **`gemm.fp8_attn_proj` — FP8 decode sidecar for full-precision attention
  projections** (#984, PR #990): per-row-scale FP8 E4M3 copies of BF16/F16
  wq/wk/wv/wo, decode-only (prefill keeps the full-precision source).
  Default "auto" = full q/k/v/o on gpt-oss, whose BF16 dense projections had
  no decode cache and ran as 2 B/elem FP16 GEMVs (33.5% of the decode
  window). gpt-oss-20b decode 349.7 → 391.2 tok/s (+12%), turning the
  llama.cpp b9976 statistical tie into a +13–19% lead. Teacher-forced PPL
  unaffected by construction (nsys-verified); degen_suite 33/33. Modes:
  `auto` / `qo` (q+o only) / `on` / `off`.

### Changed
- **Dependency bumps**: CUTLASS v4.5.3 → v4.6.0 (upstream changes are
  CuTe-DSL/Python-centric; measured perf-neutral on sm_120 decode A/B —
  Qwen3-8B Q8_0 +0.8%, Qwen3-14B NVFP4 −0.5%, within trial spread), then
  v4.6.0 → v4.6.1 (PR #1010: upstream patch release with CuTe-DSL fixes
  only — no changes to the C++ GEMM collectives imp uses) and
  cpp-httplib v0.48.0 → v0.50.1 (picks up three security fixes: multipart
  `Content-Disposition` header injection, CRLF injection in chunked
  trailers, TLS use-after-free in `SSLClient`/WebSocket teardown — imp-server
  uses `SSLClient` for image-URL fetches). Dockerfile ARG defaults re-synced
  with `cmake/imp-deps.cmake` (CUTLASS default had drifted at v4.5.2).
- **`gemm.nvfp4_lm_head` is now `"auto"` with a per-model net rule** (#982,
  PR #990): ON for native BF16/F16 LM heads (+8-16% decode, +2.2% PPL — the
  long-standing documented trade) and for small dense GGUF heads
  (d_model ≤ 4096, where the measured decode win exceeds the PPL cost);
  OFF for larger or MoE GGUF heads (14B/30B-A3B measured net-negative in the
  2026-07-12 parity sweep — those now return to default PPL parity at the
  cost of −1.9%/−3.4% decode). Legacy `true`/`false` values still parse.

### Fixed
- **KV-pool exhaustion at decode is now diagnosable** (PR #1042): the
  reject-newest cancel used to be completely silent and surfaced as a bare
  "internal error" (Gemma-4-12B-NVFP4 at ctx 16384 on a VRAM-clamped
  16384-token FP16-KV pool). The scheduler now logs the exhaustion (block
  numbers + remedies) and warns at admission when a prompt leaves less
  than one KV block of decode headroom; `imp_decode_step` returns
  `IMP_ERROR_CANCELLED` for engine-cancelled requests instead of
  `IMP_ERROR_INTERNAL` (natural FINISH keeps the INTERNAL end-of-stream
  contract).
- **#998 — n-gram spec decode −39% at moderate context on GGUF K-quants**:
  the verify-chunk forward (M = 2..33) took the M>1 prefill dispatch, which
  dequantizes the full quantized source per GEMM — on Qwen3-14B Q6_K a
  verify step cost ~7x a decode step (`dequant_q6k` alone was 52% of the tg
  window at ctx 2048), so speculation lost even at 100% accept and 4
  tok/verify. Verify-chunk GEMMs now read the NVFP4 decode overlay in one
  weight pass per MR≤4 tile (`gemm_nvfp4_batched`, same tiling as the
  batched LM head; kill switch `speculative.verify_nvfp4_gemm`). 14B Q6_K
  tg128 at ctx 2048: 91.9 → 153.2 tok/s (+67%, now ahead of spec-off
  150.6); ctx 4096: 89.7 → 138.0. Qwen3-8B Q8_0 at ctx 2048: 209.8 → 312.6
  (+49% — the single weight pass also beats the M-independent mmq_imma
  path). Real prefills never take the branch; greedy output is
  byte-identical to spec-off and to the previous dequant verify.
- **Prefill CUDA graph replayed stale chunk geometry on continuation chunks**
  (#981): the captured chunk forward bakes `ctx_len`/`q_offset` as host args
  into the attention launches, and the graph was only invalidated on
  (chunk_len, block_count) changes — so chunk 2+ of a multi-chunk prefill
  replayed chunk 1's graph and attended with chunk-1 geometry, silently
  truncating long context (teacher-forced PPL 8.30 → 15.35 past the second
  chunk on Qwen3-4B Q8). The scheduler now captures only offset-0 chunks
  (whose geometry repeats across requests); continuation chunks run eager.
  Exposure was narrow — quantized-KV appends abort the capture (see below)
  and the NVFP4 M>1 dequant fallback marks most GGUF models uncapturable —
  but any capturable fp16-KV config with prompts past one chunk was silently
  wrong. Regression test: ChunkedPrefillTest.LongContext_Chunk_Invariance_GraphsOn.
- **Quantized-KV prefill chunks no longer attempt graph capture**: the
  dynamic-scale KV append does a D2H absmax sync per chunk, which is illegal
  under capture — every chunk aborted its capture, spamming CUDA errors and
  wasting one full forward per chunk (Qwen3 GGUFs have been on FP8-KV-auto
  since #977, so every >2048-token prompt paid this). F16 KV remains the only
  capturable append path; the rest run eager without the wasted forward.

## [0.19.0] - 2026-07-12

Highlights: first cross-engine PPL-parity measurement (release bar 1) with two
real tokenizer/quant fixes out of it; dense n-gram speculation now wins long
context (#964); FP8 KV auto-on for hint-less Qwen3 GGUFs (+41% at 16k); batched
serving 861 → 1173 tok/s at 16 streams (above the published vLLM reference);
suspend/resume and warm weight cache; competitive re-sweep vs llama.cpp b9976
(dense +42-48%, hybrid +18%, Gemma-4 +21%; gpt-oss now a statistical tie — #984).

### Added
- **Suspend to RAM** (`POST /admin/suspend` / `POST /admin/resume`): park the
  loaded weights in host RAM and free the GPU completely, then resume serving in
  seconds. Only the weights stay warm — sessions and KV do not survive. Models
  whose device buffers are transformed in place (native MXFP4 GGUF, gpt-oss,
  Gemma-4 fused experts) answer a clean 501, and capture is gated on host
  `MemAvailable` (507) rather than driving the host into swap. Config:
  `[suspend]`; C API: `imp_weights_snapshot_*`, `imp_gpu_release`.
- **On-disk warm weight cache** (`[warm_cache] enabled`, default on): the first
  cold load persists its *transformed* uploads (BF16→FP16 conversions, GPU
  dequants, split layouts) to `~/.cache/imp/warm`; later boots mmap them and skip
  the conversion. Raw quant payloads are never duplicated, so the cache is
  near-zero for GGUF and NVFP4 and ~model-size for BF16-dense, where it saves the
  most. Version- and fingerprint-guarded; a stale cache means a normal cold load.
- **`diagnostics.ppl_first` / `diagnostics.ppl_last`**: NLL counting window for
  `imp-cli --perplexity`, matching llama-perplexity's `first = n_ctx/2` for exact
  cross-engine alignment. Recipe and results:
  `docs/audit/ppl_parity_2026_07_12.md`.

### Changed
- **Dense n-gram speculation now WINS on long context** (#964). The verify chunk
  moved off the small-M prefill FA2 tile onto the batched-decode split-K paged
  kernels — 557 → ~65 µs/layer at 16k — and a depth-aware gate
  (`speculative.shallow_draft_ctx`, default 12288) discards 1-token drafts past
  the point where a verify stops paying for itself. Versus speculation off
  (Qwen3-8B Q8_0): **+45% at 512 ctx, +27% at 13312, −0.6% at 15872**, where the
  same three points read −8% / −23% / −62% before. Output stays token-identical
  to plain greedy. `speculative.draft_ctx_cap`, the interim gate, now defaults 0.
- **FP8 KV cache auto-enables on GGUF Qwen3 dense/MoE** (`kv_cache.dtype =
  "auto"`): GGUF exports never declare the FP8 hint the auto policy required, so
  long-context GGUF decode was leaving ~40% on the table (Qwen3-8B Q8_0 at 16k:
  150 → 211 tok/s). A stricter no-hint arch gate admits only families measured
  PPL-neutral at 16k (QWEN3, QWEN3_MOE, ≤0.15%); which families stay excluded and
  why is in `docs/roadmap.md`. Opt out with `kv_cache.dtype = "fp16"`.
- **Concurrent decode at 16 streams: 861 → 1173 tok/s (+36%)** on
  Qwen3-Coder-30B-A3B-FP4 — above the published vLLM reference for this model
  class on a 5090 at per-stream TPOT parity, with single-stream decode unchanged.
  Four nsys-attributed fixes: one pinned strided sampling readback per step
  instead of a pageable one per row (which blocked the host ~850 µs per sequence
  per step), row-parallel top-k/top-p, a vectorized copy kernel for the
  per-MoE-layer residuals (WSL2 WDDM blocks the host ~165 µs per
  `cudaMemcpyAsync`), and a device-cached banned-token list.
- **Pipelined batched decode** (`runtime.decode_pipeline`, default on): at n≥2
  with CUDA graphs the engine keeps one decode step in flight, so host
  bookkeeping, scheduling and SSE delivery overlap GPU compute instead of idling
  it. Coder-30B-FP4 at 16 sustained streaming: 915 → 970 tok/s (+6.0%), TPOT 17.0
  → 16.1 ms; n=1 never pipelines. Uniform-composition runs are bit-identical to
  the per-step path (`EngineIntegrationTest.PipelinedBatchedDecodeMatchesPerStep`).
- **`gemm.nvfp4_lm_head_cutlass` is now default ON** — the batched-decode LM head
  runs as one CUTLASS NVFP4 GEMM, one head weight read per batch instead of
  ceil(n/4); this was the opt-in behind the 1173 tok/s headline. Measured PPL cost
  +1.9-2.1% on MoE/hybrid, +0.2-0.5% on dense (inside run-to-run spread), and
  batch=1 output is bit-identical either way. Set it false for maximum
  batched-serving coherence.
- **FP8 SSM-projection decode sidecar extended to GGUF hybrids**
  (`gemm.fp8_ssm_proj`, default on): the Q8_0-kept GDN projections of UD quants
  were in no decode cache at all and paid a dequant→cuBLAS round-trip per token.
  Qwen3.6-35B-A3B UD-Q4_K_M decode 224.4 → 272.0 tok/s (+21%), now ahead of
  llama.cpp's ~229; PPL +1.8%, a documented trade — `--set
  gemm.fp8_ssm_proj=false` reverts. Sub-8-bit GDN sources are excluded on purpose
  (FP8 would *increase* their decode bytes).
- **Roofline baseline re-pinned** (`cf1b382a_20260711_193211`, config_version 4),
  the old pin predating FA2-hd256 (#932) and the FP8 sidecar (#949/#962). Adds an
  `nvfp4-hybrid` cell — first kernel-level coverage of Qwen3.6-35B — and reaches
  0 unclassified kernels, from 51-63% of the q4k-moe prefill window.

### Fixed
- **Qwen3.5/3.6 GGUF tokenization was non-canonical**: `tokenizer.ggml.pre =
  "qwen35"` fell through to the gpt2 per-char-punct fallback, over-splitting
  symbol runs by +13% tokens on a 95 KB corpus. It now routes to the qwen2
  scanner, with token streams verified identical to `llama-tokenize`. Found by
  the first PPL-parity sweep (`docs/audit/ppl_parity_2026_07_12.md`).
- **The NVFP4-LM-head opt-outs were dead on GGUF checkpoints**: the
  quantized-source cache collector added the head unconditionally, so
  `gemm.nvfp4_lm_head=false` and the GOAL-listed `gemm.nvfp4_lm_head_gdn=false`
  silently did nothing — and that head's quantization is the entire cross-engine
  PPL gap vs llama.cpp (+1.5…+4.8%, model-size-dependent). Defaults are
  byte-identical.
- **Qwen3.6-35B illegal memory access / silent garbage past 16k context** (#963):
  when StreamingLLM auto-enabled, eviction retained a ceil-aligned sliding window
  while the paged decode kernels start reading floor-aligned, so they read a freed
  −1 sentinel block — an IMA on a VRAM-full card, silent out-of-bounds attention
  otherwise. Eviction now keeps one extra boundary block, and the paged loops skip
  negative sentinels outright.
- **gemma-3-12b GGUF decode illegal-memory-access**, the last hard crash in the
  known-issues list: gemma-3's NVFP4 decode cache must be built from an FP16
  companion, and VRAM-budget starvation silently dropped 35 of 49 tensors to the
  from-scratch build that corrupts decode. Those weights now stay on the
  dequant-at-decode path — a bandwidth loss on the uncached fraction, never
  garbage. Pinned by `tests/test_nvfp4_gemv_gemma3_dims.cu`.
- **KV floor now covers the full advertised context on cheap-KV models** (#963
  follow-up): the old min(16384, 4×max_seq_len) floor gave a hybrid a 16.4k pool
  that a 16k prompt fills to 94%, tripping the StreamingLLM valve on a request
  that fits outright. When full coverage plus 12.5% headroom costs ≤1 GiB the
  floor now takes it; expensive-KV models keep the old floor.
- **Greedy request-order independence in default mode** — the documented "30B
  NVFP4 MoE nondeterministic at temp=0" flipper: the first request of a process
  ran one decode step on a different kernel mix than every later one, flipping
  greedy output on near-tie logits. The decode graph pool is now pre-armed in
  warmup and `runtime.warmup` defaults true (+2-4 s init on a 30B). Verified 3
  fresh processes × 12 requests = 36/36 byte-identical. See `docs/determinism.md`.

### Removed
- **`gemm.nvfp4_ssm_proj`** (the 2026-05-30 opt-in forcing GGUF-hybrid GDN
  projections into the NVFP4 decode cache): bit-rotted in the tier refactors — 71
  tok/s against its original 248 on Qwen3.6-35B Q4_K_M — and superseded by the
  GGUF branch of `gemm.fp8_ssm_proj`, which is both faster and quality-safer than
  4-bit into the recurrent scan. Stale `imp.conf` entries now log the standard
  unknown-key warning.

## [0.18.1] - 2026-07-10

### Changed
- **The per-token SSE streaming loop is now a single shared driver**
  (`tools/imp-server/stream_driver.cpp`, #951): the outer token loop —
  disconnect/timeout/keepalive, structural-stop filtering, Harmony/Gemma
  channel filters, reasoning demux, streaming tool-call demux, stop-sequence
  holdback, metrics/JSONL accounting — was hand-copied per dialect
  (chat/messages/responses, ~600 LOC each) and kept drifting (#892, #941).
  The three handlers are now thin wire-format adapters; the OpenAI-params →
  `imp::Request` mapping is likewise a single `build_imp_request_()` instead
  of four hand copies. Net −732 LOC.
- Internal single-sourcing follow-ups from structural audit #6 (#952): the
  SSM conv-channel count is a derived `ModelConfig::ssm_conv_channels()`
  (was hand-derived at 9 sites), and the native-NVFP4 cache-demand tensor
  scan runs once per engine init instead of four times.

### Fixed
- Drift bugs surfaced by the streaming-loop unification (#951): `/v1/responses`
  emitted an empty `function_call_arguments.delta` for buffered (non-JSON
  layout) tool calls (use-after-move; arguments only appeared in the `.done`
  event) and reported `reasoning_tokens: 0` regardless of actual reasoning;
  `/v1/messages` streams were missing the `imp_inter_token_seconds` metric,
  did not record streamed tool-call arguments on a mid-args cutoff, and left
  the engine request running when a keepalive write failed (now cancelled
  like a disconnect); `/v1/messages` and `/v1/responses` requests never set
  `started_in_think`/`in_think_block`, so think-budget enforcement did not
  engage on those routes, and Predicted-Outputs prediction tokens were
  dropped.

### Fixed
- **The decode CUDA graph now re-derives its launch topology when the context
  high-water mark grows — a long-prompt request after short ones no longer
  wedges the engine with an illegal memory access** (#948). The decode-
  attention launch topology (split-K `num_splits`, GQA-vs-split-K kernel
  choice) derives from `max_context_len` on the host and is baked into the
  captured graph. The intended re-capture trigger — the pow2 `max_blocks`
  bucket — never fires because the decode batch pool pads
  `max_blocks_per_seq` to the pool stride, so a graph captured at ctx≈35
  replayed a stale topology at ctx≈2400 and faulted; the engine then never
  recovered (every subsequent request returned 0 tokens after 300 s,
  `/health` unhealthy). New trigger: pow2-bucketed context high-water mark
  (monotonic, ~log2(max ctx) re-captures per process; shrink replays are
  served by the large-ctx capture via the split-K empty-split sentinels).
  The full degeneration suite now passes against the Qwen3.6-35B server for
  the first time (33 checks, 38 s — previously wedged at ~14 requests).
  Also hardened `resize_workspace` against running while the decode
  workspace is the active alias (would free the decode-shared buffer and
  leave it dangling — latent, found during the same investigation).

### Added
- **`gemm.fp8_ssm_proj` (default ON): FP8 E4M3 decode sidecar for the
  native-precision GDN/Mamba in/out projections on NVFP4 hybrids — Qwen3.6-35B
  decode +19%** (tg256 268.6 → 320.3 spec-off, 261 → 308 with default
  speculation), PPL flat with per-row scales (one per-tensor scale over the
  heterogeneous fused GDN input pack cost +4% PPL; per-row is 8.021 → 8.012).
  These projections were the single largest decode slice (34.6% of GPU time as
  FP16 GEMVs) because the producer recipes exclude them from NVFP4 and 4-bit
  GEMV on their wide shapes *regresses* (measured 2026-05-30). FP8 halves the
  bytes with byte-aligned loads instead. Prefill and M>1 verify chunks keep the
  FP16 source; only the M=1 decode GEMV (`gemv_fp8_rowscale`) takes the sidecar.
  Nemotron-3-Nano PPL flat (4.184 → 4.117); no-op on GGUF hybrids (see
  `nvfp4_ssm_proj`) and on checkpoints whose SSM projections are already NVFP4
  (Qwen3.6-27B).

### Fixed
- **Streaming `/v1/responses` requests now update server metrics and send SSE
  keepalives** (#941). The responses token loop never touched `requests_total`,
  the token counters, or the TTFT/duration/inter-token histograms (only
  `requests_cancelled`), and emitted nothing during long prefills — reverse
  proxies could kill the idle stream. Both blocks now match the chat/messages
  streams.
- **The pre-upload KV reserve computed 0 bytes for NVFP4/MXFP4_KV cache
  dtypes** (#942): it multiplied by raw `dtype_size()`, which has no case for
  the packed 4-bit KV dtypes (and counts INT4 at twice its packed size, with
  no scale overhead anywhere). The packing- and scale-aware per-block
  calculation is now shared (`kv_block_bytes_per_layer`) between the VRAM
  budget planner, the expert-offload reserve, and the KV init log.
- **`workspace_estimate()` no longer charges the 256 MiB cuBLAS S-matrix on
  FA2-served configs** (#943): the allocator skips that buffer since #932, but
  the estimate still reserved it, holding phantom headroom out of the
  cache/KV planners during weight upload. The gate (`fa2_serves_all_prefill`)
  is now a shared predicate so the two sites cannot drift.

### Changed
- **Directory sweep follow-up**: removed four March-era one-off tools with zero
  references (`tools/prompt-test.sh`, `tools/benchmark.sh`, `tools/chat.sh`,
  `tools/download-models.sh` — superseded by `imp-cli --bench`, `bench/`,
  `scripts/verify.sh`, and the server batteries), and dropped the dead
  `CUDA_ARCHITECTURES` build-arg from `docker-compose.yml` (the Dockerfile has
  no such ARG since the sm_120a-only build; the value was silently ignored).
  Verified live and kept: `bench/` (benchmark-cuda tooling), `monitoring/`
  (compose Prometheus/Grafana stack), the compose deployment itself
  (`docker-entrypoint.sh` translates the container-level `IMP_*` interface to
  live CLI flags), `third_party/stb` (vision image loading), `tools/analysis`,
  `tools/roofline` (append-only tracked history by design).
- **Docs consolidated under `docs/`; repo root cleaned up.** `GOAL.md`,
  `BENCHMARKS.md`, `BENCHMARKING.md` moved to `docs/`; `AUDIT_FILESIZE.md` and
  `PERF_LOG.md` to `docs/audit/`; the stray q4k-MMQ design spec from
  `docs/superpowers/specs/` to `docs/plans/`. Superseded point-in-time reports
  removed (full text in git history, indexed in `docs/archive/README.md`):
  root `AUDIT.md`, `AUDIT/agentic_server_scout.md`, `BATTERY_REPORT.md`,
  `tests/TEST_AUDIT.md`, and the stale Qwen3.5-27B host-dequant design memo.
  All references (README, CLAUDE.md, AGENTS.md, skills, CI, scripts, code
  comments) updated. The root now holds only README / CHANGELOG / CONTRIBUTING
  / AGENTS.md / CLAUDE.md.
- **Structural audit #6** (`docs/audit/structural_debt_2026_07_10.md`): swept the
  ~40 PRs since audit #5. Confirmed findings filed as #941 (responses-stream
  metrics/keepalive drift), #942 (pre-upload KV reserve computes 0 bytes for
  4-bit KV dtypes), #943 (`workspace_estimate()` still reserves the S-matrix on
  FA2-served configs). Cleanup landed alongside: dead `MemAccount::total_vram_`
  and unused `KVCacheManager::swa_window()/swa_slack()` getters removed; stale
  "fa2_hd256 default off" comments (#932 flipped it on) and hd=256 routing docs
  (`attention-dispatch.md`, `architecture.md`) refreshed; `process_diag`
  fallback defaults for `fa2_f16acc`/`fa2_pv_f16acc`/`fa2_hd256` aligned with
  the config defaults per the header's own contract (the crosspath test now
  pins the f32-score-chain explicitly instead of inheriting it).

### Fixed
- **An explicit `enable_thinking: true` request is now honored on templates
  that default the switch to a closed block** (e.g. Qwen3.5-4B). The chat
  render only ever stamped `enable_thinking=false` (suppression); an explicit
  *true* was silently dropped, so the template kept rendering its pre-closed
  `<think>\n\n</think>\n\n` block and the model answered directly instead of
  reasoning. The server now stamps `enable_thinking=true` into the render for
  an explicit request, so such templates open the think block; the reasoning
  then separates into `reasoning_content` as expected. Default (no explicit
  request) still leaves the variable undefined so each template author's own
  default wins. The thinking-state decision is documented as one pipeline
  (intent → render → rendered-prompt ground truth).

## [0.18.0] - 2026-07-09

### Changed
- **HD=256 FA2 default-on + FP8-KV deterministic forcing lifted (stage 3).**
  `attention.fa2_hd256` now defaults to true: head_dim=256 models (Qwen3.6
  hybrids, gemma-3-class) route prefill through the register-resident f16-QK
  FA2 kernel by default. The single-shot prefill path gains the chunked
  path's uniform-shape refinement — GDN/Mamba2 hybrids (one distinct
  attention shape) now take FA2 single-shot at any head_dim instead of
  unconditionally falling to cuBLAS; learned sinks (gpt-oss) and
  heterogeneous per-layer shapes (gemma-4) keep cuBLAS. With FA2 serving
  every attention call on uniform hd=128/256 models, the FP8-KV
  deterministic-cuBLAS forcing (`engine_init_resolver`) is skipped for
  hd=256 too — FP8 KV on Qwen3.6-35B no longer drags in the model-wide
  deterministic algo pinning. Validation battery in PR.

### Added
- **Stage-1 HD=256 FA2 port (`attention.fa2_hd256`, default off).** The
  register-resident FA2 prefill kernel gains head_dim=256 instances (fp16-qk,
  Bq=64/Bkv=64/TWOSLOT — the double-buffer would need 135 KB smem vs the 99 KB
  sm_120 opt-in; pv-f16 variant fits in 228 regs with zero spills) and the
  chunked-prefill router accepts hd=256 behind the flag. Measured on
  Qwen3.6-35B-A3B-NVFP4 (hd=256, GDN hybrid): kernel 4.3x vs the SMEM-tiled
  WMMA FMHA; e2e prefill +10.6% pp4096 / +24.8% pp8192; teacher-forced PPL
  10.44 vs 10.58 baseline (no quality loss). Opt-in until the split-D stage-2
  port restores hd=128-class occupancy and the route is validated across the
  gemma-class models.

### Fixed
- **Qwen3.5-4B-mxfp4 (and any closed-block reasoning template) now returns its
  answer in `content`, not `reasoning_content` (#934 follow-up).** The server
  defaulted thinking ON whenever the chat template merely *mentioned*
  `enable_thinking`, but Qwen3.5-4B's template defaults it to a pre-*closed*
  empty block `<think>\n\n</think>\n\n` (the model answers directly). Starting
  the reasoning splitter in REASONING then trapped the whole answer in
  `reasoning_content` with empty user-visible `content` on every OpenAI /
  Anthropic / Responses endpoint. `enable_thinking` is now reconciled against
  what the template *actually* rendered into the prompt tail (open prefix →
  thinking on; pre-closed block → off), via a pure, unit-tested helper. Genuine
  reasoning models (open-`<think>` prefix, e.g. Qwen3-14B) are unaffected.
- **Qwen3.5-4B-mxfp4 decode no longer aborts CUDA-graph capture with cuBLAS
  status-14.** The GDN projection GEMM (FP16, N=32) was rejected by the
  capture-safe sm_120 WMMA kernel's `N < BN` guard and fell through to
  cuBLASLt, which fails under stream capture on sm_120 → the whole decode graph
  fell back to per-step. The kernel already masks partial N/M tiles in both the
  load (cp.async src-size=0 zero-fill) and the store (`g_col >= N`), so the
  guard was needlessly conservative; narrow N is now accepted and capture
  succeeds. Correctness pinned by a new N=32 test.
- **MXFP4-GDN hybrids no longer serve `!!!…` garbage when VRAM is tight
  (#934).** On GDN hybrids the native MXFP4 GEMV is unavailable, so decode
  requires the FP16 dequant cache (~4x the raw MXFP4 bytes) resident alongside
  the weights — but the VRAM budget never charged for it. At the server's
  default (large) `max_seq_len` the KV pool ate the headroom, the FP16 fallback
  then failed to allocate, and decode ran against weights with no usable kernel
  → uniform logits → token-0 (`!`) garbage. `compute_vram_budget` now reserves
  the MXFP4→FP16 fallback up front (mirroring the SWA/SSM/NVFP4/Q4_K reserves)
  so the KV pool sizes down to leave room, and the fallback path in
  `pre_dequant_phase3_cutlass` now throws a legible out-of-VRAM error at load
  instead of silently skipping the alloc and serving garbage. Verified on
  Qwen3.5-4B-mxfp4 via imp-server at default context: coherent output restored.
- **Deterministic cuBLAS GEMM now validates its algo choice (intermittent
  status-14 garbage on FP8-KV / head_dim=256 models).** `runtime.deterministic_gemm`
  (also force-enabled for FP8 KV on non-FA2 / head_dim!=128 models like
  Qwen3.6-35B-A3B-NVFP4) picked cuBLASLt's top heuristic candidate blindly,
  skipping the per-candidate runtime warmup the timing path uses precisely
  because "the heuristic can return algos that fail at runtime on sm_120". A
  bad `results[0]` for some FFN shapes (e.g. M=608 K=2048 N=8192) then failed
  with `CUBLAS_STATUS_INTERNAL_ERROR` (14) and the `void` GEMM wrapper
  continued with a garbage buffer → repeated-token gibberish. The deterministic
  path now warmup-probes candidates in (stable) heuristic order and picks the
  first that survives — reproducible AND valid. Determinism preserved (dense Q8
  greedy output byte-identical run-to-run).
- **A totally-failed GEMM is now fatal instead of silent garbage.** When both
  cublasLtMatmul (after algo reselect) and the cublasGemmEx fallback fail in the
  generic FP16/INT path, `gemm` throws (translated to ImpError at the API
  boundary; aborts CUDA-graph capture → per-step fallback) rather than leaving
  an uninitialised output buffer for the forward pass to turn into gibberish.

## [0.17.3] - 2026-07-09

Native-NVFP4 serving fix release: the mandatory decode caches are now
physically reserved before the elastic VRAM consumers, so large NVFP4-prequant
MoE models (Qwen3.6-35B-A3B-NVFP4) reach full decode-cache coverage + captured
decode graphs under pure default config. No perf change for GGUF/FP16 models
(budget arithmetic byte-identical, pinned by test); perf baseline untouched.

### Fixed
- **VRAM ordering: mandatory NVFP4 decode caches are now reserved before
  workspaces and the KV pool.** For native-NVFP4 models the CUTLASS SfAtom SF
  slab (~2 GB) and the nvfp4_moe decode cache were built last from already-
  starved free VRAM; partial caches abort decode CUDA-graph capture (one
  uncovered MoE layer -> host-args path -> capture throws), pinning
  Qwen3.6-35B-A3B-NVFP4 at 26-40 tok/s under default config. A balloon
  allocation right after weight upload holds the exact demand (new
  `compute_native_cache_demand`, sized with `cutlass_nvfp4_sf_size` and now
  including the GDN/SSM projections the old estimate missed) until the cache
  build, and the phase-3 budgets are floored at the balloon-backed guarantee
  (live `cudaMemGetInfo` lags async frees). Default config now reaches full
  caches + captured decode graph: 247-249 tok/s with a 138k-token KV pool
  (was 26-40). Non-prequant (GGUF/FP16) budget arithmetic is unchanged
  (pinned by test); escape hatch `[vram] native_cache_reserve` (default on).
  A new post-build coverage log states FULL/PARTIAL cache status and remedies.
  (#926)
- **Loud WARN when the KV pool collapses below its token floor (#927):** with
  an oversized `max_batch_size` the batch-scaled workspaces can shrink the KV
  pool to the 16-block minimum; longer requests were silently cancelled at
  admission while `/v1/models` kept advertising the full context. The budget
  planner now warns with the real pool size and remedies. Log-only.

## [0.17.2] - 2026-07-08

Small server-compatibility release: the served context window is now
discoverable through the three field conventions OpenAI-compatible clients
already probe, so they can auto-detect it instead of keeping a hard-coded
table. Server-only, no functional change to inference; perf baseline untouched.

### Added
- **Context-window auto-detection across the three live conventions (#921):**
  `GET /v1/models` now carries the context length as vLLM's `max_model_len` and
  llama.cpp's `meta.n_ctx_train` on the model object (plus `created` for OpenAI
  compliance); new `GET /props` (llama.cpp shape — `n_ctx`) and `GET /info`
  (TGI shape — `max_total_tokens` / `max_input_tokens`). All three report the
  same engine-detected `max_seq_len`.

## [0.17.1] - 2026-07-08

Follow-up to the 0.17.0 C++23 toolchain bump: now that the whole tree builds as
C++23, this release adopts the C++23 language idioms that genuinely fit the
codebase. Behavior-neutral — identical values, same accessors, same functors;
decode verified coherent and unchanged, perf baseline untouched. `std::expected`,
`std::mdspan`, `std::print`, and `[[assume]]` were deliberately left out (the
throw-based error model, device-side nvcc limits, and — for `[[assume]]` —
proven-inert codegen on the NVFP4 GEMV decode path).

### Changed
- **Adopted C++23 idioms across the tree (#919), no functional change:**
  `std::to_underlying` at ~67 real enum-to-underlying cast sites (with `<utility>`
  added per translation unit — `to_underlying` is a hard compile error on
  non-enums, so the build itself confirms every site); `deducing this` collapses
  four duplicated const/non-const accessor pairs into one overload each
  (`Model::layer`, `SchemaConstrain::top`, jinja `Value::as_object`,
  `WeightRegistry::handle`); `static operator()` on six stateless functors (two
  host hash functors plus four `__device__` activation functors — nvcc 13.3
  accepts `static operator()` in device code).

### Tests
- Cover `format_tool_response` + `reconstruct_tool_call_output` (#914).

## [0.17.0] - 2026-07-08

Toolchain-modernization release: the engine now builds as **C++23** on an
**Ubuntu 26.04 / GCC 15.2 / CUDA 13.3** base (was C++20 / Ubuntu 24.04 / GCC 13).
The standard bump changes no default-path behavior and no perf (decode verified
neutral). Ships alongside two FP8 tile decode-attention kernels (large
long-context wins), the MLA/MTP RoPE correctness fixes, an async-mempool teardown
fix, and the server-hardening / config / VRAM cleanups from the 2026-07-07
structural audit.

### Changed
- **C++ standard raised to C++23** (host + CUDA). CMake's NVIDIA-CUDA module has no
  CUDA23 dialect flag, so the build teaches it `-std=c++23` explicitly (shim to drop
  once CMake ships a native mapping). No source changes were required (#916).
- **Build toolchain → Ubuntu 26.04 / GCC 15.2** (CUDA stays 13.3); the Dockerfile and
  both CI compile containers moved, which catches the GCC-15 missing-include class in
  CI. **Note:** nvcc silently drops `-std=c++23` on a host compiler older than GCC 14,
  so dev/profiling images must be on this base — the `impdev:ncu` recipe is now
  committed at `tools/Dockerfile.ncu` (#907).
- Retired the legacy config surface: env-var seeding (down to `IMP_DETERMINISTIC` +
  `IMP_FMHA_FA2`), turboquant aliases, and dead flags (#879); `imp.conf.example`,
  `--help`, and config comments synced to parser reality (#878).
- VRAM-layer audit: dead modules removed, one reserve floor, honest budget logs (#877).
- Tokenizer: dropped the duplicated JSON parser in favor of shared `model/json_util` (#887).
- Analysis/roofline tooling: PTX survey scripts track the latest CUDA toolkit (#908);
  Python 3.14 plot env + roofline baseline re-pin (#904).

### Added
- **FP8 tile decode-attention kernels.** Token-tiled FP8 split-K decode (K and V staged
  in one cp.async group) — long-context decode **+51%** (#899); a GQA-batched variant
  reads each KV head once across the warp group for a further **+14%** (#900).

### Fixed
- **MLA (DeepSeek-V2/V3) YaRN rope-mscale**: the RoPE cos/sin were scaled by
  `yarn_get_mscale(factor, mscale_all_dim)` (=1.261 for V2-Lite) instead of the
  HF ratio `yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)`
  (=1.0 when the two coincide, as in V2-Lite). imp was inflating the rotary
  embedding by 1.261×; the error compounds with position, so teacher-forced PPL
  degraded with sequence length. `mscale` and `mscale_all_dim` are now loaded
  separately: the softmax attention scale keeps `mscale_all_dim²` (unchanged),
  the rope factor uses the ratio. Same-corpus PPL vs HF bf16 on DeepSeek-V2-Lite:
  534-tok **+24.4% → +2.75%** (imp 7.78→6.43, HF 6.25); 196-tok +5.0% → +0.8%.
  The residual ~1-3% is F16-vs-bf16 compute precision. Applies to both
  DeepSeek-V2-Lite and DeepSeek-Coder-V2-Lite (same config); generalizes
  correctly to V3 (where the two mscales differ) (#880).
- **MTP draft-head mrope** now applies YaRN / rope-scaling (was plain NeoX RoPE), so the
  drafter no longer drifts from the verifier on rope-scaled models — speculative
  acceptance no longer degrades with position (#913).
- **Async mempool** is now trimmed on `Model` teardown, not only at the C-API boundary,
  releasing device memory between in-process model swaps (#915).
- **Capture-poisoned engine wedge**: a failed CUDA-graph capture no longer wedges the
  engine; plus planner-driven KV-pool sizing (#874, #875).
- **GCC 15 build**: added the `<algorithm>` / `<numeric>` includes that libstdc++15 no
  longer pulls in transitively (#903, #906).
- No-GPU audit sweep #888–#894: server admission control / observability / `/health`
  locking, embeddings, API strictness, and tool-call suppression, plus dead-code and
  doc/comment drift (#901).

## [0.16.2] - 2026-07-04

FP4-attention research batch: the #846 program (SageAttention3 → ThriftAttention →
KV-append-quant) is closed end-to-end with measurements on every branch. All new
knobs are research scaffolds and ship **default-off**; no default-path behavior
changes.

### Added
- `attention.mxfp4_promote_budget` (default 0): ThriftAttention-style outlier
  block promotion (arXiv 2605.23081) in the MXFP4 FMHA — per q-tile, the
  top-scoring fraction of visible KV tiles (block-mean score Q̄·K̄ᵀ, sink +
  diagonal force-included) computes exactly instead of FP4. Takes the FP4
  attention quality gate from +9.9%/+4.4% NLL (@1k/9.3k, prose) to −0.6%/−0.2%
  at 5% budget (#870).
- `attention.mxfp4_paged_kv` (default off): chunked-prefill continuation reads
  K/V directly from the paged NVFP4 KV cache (quantization paid once at
  append; no gather→FP16 pass, no in-kernel quant); the current chunk stays
  fresh FP16 via force-promoted tiles. Quality gate passes (+0.34% NLL @9.3k
  at 5% budget); kernel-level perf refuted — quality-validated scaffold (#872).

### Changed
- Docs: MISSION_JOURNAL records the full measurement chain — FP4-MMA delivers
  as advertised (tensor pipe 40.8%→2.2%) but in-kernel K quantization costs
  3.34× FA2's instruction budget; the smem-materializing kernel is
  latency-bound (pure paged-MMA floor 8.5× FA2); quantizing the RECENCY window
  is the entire quality cost of FP4 KV storage (stored-FP4 current chunk
  +3.7–5.4% NLL even with exact compute, stored-FP4 past ≈ free); nvfp4 KV
  costs ~+0.8% NLL in the decode-recency regime (FP8 auto-default clean) —
  nvfp4-KV quality claims need a small-chunk (≤64) PPL arm (#871, #872).

## [0.16.1] - 2026-07-04

Spec-verify economics batch (#847 ladder): chunk-path overhaul — default
suffix-speculation on Qwen3.6-27B prompt-echo 81 → 131 tok/s (**+61% vs
v0.16.0**), 35B-A3B +10-15% — plus the SuffixDecoding drafter, MTP verify
activation (opt-in), hybrid verify capture, nomic-bert embeddings, and the
NVFP4-attention research spike (#846, refuted, knobs default-off).

### Added
- **Encoder/embedding-model support** (#836) — `nomic-bert` GGUF checkpoints
  (nomic-embed-text-v1.5) load into a dedicated encoder path: bidirectional
  no-KV forward (post-LN with bias, rotary, SwiGLU), BERT WordPiece
  tokenizer (llama.cpp GGUFs store it in SPM convention: word-initial `▁`,
  bare continuations), true LayerNorm-with-bias kernel, mean pooling + L2
  on device. `/v1/embeddings` serves it with [CLS]/[SEP] framing.
  HF-oracle-verified: cos(imp, HF trust_remote_code) ≥ 0.999 on Q8_0.
  Classic BERT/bge/e5 (learned positions, CLS pooling) stay rejected.

- **SuffixDecoding-style suffix drafter** (#848) — per-request suffix index
  over prompt + generated output with frequency-voted continuations and
  adaptive draft length replaces plain n-gram prompt-lookup as the default
  draft source (`speculative.suffix`, n-gram matcher stays as fallback).
- **Speculative decoding on hybrid (GDN/SSM) models** — `speculative.hybrid`
  (default on; imp-cli `--bench` pins it off): the verify chunk snapshots the
  committed recurrent-state slab and, on partial acceptance, restores it and
  re-forwards the accepted prefix, so suffix/n-gram speculation now engages on
  Qwen3.5/3.6 and Nemotron-H hybrids. Measured (greedy, temp 0): Nemotron-3-
  Nano-30B code-edit +60% tg, Qwen3.6-35B code-edit +18%, Qwen3.6-27B
  prompt-echo prose +156%; draft-poor prompts are unchanged (miss-burst
  hybrid). Token-lossless verified on Qwen3.5-4B (#847 enabler).
- **MTP verify activation** (#847) — the trained MTP head now feeds the
  verify loop as a draft source when the suffix matcher has no match
  (`--mtp-spec-decode <k>` / `speculative.mtp_k`, default off). Loads both
  sidecar MoE heads (Qwen3.6-35B `model_mtp.safetensors`) and embedded dense
  heads (Qwen3.6-27B `mtp.*` in the main shard, new). Chunked-prefill-capable
  MTP cache feed with DeepSeek-aligned pairing and feed-only forwards (no
  lm_head), multi-turn prefix resume, and an economics guard that dooms MTP
  drafting per request when average emitted/verify cannot beat the async
  loop (measured: accept 44-91% but net-negative on current verify
  economics — see #847 for follow-ups).
- **MTP chain lm_head via the NVFP4 decode cache** (#847 lever 3) — the MTP
  chain's per-draft full-vocab logits GEMV reads the NVFP4 LM-head decode
  cache when one exists (~3.5x less HBM traffic than the raw FP16 weight:
  2.5 GB → 0.7 GB per drafted token on Qwen3.6-27B's 248k vocab).
  `speculative.mtp_nvfp4_head` (default on) is the kill switch; draft-only
  precision, verification stays lossless. The MTP economics-guard threshold
  is now configurable (`speculative.mtp_econ_min_emit`, default 4.0,
  0 disables) since the break-even moves with chain/verify costs — note a
  chain of k emits at most k+1 tokens per verify, so k=2 cannot pass the
  default threshold and dooms by construction.
- **Graph-captured verify chunk** (#856, foundation #855; hybrid extension
  #859/#861) — per-(bucket × KV-tier) CUDA graphs replace the eager verify
  forward: the chunk reads its real KV length from device (`d_kv_len`), so a
  captured graph replays correctly as context grows. Launch-pacing win on
  attention-only models (Coder-30B echo +65%, Q8 +7%); Mamba2/GDN hybrids
  capture too (slot-keyed graphs, device-length conv-tail/scan updates) but
  measure ±0 — scan-dominated, not launch-bound. `speculative.capture`
  default on.
- **Opt-in schema jump-ahead** (#849, idea #844) — char-level FSM probe
  (`forced_text`) drafts structurally-forced spans as teacher-forced chunks
  in the constrained pipeline; every emitted token is still masked-sampled
  from its true logits row (exact for greedy and sampling). Default OFF
  (`constrained.jump_ahead`): measured net-negative (−11% on 14B-NVFP4,
  re-measured post-chunk-path-fixes) — the model picks context-dependent
  tokenization splits the canonical draft misses. #844 closed; kept as
  scaffold.
- **NVFP4-attention research knobs** (#868, idea #846 — refuted) —
  `attention.mxfp4_blockscale` (per-16-element UE4M3 scales in the
  mxf4nvf4.block_scale MMA), `mxfp4_ksmooth` (K channel-mean smoothing),
  `mxfp4_pv_fp4` (FP4 P·V with two-level P scaling). Per-16 blockscale
  rescues FP4-QK from the catastrophic per-row failure mode (PPL 31546 →
  5.90 on Qwen3-14B-NVFP4 @199 tok) but the residual noise compounds with
  context (+10% NLL @9k, full recipe) — all three default OFF, documented
  as research/diagnostic in imp.conf.example.

### Changed
- **Persistent K/V gather scratch for the eager chunked path** — spec-verify
  re-enters the chunked attention per layer per verify; the per-call
  `cudaMallocAsync`/`FreeAsync` pair (~140 allocs/verify on hybrids) is
  replaced by a grow-only executor-owned scratch (64 MiB steps, per-call
  fallback if the grow fails). Small win inside run-to-run noise
  (best 27B MTP-only trial 56.4 ms/verify / 49.7 tok/s) and removes the
  acknowledged hot-loop-malloc exception in the chunk gather.
- **Small hd≠128 verify/boundary chunks prefer the tiled FMHA over cuBLAS**
  — cuBLAS re-runs its per-new-shape algo selection on every call (100 MiB
  workspace memset + candidate benchmark + blocking event sync); spec-verify
  chunks grow `ctx_len` every step, so hd=256 models paid ~93 such trios per
  verify (~12-15 ms of churn, nsys timeline on Qwen3.6-27B MTP-only). Chunks
  with n ≤ 32 and hd ≠ 128 now route to the tiled FMHA (shape-stateless,
  PPL-identical per #511) inside its correctness domain; learned sinks and
  heterogeneous shapes keep cuBLAS. Measured: 27B MTP-only verify 78 → 59
  ms/verify, 34 → 44-46 tok/s (+31%).
- **Small-M NVFP4 GEMM: batched GEMV replaces the dequant fallback** — for
  M ≤ 16 (spec-verify chunks of drafts+1, short/boundary prefills),
  `gemm_nvfp4` now runs the batched-M K-parallel GEMV (weight read once per
  MR=4 tile at 0.25x FP16 bytes) instead of dequantizing the whole weight to
  FP16 and calling cuBLAS (~2.25x FP16 bytes, re-paid EVERY chunk). nsys on
  Qwen3.6-27B MTP-only verify: `dequantize_nvfp4_kernel` drops from 48% to
  4% of GPU time; ms/verify k=2 ~77 → ~60, k=4 ~87 → ~76; Coder-30B echo
  verify unregressed. `beta != 0`, non-F16 tensors, and the
  `nvfp4-force-dequant` bisect flag keep the fallback.
- **Device-side MTP draft chain** (dense MTP heads) — chain step i's argmax
  lands in a device slot and feeds step i+1's embedding lookup on-device;
  one D2H + sync drains the whole chain instead of one host round-trip per
  drafted token (process StreamSynchronize count roughly halved on an
  MTP-only run; e2e-neutral today — the verify wall is elsewhere — but
  required groundwork for capturing the chain into a CUDA graph). MoE MTP
  heads keep the host loop (expert routing needs a per-step D2H). Also:
  persistent argmax scratch replaces a per-draft cudaMallocAsync/Free pair.
- **Batched verify/eval LM heads** (#854, #857) — the spec-verify logits
  GEMV over drafts+1 rows reads each weight tile once per 4-row batch
  instead of once per row ([1,V]-shape root cause; Coder echo +50%), and the
  dp4a LM-head path gets the same one-weight-pass-per-batch treatment.

### Fixed
- **Non-gated NVFP4 MoE da_cache never built** (#860 → #861) — `!empty()`
  id checks vs the loader's pre-sized all-invalid id vectors meant non-gated
  (RELU²) models silently took the per-call H2D fallback, whose
  `cudaMemcpyAsync` from stack vectors recorded into verify graphs replays
  from dead stack addresses: nondeterministic garbage B pointers /
  `misaligned address`. With the cache built, hybrid verify graphs record +
  replay cleanly (Nemotron-3-Nano 23/23, deterministic).
- **MoE host-args launches + NVFP4 capture-refusal fail loud under stream
  capture** (#858, #859) — the M>1 dequant fallback's silent return and the
  host-`expert_offsets` D2H + unchecked sync both recorded graphs with
  missing/uninitialized work (`misaligned address` at replay, `<unk>`
  output). Both sites now throw under capture (clean eager fallback);
  dequant workspace pre-alloc is capped instead of all-or-nothing. Also a
  genuine eager bug: the hybrid conv tail wrote zeros instead of shifting
  the previous state on chunks shorter than `conv_kernel` — Qwen3.5-4B spec
  on==off now byte-identical.
- **Chunked-prefill q_offset + fully-masked-row guard in the opt-in MXFP4
  FMHA** (#868) — continuation chunks masked with local row indices (wrong
  causal/SWA masks past chunk 1) and fully-masked rows could poison the
  online-softmax denominator; both now mirror the FA2/FP16 kernels.
- **Schema keys reject backslash escapes** (#850 → #851) — `sim_advance`
  OBJECT_KEY accepted `\` and swallowed it, so the emitted text carried the
  escape while the FSM matched the raw key (`{"\number_of…": …}`-class
  corruption).
- **tools + json_schema preamble slack raised** (#840 → #842) — the schema
  mask no longer fires mid-deliberation on tool-call requests.

## [0.16.0] - 2026-07-02

Multi-server-per-GPU (hard VRAM budget) + load/teardown robustness.

### Added
- **Hard per-process VRAM budget** — `--vram-budget <mb>` (imp-server + imp-cli),
  `[runtime] vram_budget_mb` in imp.conf, and the previously-inert C-API
  `ImpConfig.vram_budget_mb`: every sizing decision (weight caches, KV clamp,
  expert offload, workspaces, upload gates — all 19 sites) sees a virtual GPU
  of the given size, so multiple imp-server processes can share one card.
  Baseline-delta semantics: a co-tenant's pre-existing usage never counts
  against this process's budget; concurrent neighbour allocations shrink the
  view conservatively. Verified with two simultaneously-started servers
  (9000 + 8000 MiB budgets) serving concurrently at 15.9 GiB device total.
  Best-effort cap — leave ~1 GiB real headroom between the sum of budgets and
  the card. Default 0 = uncapped passthrough (#838).

### Fixed
- **Model unload leaked weights-sized VRAM** (~8.3 GiB per Qwen3-8B-Q8_0
  cycle): weights are `cudaMallocAsync`-allocated but were freed with plain
  `cudaFree`, which returns success WITHOUT returning the blocks to the async
  mempool on this stack — the pool double-booked old + new weights on reload
  and `cudaMemPoolTrimTo` could reclaim nothing. Freed with `cudaFreeAsync`
  everywhere (Model teardown + the Phase-3 MoE expert-source drops, whose
  "freed" VRAM was phantom for the same reason). The reload test now probes
  actual re-allocatability (WSL2/WDDM under-reports reclaimed pages in
  `cudaMemGetInfo`) (#834, #837).
- **Encoder-only models are rejected at load on the SafeTensors/HF path too**
  — `is_encoder_only_arch` was case-sensitive, so HF `config.json` class names
  (`NomicBertModel`, `BertModel`, `XLMRobertaModel`, …) slipped past the
  GGUF-only reject and ran a BERT encoder through the causal-LM prefill +
  sampler → CUDA illegal memory access on the first `/v1/embeddings` request.
  Both HF-config paths (`architectures` array + `model_type` fallback) now
  fail loudly at load (#818, #835).
- **Second engine on the same loaded model handle no longer IMAs** — for GGUF
  MXFP4 GDN models the first engine's pre-dequant consumes the model sources
  destructively (in-place MXFP4 raw-block compaction; GDN FP16 fallback
  re-points model tensors at executor-owned memory), so a create→free→create
  cycle rebuilt caches from dangling memory and poisoned the CUDA context.
  `Engine::init` now rejects a second engine on a consumed model with a clear
  "reload the model" error; models whose sources stay intact (dense Q8_0)
  keep supporting create/free/create on one handle (#830, #835).

## [0.15.0] - 2026-07-02

Hybrid (SSM/GDN) agentic serving: prefix caching + concurrent-decode fairness.

### Added
- **Prefix caching for hybrid (SSM/GDN) models via recurrent-state snapshots**
  — reused KV blocks alone cannot skip prefill on a recurrent model (the state
  at the skip boundary would be zero), so the engine now snapshots the
  per-sequence SSM/GDN state slab once per prefill at the largest block-aligned
  prompt position (keyed by the same chained KV block hash as the block cache)
  and restores it on a prefix hit, capping KV reuse at that boundary. Multi-turn
  requests prefill only the delta instead of the whole history: Qwen3.6-35B-A3B-
  NVFP4 per-turn TTFT goes from 1.6→6.7 s (linear in history) to a flat 1.4–1.9 s
  (**3.5× at ~10k tokens of history**, growing with context); Nemotron-3-Nano
  (pure SSM) stays flat at ~0.22 s. `usage.prompt_tokens_details.cached_tokens`
  (and Anthropic `cache_read_input_tokens`) now report hybrid cache hits. New
  `server.recurrent_snapshot_mb` budget (default 256 MiB, pre-allocated at engine
  init and accounted in the expert-offload reserve; `imp-cli --bench` pins it to
  0 so hybrid GGUF baselines are unaffected) (#831).

### Changed
- **Hybrid concurrent decode is now fair (round-robin) instead of head-of-line.**
  The SSM/GDN recurrent scan kernels are single-sequence, so concurrent sessions
  time-slice the decode; previously the batch-1 clamp kept the oldest request
  every step, so a second session produced its first token only after the first
  request finished (measured: 6.6 s of starvation on two concurrent 400-token
  Qwen3.6-35B streams). The slice now rotates every `runtime.hybrid_decode_quantum`
  tokens (default 128), with async graph-loop bursts bounded to the slice
  remainder. `0` restores the old behavior (#831).

### Fixed
- **Prefix-cache reuse no longer counts a chain hole as a reused prefix** (dense
  and hybrid). LRU eviction can drop an early block of a cached chain while later
  blocks survive; reuse now stops at the first non-cached block, so the caller
  never skips prefill over a hole with uncomputed KV (#831).
- **Decode graph pool invalidates when the recurrent state slot changes.** The
  captured decode graph bakes the SSM/GDN state pointers of one slot; overlapping
  request lifetimes (and the new decode rotation) could replay a graph against a
  different sequence's state. The pool now re-captures (graph-exec update) on slot
  change (#831).

## [0.14.0] - 2026-07-02

Agentic-serving batch: multi-turn correctness + speculation + throughput.

### Added
- **OpenAI Predicted Outputs (`prediction`)** — client-supplied predicted
  completion text is tokenized into the n-gram speculative-decode draft corpus
  (never forwarded through the model, so output stays a faithful greedy decode),
  giving guaranteed drafts for code-edit / rewrite workloads.
  `usage.completion_tokens_details.{accepted,rejected}_prediction_tokens` is
  reported on the non-streaming and streaming chat routes; `/v1/completions`
  accepts the string form (#825).
- **Streaming through the conditional-graph decode loop** — streaming requests
  now poll the mapped ring buffer per token instead of taking a per-step host
  round-trip, so SSE delivers each token as the device burst runs (Q8 max inter-
  token gap 197→17 ms, MoE 28→7.6 ms) on the same fast async loop as everything
  else (#822).
- **Agentic API-compliance batch** — streaming tool-call dialects (Gemma-4
  `<|tool_call>`, Qwen3.6 XML) via a shared stream filter, `/v1/messages/count_tokens`,
  OpenAI SSE keepalives, `max_completion_tokens`, stop-cap 4→16, and a clean
  encoder-only-arch (nomic-bert) load reject (#818, #820).
- **`tools/multiturn_bench.py`** — agentic multi-turn replay benchmark (growing
  conversation prefix, per-turn TTFT/decode via SSE, prefix-cache visibility),
  OpenAI-compatible so it runs unchanged against vLLM/llama.cpp (#826, #827).

### Performance
- **n-gram speculation on native-NVFP4 MoE** — the `is_moe` speculation gate is
  relaxed for native-NVFP4 experts (`speculative.moe`, default on; GGUF-MoE stays
  on the async loop where verify re-dequantizes experts): Qwen3-Coder-30B-A3B-FP4
  code-edit **+49-81%** (93% draft acceptance), Modelopt-30B +29%. `imp-cli
  --bench` pins it off so the gated decode baseline stays a raw signal (#824).
- **Serving latency** — decode-aware prefill chunk cap while a decoder is active
  (`runtime.prefill_chunk_decode_cap`, default 1024), admission-aware decode
  bursts so a waiter's prefill isn't starved between bursts, and cross-turn output
  KV reuse. All no-ops for single-stream; greedy byte-identical (#823).

### Fixed
- **Qwen3-Coder-30B multi-turn empty output** — imp's Jinja engine did not strip
  the template file's single trailing newline (unlike Jinja2's
  `keep_trailing_newline=False` that HF/vLLM use). Qwen3-Coder's
  `chat_template.jinja` ends in a newline, so imp rendered the generation prompt
  as `<|im_start|>assistant\n\n`; the extra blank line made the model emit an
  immediate EOS on borderline multi-turn contexts (turns came back empty,
  non-deterministically). `Template::parse()` now strips one trailing
  `\n`/`\r\n`/`\r`. Templates without a trailing newline (Qwen3/Modelopt) were
  unaffected (#828).
- **Native-NVFP4 VRAM budget starved the weight caches** — the budget estimated
  0 bytes for native-NVFP4 checkpoints (qtypes are still wire dtypes at budget
  time), so the KV hard-clamp took all post-weight VRAM and the CUTLASS SfAtom SF
  cache (1.8 GiB on Coder-30B) never built, dropping to dequant fallbacks. Now
  reserved before the KV clamp: **Qwen3-Coder-30B-FP4 server 31.8→300 tok/s**, and
  dense Qwen3-14B-NVFP4 (silently halved by the same bug, masked by CLI batch=1)
  106→209 tok/s (#826).
- **json_schema not enforced under concurrency** — the engine-global
  `ConstraintManager` had per-request state holes (attach gated on batch size 1,
  every-prefill clobber, any-finish reset). Constraint state is now per-request
  with an engine pool; a concurrent schema request that previously invented keys
  is now enforced (#821).
- **Long-context chunked-prefill abort / 32-token chunks** — the engine clamped
  every prefill chunk to `cap²/total` (32-token chunks at 128k on hd≠128) and
  aborted gpt-oss when the clamp hit 0. Offset-aware `max_safe_prefill_chunk()`
  (only when cuBLAS serves) plus `attn_shapes_uniform()` lets uniform-per-layer
  GDN/Mamba hybrids use the O(n) chunked attention paths: **Qwen3.6-35B pp10k +80%,
  Nemotron +115%, gpt-oss pp40k +69%, Gemma-4 +35%**, PPL parity ≤1% (#819).
- **Paged-decode split-K / cluster launch failure** — falls back to single-split
  GQA/MHA instead of erroring (#817).

## [0.13.0] - 2026-06-30

### Added
- **DeepSeek-V2 Multi-head Latent Attention (MLA)** — first MLA architecture in
  imp. Stage A (materialized): the full K/V is reconstructed from the latent at
  projection time so every existing attention / paged-KV / RoPE kernel is reused
  unchanged, correctness-first and verifiable against HF (#802). Phase 3 (opt-in):
  absorbed latent-KV-cache decode that stores only the 512-dim latent + 64-dim
  decoupled-RoPE key in the cache for the long-context VRAM win (#803). Validated
  on DeepSeek-V2-Lite (28 GB, experts host-offloaded on 32 GB → graphs disabled);
  see `docs/supported-models.md`.
- **`gemma4_unified` multimodal Gemma-4 checkpoints** — the unified
  text+vision+audio packaging (`model_type: gemma4_unified`,
  `Gemma4UnifiedForConditionalGeneration`, LM weights under
  `model.language_model.*`, e.g. the dense **Gemma-4-12B-NVFP4**) is now mapped to
  the Gemma-4 arch instead of falling back to the llama path and crashing. The
  text tower is a standard dense Gemma-4; the `model.language_model.` prefix-strip,
  vision/audio-tower skip, and nested `text_config`/`rope_parameters` parsing
  already existed and fire once the arch resolves (#814).
- **NVFP4-quantized GDN (`linear_attn`) projections** — load the quantized
  linear-attention projections for NVFP4 hybrid checkpoints (Qwen3.6-27B-NVFP4-MTP)
  so the GDN path is coherent instead of garbage (#812).
- **SafeTensors/NVFP4 prompt-test battery coverage** — `validate_safetensors.py`
  now sweeps one model per (architecture, NVFP4 source-format) cell (dense Qwen3,
  Phi-4, Gemma-4 dense, Qwen3.6+MTP, both Nemotron variants), not just a handful of
  MoE checkpoints; all loadable archs come back engine-healthy (#815).

### Fixed
- **Gemma-4 garbled after a GDN/SSM model in the same process — GQA paged-decode
  shared-memory opt-in.** Gemma-4's global attention layers use `head_dim=512`
  (sliding layers use 256), whose FP16 paged-decode GQA kernel needs ~64 KiB
  dynamic shared memory and therefore a `cudaFuncSetAttribute` opt-in. That opt-in
  sat behind a one-shot `static bool` guard on a kernel shared by every
  model/head_dim in the process, so after a GDN model (`head_dim=128`) loaded
  first the 64 KiB launch was issued without a valid opt-in →
  `cudaErrorInvalidValue` every decode step on every global layer → degenerate
  output. Re-armed as a high-water-mark opt-in (set on growth, result-checked,
  error-drained). GDN→Gemma-4 in one process now passes the full e2e suite. (A
  zero-production-impact, multi-model-per-process interaction; one model per
  process — server/CLI — was never affected.)
- **Cross-model CUDA-error-leak hardening** — a failed best-effort L2
  access-policy-window (`cudaStreamSetAttribute`) recorded a sticky per-context
  error that could outlive the engine and trip the next model's
  `cudaGetLastError()`-guarded kernels. Drain after every L2-hint set, reset the
  persisting-L2 reservation per model load, and drain at engine teardown so no
  per-context error crosses a model boundary (#815).
- **NVFP4 MoE expert-scale `cudaFree` guard against offset pointers** (#813).
- **MTP draft head — sigmoid (not silu) attn-output gate** — the multi-token-
  prediction attention-output gate used silu where the head expects sigmoid,
  which crippled draft quality. Correcting the gate lifts K=1 acceptance from
  ~10% to 85%+ on Qwen3.6, un-blocking the MTP draft head (#804). (Spec-decode
  *generation* via this head remains parked — the GDN-hybrid MTP model carries
  irreversible recurrent state through verify and the economics are net-negative;
  it needs a non-recurrent MTP model.)
- **gpt-oss GGUF 2^-4 residual rescale** — the official Q8_0-dense
  `gpt-oss-20b-mxfp4.gguf` produced garbage (PPL 2739, degenerate decode) while
  the bf16-dense GGUF and SafeTensors were fine. The gpt-oss residual-stream
  rescale (Wo + biases) applied ×2^-4 by subtracting 4 from the fp16/bf16 biased
  exponent — wrong for small scales: denormal scales (biased exp 0) were left
  unscaled (16× too large) and exponents 1..4 were flushed to zero. On the
  official file the attention `wo` had 91922/368640 blocks with exp==0, inflating
  its L2 to 4.92 vs the correct 2.61, which cascaded into a ~20× layer-0 MoE
  blowup and wrong expert routing. Now scaled in the float domain (exact for
  normals, correct for denormals/underflow): Q8_0 PPL 2739 → 4.65, matching the
  bf16/HF reference, decode coherent (#808). A CPU unit test sweeps all 65536
  fp16 bit patterns to guard the rescale in CI (#809).

## [0.12.6] - 2026-06-26

Patch release: a focused fix chain for the post-`</think>` answer-headroom logic
across all three reasoning formats — single-token `<think>` (Qwen3 dense),
multi-token `</think>` (Qwen3.6 NVFP4), and gpt-oss Harmony channels. Short
answers now stop cleanly instead of padding/repeating, and reasoning models no
longer return empty content when `max_tokens` is tight (the residual called out
in v0.12.5 is now resolved for gpt-oss). Validated on Qwen3-8B-Q8, Qwen3.6-35B-
NVFP4, and gpt-oss-20b (250-prompt corpus per model).

### Fixed
- **Post-`</think>` grace is content-aware** — the grace that suppresses a
  too-eager stop after the think block now releases the instant a real answer
  token appears, so a complete short answer (`VIOLET-2218`, `Paris`, `4`) stops
  on its own `<|im_end|>` instead of being padded or repeated to the
  raw-distance budget. `kMinAnswerAfterThink` stays a hard cap only for the
  empty-think (0-content) case. Wired through both the host per-step and the GPU
  conditional-graph decode paths (#798).
- **Whitespace tokens no longer release the grace** — the `\n`/`\n\n` a model
  emits right after `</think>` must not count as answer content; otherwise a
  stop following that newline produced a 0-content completion (reproducible
  ~75% on Qwen3.6 for terse prompts). A per-token whitespace mask, built once
  for think models and mirrored to the device, gates the content check on both
  decode paths (#799).
- **gpt-oss Harmony answer-headroom budget** — gpt-oss reasons in the Harmony
  `analysis` channel (closed by `<|end|>`) and has no `<think>` token, so the
  answer-headroom budget never armed and a tight `max_tokens` was consumed
  entirely inside reasoning, returning `finish=length` with an empty final
  channel. The budget now force-emits the whole final-channel opener
  (`<|end|><|start|>assistant<|channel|>final<|message|>`) when reasoning
  reaches the reserve limit — forcing `<|end|>` alone is not enough because the
  model re-opens analysis — committing it to the answer channel. Corpus
  empty-content count dropped from 18 to 2 (the rest are adversarial) (#800).

## [0.12.5] - 2026-06-26

Patch release: stops streaming reasoning models from leaking chain-of-thought
into the `content` channel (the streaming demux is now a single shared unit;
the think-budget force-cut is mitigated, though a residual remains when
`max_tokens` is too small for the model to finish thinking — increase
`max_tokens` for reasoning models). Plus repo and test-coverage housekeeping.

### Added
- **Adversarial degeneration prompt corpus** — 250 prompts across 8 categories
  (repetition, think-leak, special-tokens, adherence, long-context, multi-turn,
  multilingual, format) driven by a data-driven `degen_suite.py --corpus` runner,
  replacing the previous five-prompt battery (#795).
- **Task → skill routing table** at the top of `CLAUDE.md`: a fresh session maps a
  task (build, kernel, benchmark, degeneration, quant, server, new arch, PR, audit,
  docs) straight to the imp-specific skill that carries its playbook (#794).

### Changed
- **`DegenerationTest` loads SafeTensors/NVFP4 models**, not just GGUF, closing a
  coherence-coverage gap on the priority quant (#792).

### Removed
- **graphify knowledge-graph integration** (the skill, `.graphifyignore`, the
  generated-output ignore rules, and the `## graphify` section in `CLAUDE.md`) — it
  bloated the repo without enough payoff for this codebase (#794).

### Fixed
- **Streaming reasoning leaked into the `content` channel** for reasoning models
  (Qwen3.6-NVFP4 and others) while `reasoning_content` was also populated. The two
  streaming handlers shared a private demux that flipped to content at the *first*
  `</think>` and could not re-enter on a multi-token `<think>` (Qwen3.6 ships the
  markers as multi-BPE added tokens). The logic is now a single shared
  `StreamReasoningSplitter` that re-enters via text scan and holds back only a partial
  marker. Also caps the think-budget answer reserve (`max(max_tokens·budget,
  max_tokens − 256)`) so a model that finishes thinking within the window is no longer
  force-cut, which had spilled continued reasoning into `content` (#793).

## [0.12.4] - 2026-06-25

Patch release: fixes a server crash on native-NVFP4 MoE models that the v0.12.3
agentic long-context defaults exposed.

### Fixed
- **Native-NVFP4 MoE server crash on the first request** (`[FATAL] gemm_nvfp4:
  B.shape[1]=… must equal weight K=…`). The phase-2 weight-dispatch shim derived
  the M>1 prefill GEMM's K from the weight handle's `shape[1]`, which is the
  *packed* K/2 for prequant-loaded NVFP4 weights — so the dequant→cuBLAS fallback
  aborted. K is now taken from the activation (the logical K by the GEMM contract).
  The fallback was only reached because the v0.12.3 agentic KV-budget defaults can
  starve the CUTLASS NVFP4 prefill workspace; native-NVFP4 models (e.g.
  Qwen3-30B-A3B-NVFP4) on imp-server were affected. Regression-tested with a
  packed-shape `WeightDispatchTest` case. (#790)

## [0.12.3] - 2026-06-25

Agentic-serving and vision features, an NVFP4-MoE decode speedup, and a much
faster CI. No model-output or quality regressions.

### Added
- **Agentic server hardening** (#770): per-request speculative-decode toggle,
  inter-token-latency + cancellation metrics, prefix-cache safety under
  concurrency, an Anthropic-style keep-alive ping, and an agent benchmark harness.
- **Long-context KV-budget defaults for agentic workloads** (#771): the auto
  `max_seq_len` cap and KV-cache fraction are tuned so NVFP4 models no longer
  starve the KV cache while VRAM sits free.
- **Per-request vision binding** (#774): images/embeddings travel on the request
  and the worker encodes on admission, so vision requests batch with text instead
  of pausing the engine.
- **n-gram speculative decode is on by default** for dense models (gated off for
  MoE) (#781).
- **`parallel_tool_calls` is honored** on `/v1/chat/completions`: `false` emits at
  most one tool call, on both the non-streaming and streaming paths (#782).

### Changed
- **NVFP4 MoE decode ~2.6% faster** (Qwen3-30B-A3B-NVFP4, tg256): the SwiGLU
  down-projection precomputes `silu(gate)*up` once per element instead of once per
  output row; greedy output is byte-identical (#787, #788).
- **CI is much faster on PRs**: clang-tidy moved to its own non-required job
  scoped to changed files, so the required `Build` check dropped from ~26 min to
  ~4 min (#785); docs-only PRs skip the CUDA build (#780); the changed-files
  filter no longer fails closed (#783).
- **File-size gate** (`tools/check_filesize.py`, CI): flags oversized translation
  units by recompile blast radius; the 12 largest god-files were split into focused
  TUs with no functional change (#784).

### Fixed
- **Vision global image bind restored** for the C-API / imp-cli path (a #774
  regression) (#776).
- **Soundness & hardening** (#772): four HIGH-severity fixes — KV-cache
  use-after-free on eviction, fail-fast on poisoned context, Anthropic `x-api-key`
  auth, and a bounded decode-burst.
- **NUL byte removed** from a `handlers.cpp` comment that made `grep`/`ugrep` treat
  the most-edited server file as binary (#782).

## [0.12.2] - 2026-06-23

Follow-up bug-fix release from a v0.12.1 acceptance re-test. Server/API polish
and gpt-oss output formatting; no kernel or perf changes.

### Fixed
- **gpt-oss Harmony channels are now parsed** instead of leaking into the
  response. The model emits `<|channel|>analysis<|message|>…<|end|><|start|>assistant<|channel|>final<|message|>…`;
  the analysis/commentary channels now map to `reasoning_content` (Anthropic:
  `thinking` block) and the final channel to `content`, with all control markup
  stripped. Applied on non-stream + streaming `/v1/chat/completions` and
  `/v1/messages`. (The MXFP4 decode itself was already correct.) (#765)
- **`/v1/completions` streaming is per-token again** for think-capable models —
  it was buffering the whole response into a single SSE frame. (#766)
- **Port-in-use fails fast**: the listen socket is bound before the model load,
  so a port conflict errors in <1 s instead of after a full model load. (#766)
- **gpt-oss MXFP4 SafeTensors load** no longer prints the stale "no SafeTensors
  MXFP4 decode path" warning — the path exists (experts are transcoded
  MXFP4→NVFP4 at init and run the CUTLASS NVFP4 grouped GEMM). (#765)

### Changed
- **cpp-httplib v0.46.1 → v0.48.0** (security hardening + a fix that ignores
  Range headers on unknown-length streaming responses; other pinned deps —
  CUTLASS v4.5.2, GoogleTest v1.17.0, nlohmann/json v3.12.0, CUDA 13.3 — were
  already current).

## [0.12.1] - 2026-06-23

A bug-fix release closing the issues found in a black-box acceptance test of
v0.12.0 — mostly server/API correctness and out-of-the-box defaults. No kernel
or perf changes; published benchmark numbers are unaffected.

### Fixed
- **SSE streaming is real per-token again** on `/v1/chat/completions` and
  `/v1/messages`. A single-stream request was buffering every token and
  flushing them all at generation end (TTFT ≈ full latency): the GPU-autonomous
  conditional-graph decode loop only surfaces its tokens to the host once the
  whole burst completes. Streaming requests now stay on per-step decode for
  genuine per-token delivery; non-streaming requests keep the faster loop
  (~2–5% decode cost for streams on 8B-Q8). (#754)
- **Streaming no longer hangs the client.** `/v1/messages` with a thinking model
  and a small `max_tokens`, and `/v1/completions` with `stream:true`, could spin
  forever without ever emitting a terminal event (`message_stop` / `[DONE]`) when
  the final token was swallowed by the reasoning / think-strip path. The stream
  loops now always terminate. (#755, #757)
- **`response_format: json_schema` can no longer emit invalid JSON.** An unbounded
  `integer`/`number` field let a degeneration-prone model run a digit loop to
  `max_tokens`, leaving the JSON unterminated. The `NUMBER_VALUE` grammar now
  caps the digit run so the number always closes. (#751)
- **`think_budget = 0` now disables thinking** (as documented) instead of removing
  the budget cap — which made a think-capable model reason until `max_tokens` and
  return empty `content`. (#752)
- **SafeTensors directory with a trailing slash is addressable again.** The served
  model id is derived from the path basename; a trailing slash made it empty, so
  `/v1/models` returned `id: ""` and every request was rejected. Trailing
  separators are now stripped. (#756)
- **Clearer model-load errors.** A present-but-corrupt file (e.g. bad GGUF magic)
  now reports *"invalid or corrupt model file"* instead of *"file not found"*, and
  a missing local path is no longer misrouted to the HuggingFace resolver with a
  nonsensical `git clone` hint. (#759)
- **Version string no longer drifts.** `imp_version()` was hardcoded `0.11.2`
  while the project version was `0.12.0`; it is now single-sourced from the CMake
  project version. (#760)
- **Docs:** dropped the stale "no continuous batching" claim, added the required
  `model` field to the README quickstart curl, removed the trailing slash from the
  SafeTensors examples, and documented the C API as source-build-only (the runtime
  image ships only binaries). (#760)

### Changed
- **Prefix/prompt caching is now ON by default for the server.** It shipped
  default-off, so `cache_read_input_tokens` always reported 0 and warm prompts got
  no TTFT win unless an `imp.conf` opted in — contradicting the documented
  behaviour. Library/C-API embedders (who drive `EngineConfig` directly) are
  unaffected; it remains auto-disabled for SSM/GDN models. (#758)
- **The released Docker image now ships `imp-bench`** (it is documented; CI keeps
  it off for build speed). (#760)

## [0.12.0] - 2026-06-21

### Added
- **VRAM-aware auto `max_batch_size` — up to ~2.4× server throughput on MoE.**
  The old heuristic sized the concurrency cap purely by weight footprint, so a
  >20 GB model (e.g. Qwen3-Coder-30B-A3B-FP4) was pinned to batch=1 and served
  concurrent requests strictly one at a time — even with ~10 GB of free VRAM
  sitting idle. The cap is now derived from the real *post-load* headroom (free
  VRAM minus the about-to-be-uploaded weight footprint), sizing each concurrent
  slot to keep a 4096-token serving-context floor of KV within 60 % of that
  headroom. The KV cache is already a shared paged pool clamped to free VRAM
  downstream, so a larger cap engages continuous batching **without** OOM risk;
  the weight-footprint tier remains a floor, so the cap never regresses below the
  old default. Measured on Qwen3-Coder-30B-A3B-FP4: auto cap **1 → 15**, aggregate
  decode **258 → 609 tok/s at 16 concurrent (2.4×)**; Qwen3-14B-NVFP4 **4 → 17**,
  no OOM. Serving-only: `imp-cli`/`--bench` still force batch=1, so single-stream
  throughput, the perf-baseline gate, and CLI behaviour are unchanged.
- **`imp-bench nvfp4` — isolated NVFP4 dense GEMM bench mode** that times the GEMM
  kernel on its own (used to refute the cp.async-occupancy hypothesis on sm_120a).

### Changed
- **MoE NVFP4 models load materially faster.** Two init-time wins on the
  per-expert scale-factor path: the CUTLASS NVFP4 SF cache is now slab-allocated
  in a **single** `malloc` instead of 18.6k (≈ **−785 ms** on a 30B MoE), and the
  per-expert SfAtom conversion is batched (**18.6k → 337** convert launches).
- CI: `setup-python` v5 → v6 (Node 24) to clear the Node 20 deprecation warning.
- **Internal: clang-tidy cleanup.** Fixed ~51 host-side findings — int-multiplication
  widening before 64-bit use (KV/SSM/weight-upload size math), uninitialized struct
  members, dead stores, set-but-unused counters, an unused lambda capture, unused
  `using` declarations, and inefficient string concatenations — plus intent comments
  on the deliberately-empty parse-fallback catches. No behavior change; the remaining
  findings are intentional (documented empty catches, two large functions) or
  clang-tidy parser artifacts.
- **Zero-warning build + single-arch CI.** Silenced every remaining compiler
  warning — nvcc `#128-D` (unreachable loop, fixed via `if constexpr`/`else`),
  `#550-D` (set-but-unused), GCC `-Wformat-security`, a test hex-escape that was
  actually round-tripping invalid UTF-8 instead of "Grüße", and ignored
  `[[nodiscard]]` results. CI now compiles `sm_120a` only
  (`IMP_DISABLE_120F_FALLBACK=ON`), halving device-compile time and no longer
  emitting each `.cu` diagnostic twice; the shipped fatbin (release-docker) keeps
  the `compute_120f` PTX fallback for 5080/5070 SKUs.

### Fixed
- **Dense models no longer OOM-crash at startup under auto `max_batch_size`.** The
  VRAM-aware auto batch sizing (above) sizes the cap against a 4096-token reference
  context, but the KV pool is provisioned at `max_batch_size × max_seq_len`. On a
  small-weight / large-headroom config (e.g. dense Q8 on a 32 GB card: batch auto→25,
  25 × 16384-token slots = 25600 blocks / 57.6 GB) the KV `cudaMalloc` exceeded VRAM
  and the server aborted with `out of memory` at context creation. The KV pool is now
  clamped to the post-weight VRAM that physically remains — it is a paged pool with
  scheduler admission control, so a smaller pool only bounds concurrency under load and
  never under-serves a single sequence. MoE/NVFP4 configs (which already fit) are
  unchanged; verified no-OOM + coherent + continuous-batching on dense Q8, MoE NVFP4,
  and dense NVFP4.
- **`docker run imp:latest --help` now prints imp-server's flags.** A leading flag
  was taken as the *command name*; it didn't match `imp-server|imp-cli`, fell
  through the entrypoint's passthrough branch, and ran `exec --help` — printing the
  bash `exec` builtin's help instead of the server's. The entrypoint now follows
  the standard official-image pattern: if the first argument starts with `-`, it is
  a flag for the default command, so `imp-server` is prepended before dispatch and
  `--help`/`--version` reach the real binary.

## [0.11.3] - 2026-06-17

### Added
- **Stage-3 server test gate** (`make test-server`) boots a real `imp-server` and
  gates on the OpenAI+Anthropic wire batteries; plus a gcov coverage harness
  (`make coverage`) and a coverage-hardening oracle sweep (Q4_0/Q5_K/FP8/mxfp4,
  tool-call + Bearer-auth units) on a two-stage CI gate.
- **Developer tooling & build hygiene:** `CMakePresets.json`, `AGENTS.md`, an
  in-repo `CLAUDE.md`, `.clang-tidy` + `make tidy`, a CI `lint` job (changed-lines
  clang-format + advisory clang-tidy), single-sourced dependency pins
  (`cmake/imp-deps.cmake`), `.gitattributes`, `BENCHMARKING.md`, and a reusable
  `scripts/bench_gate.sh` perf gate.
- **Anthropic `/v1/messages` honours the `thinking` field.** `{"type":"enabled","budget_tokens":N}`
  enables extended thinking with a budget and `{"type":"disabled"}` turns it off;
  previously the field was dropped in the Anthropic→OpenAI conversion, so the
  request could not influence thinking at all. The field routes to the same
  internal controls as the OpenAI path (`enable_thinking` + `think_budget`);
  `budget_tokens` maps to imp's fractional `think_budget` (`budget_tokens /
  max_tokens`, clamped to 1.0).

### Changed
- **CUTLASS v4.5.1 → v4.5.2.** Verified by a full build + 187/187 quant tests
  (including the grouped-GEMM fp64 oracles) — NVFP4 GEMM bit-exact under 4.5.2.
- **Internal hygiene:** a structural audit (`AUDIT.md`/`STRUCTURE.md`/`DISPATCH.md`);
  84 mechanical clang-tidy fixes (member-init, multiplication-widening,
  value-param) with two verified-false-positive checks suppressed;
  reserved-identifier `_COUNT`→`COUNT`; `.gitignore` no longer ignores
  `src/core/`; a canonical sm_120a kernel-spec doc + standalone reference kernels.

### Fixed
- **Q4_0 GGUF decode was silently wrong (correctness fix).** The Q4_0 dp4a GEMV
  read packed nibbles INTERLEAVED while ggml Q4_0 is SPLIT (low nibbles =
  elements 0–15, high = 16–31) and trusted a mis-scaled zero-point, so every Q4_0
  dense **and** MoE decode produced garbage. It was never caught because no Q4_0
  model is in the test suite; a new fp64 oracle surfaced it. Fixed to the proven
  Q4_K nibble extraction with an internally-summed zero-point — one traits
  function covers the dense decode, fp32, residual, and MoE Q4_0 paths.
- **`scripts/bench_gate.sh` perf gate was silently broken.** A stray `2>&1` left
  the parsed stderr empty, so `set -e` exited before the gate could report — and
  the gate had in fact never run (the GPU CI job is skipped). Fixed and validated
  live on the RTX 5090.
- **Disabling thinking now actually suppresses reasoning.** When thinking was
  turned off, a think-model still reasoned for many prompts (on both `/v1/messages`
  and the OpenAI path with `enable_thinking:false`+`think_budget:0`). Two root
  causes, both fixed:
  - **Tokenizer:** Qwen3 ships `<think>`/`</think>` (and `<tool_call>`, `<|fim_*|>`)
    as added tokens with `special=false, normalized=false`. imp only atomic-matched
    `special` tokens, so these were BPE-split into `"<","think",">"` — the model's
    `<think>` never hit the single-token stop guard, and the template's closed
    `<think></think>` no-think block was just text the model re-opened. imp now
    follows HF semantics and atomic-matches any `normalized=false` added token
    (also fixes tool-call/FIM marker tokenization).
  - **Server:** the heuristic that re-enables thinking when the prompt tail
    contains `<think>` (for Nemotron/Phi-4 templates injecting an *open* prefix)
    fired on the *closed* no-think block too. It now requires an **unclosed**
    prefix (`<think>` present, no matching `</think>`).
- **Embeddings reject inputs longer than the single-pass hidden buffer (was a server abort).**
  Follow-up to the over-long-prompt fix below: `/v1/embeddings` mean-pools every
  token's hidden state, which only fits when the whole input is prefilled in one
  pass. A longer input is chunked and `hidden_` keeps only the last chunk, so
  `view_hidden(n)` sliced out of a `[max_tokens, *]` buffer and aborted the whole
  process (`Tensor::slice` IMP_CHECK) — and `max_tokens` (the executor workspace,
  e.g. 4096) can be far below `max_seq_len` (e.g. 32768). The embeddings guard now
  also bounds on `executor->max_tokens()`. (Found via the new coverage harness.)
- **Over-long prompts are rejected (400) instead of crashing the server (SIGSEGV).**
  The server gated prompt length on the model's *declared* max context
  (`imp_model_max_seq_len`), but the engine VRAM-auto-sizes the *actual*
  allocated context, which can be much smaller (e.g. ~4096 for a 14B on a tight
  budget). A prompt longer than the allocated context but shorter than the model
  max passed the length check and overran the per-sequence KV/position buffers →
  SIGSEGV. Added `imp_context_max_seq_len()` (the engine's effective allocated
  context) and gate on it; also added the missing length guard to the
  `/v1/embeddings` path. Over-long input now returns
  `400 "… exceeds context window (N tokens >= M max)"`.
- **`[runtime] max_batch_size` from imp.conf is now honored for engine sizing.**
  The server built `ImpConfig.max_batch_size` only from the `--max-batch` CLI
  flag (default 0 = auto), so the imp.conf value was dropped — it reached the
  engine only as the decode-batch cap, never the scheduler/KV/workspace sizing.
  Precedence is now: per-request override > `--max-batch` > `[runtime]
  max_batch_size` > 0 (engine auto-sizes from the model's weight footprint). The
  `runtime.max_batch_size` default changed 4 → 0 to match the documented
  "0 = auto" semantics (and the engine now logs the resolved batch size). Audit:
  this args-vs-imp.conf drop was unique to `max_batch_size`; all other imp.conf
  keys reach the engine via the ImpConfig builder or a direct `runtime_config_`
  read.

## [0.11.2] - 2026-06-14

Patch release. Hardens the HTTP surface so malformed/invalid client input can
never crash a request into an opaque 5xx.

### Fixed
- **Bad request input now returns 400 + a JSON error envelope, never a bare 500.**
  Invalid UTF-8 in a request body (e.g. an agent byte-truncating a prompt and
  splitting a multibyte char) made `json::parse` throw, the error envelope
  echoed the offending bytes, and `err.dump()` then threw `json::type_error.316`
  (dump rejects ill-formed UTF-8) — which escaped the parse-error catch and
  surfaced as an opaque `500 Internal Server Error` with no body. Added a global
  `set_exception_handler` (json exceptions → 400 `invalid_request_error`, others
  → 500 with a JSON body) plus a `dump_safe()` helper (`dump` with
  `error_handler_t::replace`) used on every response / SSE / error / request-log
  body, so an ill-formed byte can never crash a request. Audited all body-taking
  endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  `/v1/messages`, `/tokenize`, `/detokenize`). (DEBUG-500-on-bad-input.md)

### Added
- `tests/test_server_robustness.py` — manual server-level battery asserting that
  malformed JSON / invalid UTF-8 / non-object / wrong-type / missing-field input
  on every endpoint returns 4xx + an error envelope (never 5xx) and valid
  requests return 2xx.

## [0.11.1] - 2026-06-14

Patch release. Fixes a server wedge where interleaved `/v1/embeddings` +
`/v1/chat/completions` traffic returned empty completions.

### Fixed
- **Embeddings no longer cancels in-flight generations (#710).** The
  `/v1/embeddings` handler took exclusive C-API access by calling
  `BatchingEngine::stop()`, which cancels every in-flight request. Under
  interleaved embed+chat load any concurrently-running generation came back
  empty (`finish_reason:"cancelled"`, the lone reasoning token logged as
  "0 completion tokens"); `stop()` also left the cancelled sequences' KV blocks
  allocated, piling up orphaned decode work under sustained load. Added a
  graceful `BatchingEngine::pause()`/`resume()` handshake that lets the worker
  *finish* in-flight requests before parking (no cancellation, no thread churn)
  and switched the embeddings path to it.

### Added
- `tests/test_server_0token_battery.py` and
  `tests/test_server_embed_chat_interleave.sh` — manual server-level regression
  coverage for the wedge (content / size / temperature / trailing-cue /
  assistant-prefill / sustained-load lanes; gate on the empty-completion rate).

## [0.11.0] - 2026-06-13

47 commits since v0.10.0. Headlines: faithful per-family pre-tokenizers (#657 —
the entire measured cross-engine perplexity gap turned out to be tokenization,
not numerics; four families are now byte-identical to llama.cpp/HF), NVFP4
long-context prefill pushed at-or-ahead of vLLM (chunk-2048 default + FP16-QK
FA2 as the primary hd=128 prefill — MoE pp4096 now leads vLLM, dense pp4096 a
near-tie), `kv_cache.dtype=auto` honoring the model's FP8-KV hint for verified
Qwen3 families (~768 MiB saved), opt-in n-gram speculative decoding, full
gpt-oss-20b GGUF MoE support, and a VRAM audit that reclaimed several GiB on
NVFP4. Benchmarks refreshed (`BENCHMARKS.md`, commit-anchored); correctness gate
(full GTest suite) green.

### Added

- **N-gram prompt-lookup speculative decoding** (#668–#670, opt-in). Draft tokens
  are matched from the prompt/context suffix and verified in a burst-hybrid loop;
  output stays token-identical to plain greedy. `--set speculative.ngram=true`
  (knobs: `k`, `min_match` default 6 — precision beats frequency, `give_up_after`,
  `burst`). CLI ~+6% on long generations; opt-in because draft-poor workloads
  regress. Server-enabled (penalties in verify, think-budget bursts).
- **gpt-oss-20b GGUF MoE** (#690) — full GGUF path: MXFP4→NVFP4 expert conversion,
  expert biases, attention sinks, sliding-window attention, residual rescale. The
  GGUF checkpoint previously NaN'd in MoE prefill; SafeTensors gpt-oss already
  worked.
- **`IMP_PPL_DUMP=full`** dumps per-position NLL for cross-engine perplexity
  forensics (#655).

### Changed

- **KV cache: `kv_cache.dtype` now defaults to `auto` — honors the model
  author's `kv_cache_quant_algo=FP8` hint for verified arch families.** Modelopt
  NVFP4 checkpoints declare `kv_cache_quant_algo=FP8`; imp previously parsed but
  ignored it (FP16, manual `--kv-fp8` to opt in). `auto` upgrades KV to FP8 E4M3
  only for arch families that pass a long-context quality gate
  (`kv_fp8_hint_default_safe`). Allowlisted today: **Qwen3 dense + Qwen3 MoE** —
  measured on a 3.9k-token context, FP8 vs FP16 KV: Qwen3-14B PPL +1.07%,
  Qwen3-30B-A3B neutral, both coherent, ~768 MiB KV VRAM saved. Other
  hint-declaring families (Phi-4, Nemotron-H, Qwen3.5/3.6, Gemma-4) stay FP16
  until verified. `dtype = "fp16"` opts out; `--kv-fp8` forces FP8 on any model.
  The `auto` resolver also makes config-file `dtype = "fp8"|"int8"|"int4"|…`
  selections take effect (previously only the CLI flags did).

### Performance

- **Prefill chunk size default 512 → 2048** (#672) — MoE pp2048 **+127%**
  (Qwen3-30B-A3B 15.7k→35.7k tok/s), pp4096 +77%; activation-quant dedupe is
  bit-identical. Also fixed a grouped device-args GEMM silent corruption at n≈900.
- **FP16-QK FA2 is now the primary hd=128 prefill** (#687) — at-or-above cuBLAS at
  every pp (pp1024 +24%, pp2048 +52%), so the S-matrix buffer is skipped for
  hd=128: **−380 MiB** device memory. Re-benched 2026-06-13: MoE pp4096 now +4%
  ahead of vLLM, dense pp4096 ~1.04× (was 1.27×).
- **FA2 full-rate accumulate default-on** (#673/#674) — f16-accumulate QK^T and PV
  in the FP16-QK FA2 prefill kernel: −18% pp4096 kernel time, MoE e2e +9.7%, PPL
  unchanged.
- **Conditional-graph decode loop for NVFP4 "think" models** (#649) — **+45%**
  think-decode by keeping the reasoning loop inside one captured graph.
- **Pipelined constrained decoding** (#650/#651) — schema-mask fast path +
  forward-N+1 pipeline: `json_schema` decode **102 → 235 tok/s**.
- **FP8-KV deterministic-cuBLAS forcing scoped to non-FA2 configs** (#682) — removes
  a −35% pp4096 MoE tax on `--kv-fp8` (it was an April-era forcing of the
  single-block deterministic MoE permute, not the gather).
- **VRAM reclamation on NVFP4** — fallback-only workspace buffers skipped on
  SafeTensors (#678, +827 MiB free), duplicated per-expert micro-scales freed
  (#679, −1728 MiB on 30B-A3B), CUTLASS scale-factor dedup (#685/#686, −1810 MiB on
  NVFP4-prequant), contiguous per-(layer,proj) micro-scale slab (#689).
- **FA2 grid-underfill band** — TWOSLOT K/V rotation runs 2 CTAs/SM at full Bkv=64
  (#653); Bkv=32 variant for the same band (#597). 16-B vectorized x/gate/up loads
  in the NVFP4 GEMV dot helpers (#671).

### Fixed

- **SPM tokenization: USER_DEFINED pieces now literal-matched — gemma
  multi-space runs were never canonical** (#657). gemma-3 stores indentation
  tokens (`'  '`…27×`' '`) and HTML tags as SentencePiece user-defined
  symbols with literal-space pieces; imp's ▁-substituting BPE could never
  reproduce them and emitted N single-space tokens per indent run. The
  special-pieces literal pre-split now includes type-4 (USER_DEFINED)
  alongside CONTROL. gemma-3-12b: token ids identical to llama.cpp, corpus
  count 3395 == llama.cpp, matched-band NLL +37.5% → **−0.4%**, corpus PPL
  15.53 → 10.57. Together with the Qwen2 pre-tokenizer fix, the entire
  measured cross-engine quality gap was tokenization, not numerics. Also:
  `--set diagnostics.dump_tokens=true` with `--perplexity` now dumps the
  full corpus token stream for cross-engine diffs.

- **Qwen2/Qwen3 tokenization was non-canonical on symbol/digit sequences**
  (#657). The gpt2 pre-tokenizer fallback split every punctuation character
  individually and grouped digits in threes, so canonical BPE merges were
  impossible (`->` became `-`+`>`, `(x):` four chunks) — on a 10 KB
  code/markdown corpus imp produced 3690 tokens vs llama.cpp's canonical
  3084 (+20%), inflating teacher-forced NLL on matched text by +70% and
  segmenting every production prompt containing code non-canonically.
  New faithful `qwen2_pre_tokenize` (contractions, prefix-char+letter runs,
  single digits, symbol runs, GPT-2 whitespace backtracking), routed via
  GGUF `tokenizer.ggml.pre=qwen2` and detected from the HF tokenizer.json
  Split regex for SafeTensors. Token streams are now id-identical to
  llama.cpp on probe texts; corpus count 3084 == llama.cpp; matched-band
  NLL gap vs llama.cpp: +70% → **+1.3%** (Qwen3-8B-Q8_0 PPL on the corpus:
  40.5 → 10.98). Greedy locks unchanged; code generation verified coherent.

- **Prefill dispatch chain exhaustion now throws instead of silently emitting
  garbage** (#654). `flash_attention_blackwell` declined hd=256 (smem over the
  99 KB sm_120 opt-in) by silently falling back to `flash_attention_prefill_tc`,
  whose launch also fails at hd=256 — unchecked — leaving the output buffer
  as garbage (teacher-forced PPL ~1e10 when forced via `fmha_sm120=never`).
  `flash_attention_blackwell` now returns a decline (`bool`, launch-checked;
  correct at hd∈{64,96,128}: forced-blackwell Qwen3-8B PPL 40.50 vs cuBLAS
  40.51), the tc fallback (which also lacked `q_offset` for chunked
  continuations) is removed from the chain, and `attention_prefill_dispatch`
  throws a descriptive error when no kernel accepts. Default routing is
  unaffected (FA2/WMMA serve all supported head dims first).

- **fp8-QK FMHA demoted to opt-in — gemma-3 long-context prefill was
  catastrophically degraded** (#511 reopened/resolved). The raw (unscaled)
  Q/K→e4m3 conversion compounds per-layer score error on real activations:
  teacher-forced PPL gemma-3-12b 16.6→549 once chunked prefill crossed the
  S-matrix cap (~3.5k ctx) and the fp8 kernel started serving (Qwen3-8B
  forced through it: 40.5→4506). The #511 long-ctx "validation" (needle at
  5.2k) never exercised the kernel — `fa2_fp16qk` served those chunks.
  `attention.fp8_fmha` now defaults to `"never"` (strictly opt-in `"on"`);
  the dispatch-chain FA2 call runs f16-QK; hd≠128 long prefill is served by
  the FP16 WMMA kernel (PPL-identical to cuBLAS, 15.53 both at n=3441 incl.
  sliding window). gemma-3-12b @8.3k-token corpus: PPL 549→11.1. Bonus:
  Qwen3-8B pp4096 +6.9% (the tuned f16-QK FA2 replaces fp8-QK in the
  unchunked chain). Follow-ups: #654 (broken `flash_attention_blackwell`
  last-resort), #655 (IMP_PPL_DUMP position mapping).

- **Speculative-decode conditional-graph loop wrote KV one slot too high**
  (#683 → #692). `CudaGraphConditionalRunner::setup()` double-incremented the
  first-forward position, so every fresh-captured verify loop duplicated the
  last burst KV entry (wrong first token on rearm). Root-caused to the
  position off-by-one (not the rearm fast path, which #684 had disabled as a
  stopgap and is now default-on again). Byte-perfect across mb=8/4/0, server
  API, and Q8 / NVFP4-dense / NVFP4-MoE. Also drains the multi-token verify
  tail when a request finishes mid-chunk.
- **`attn_logit_softcap` was silently dropped on the cuBLAS FP32-S prefill
  path** (#688) — Gemma-2-class hd=256 prefill skipped the cap. Fixed via a
  `softcap_fp32_kernel`; FA2/decode/KV-write paths were already correct.
- **`gemm_kv_batched` output stride** is now derived from the actual K/V
  pointer distance (#677/#691), fixing a Q4_0 determinism mismatch plus a
  cross-block WAR hazard in the fused FP32→FP16 softmax downcast.
- **Constrained decoding SIGBUS** on SafeTensors `json_mode` + raw control
  characters in constrained strings (#650).

## [0.10.0] - 2026-06-09

151 commits since v0.9.1. Headlines: gpt-oss-20b support, LoRA hot-swap, the
INT8-IMMA prefill family (GGUF prefill from "always behind" to ahead of
llama.cpp on the MoE/Q6_K heroes), a GeForce-Blackwell tensor-core-rate
recalibration that unlocked several opt-in compute-type levers, and a roofline
audit that mapped the remaining decode/prefill ceilings. Benchmarks refreshed
(`BENCHMARKS.md`, commit-anchored); correctness gate (full GTest suite) green.

### Added

- **gpt-oss-20b** (#547, #572, #574) — MXFP4 experts converted to NVFP4 at load,
  attention sinks, Harmony channel split, YaRN/split-K/FP16-range fixes. CUTLASS
  grouped-GEMM prefill registration took pp512 from ~1.9k to 16–19k tok/s (~10×);
  decode 310–345 tok/s.
- **LoRA / PEFT adapter hot-swap** (#522/#571) — runtime low-rank deltas, no
  weight patching.
- **IQ4_NL / IQ4_XS** i-quant GGUF support (#556/#561).
- **Gemma-4 vision** (gemma4v, #490) + Gemma-3 mmproj projector load (#489).
- **Teacher-forced perplexity tool** `imp-cli --perplexity` (#481), chunk-aware
  since #553 (determinism-proof eval).
- **Anthropic `cache_control` prompt caching** — prefix-cache pinning + usage
  accounting; `prefix_cache` default on (#522/#541).
- **`attention.fa2_f16acc`** opt-in (#597) — f16-accumulate QK^T in the FP16-QK
  FA2 kernel: +3–4 % pp2048/pp4096 NVFP4 prefill for +0.37 % PPL.

### Performance

- **INT8-IMMA prefill GEMM family** (#612–#619, default on since #617) — fused
  dequant on INT8 tensor cores for Q8_0/Q4_K/Q6_K/Q5_1, incl. MoE grouped
  variants and N-tail support. Qwen3-30B-A3B (MoE) and Qwen3-14B-Q6_K prefill now
  **ahead of llama.cpp**; gemma-4-26B MoE +111 % cumulative; Q8 dense 1.13×.
- **GeForce tensor-core-rate calibration** (#606) — sm_120 silicon runs FP4
  block-scale at ½ datasheet and FP16/FP8 f32-accumulate at ¼ rate; roofline
  peaks corrected. Unlocked `gemm.cublas_fp16_acc` (default on per-arch, denies
  Gemma-3/4 + gpt-oss; #611) and CUTLASS small-N pingpong (kv_proj 2.1×).
- **FA2 prefill** — ldmatrix operand fetch + register-resident Q + 8-warp fp16qk
  (#609, −28 % late-chunk kernel); FP16-QK short-prefill path (#525, +25–35 %
  pp512); seq-adaptive Bq (#493).
- **dp4a GEMV** — 16-B-aligned `block_q8_1` (48-B stride) removes the
  activation-load ceiling (#619); LDG.128 Q4_K/Q5_K weight loads (#607).
- **NVFP4 decode** — NVFP4 lm_head default-on for GDN/hybrid models (+11.4 %
  Qwen3.6-35B decode, #483); opt-in NVFP4 for recipe-excluded hybrid projections
  (+53 % GGUF `nvfp4_ssm_proj`, +3.8 % Nemotron `nvfp4_attn_proj`; #486).
- **RMSNorm** — warp-per-row FP16 RMSNorm for batch prefill (#620).

### Fixed

- **SafeTensors Llama-family RoPE** — NeoX layout for LLAMA/MISTRAL/MIXTRAL/LLAMA4
  (GGUF pre-permutes Q/K, HF does not); fixed Phi-4 prompt-blind output (#503).
- **Nemotron-H NoPE attention** (#518); **MXFP4 GGML type-39 nibble order is
  split, not linear** (#567); **sliding-window prefill** routed through the
  cuBLAS masked softmax — the correctness reference for hd=256 + window (#566/#569).
- **Pinned-staging reuse race** corrupted chunked continuations (root cause of the
  "fa2_fp16qk Llama bug" + long-ctx e4m3 reroute, #548/#568); **in-place
  float→half S/P-tile compaction race** in the WMMA prefill kernels (#528/#539).
- **Recurrent-state slot** leak/concurrency for SSM/GDN (#500/#501);
  **model-reload SIGSEGV + VRAM retention**, strict OpenAI model semantics, no
  auto-swap (#507).
- **Determinism** — `[runtime] deterministic` now works via the C API + bit-stable
  perplexity, proven on GDN-hybrid (#542). Prefix-cache stale-table hit fixed
  via content-compare (#538).
- **Constrained decoding** — per-token FSM simulation for schema JSON, `$ref`/
  `$defs` incl. recursive schemas, regex enforcement, whole-token validation
  (#497/#498/#499/#517/#562); **thinking** default requires template evidence,
  not just a vocab `<think>` token (#513/#563).
- **Gemma-3 garbage output** — `apply_arch_defaults` double-counted the
  `norm_weight_offset` (llama.cpp already bakes the `+1` into `*norm.weight`);
  GEMMA3 now uses offset 0.
- **Gemma-3 chunked prefill** enabled (uniform head_dim/kv_heads, byte-identical
  greedy vs single-shot across SWA boundaries); **`imp-bench` build break**.

### Changed / internal

- **`ModelProfile`** — one source of truth for architecture classification; all
  hot-path `cfg.arch == X` checks route through it; `attn_variant` enum for the
  SWA/NoPE dispatch (#622/#623/#625).
- **VRAM cache rebuild** — RAII ownership of all 8 caches (double-free is now a
  compile error), one authoritative storage tier, honest diagnostics (#621).
- **Roofline audit** (`docs/audit/roofline_2026_06_07.md` + the
  `tools/roofline/` ncu+nsys pipeline): shipped the `attn_fa2` f16-acc lever
  (#597) and documented the structural ceilings of MoE-decode `gemv_nvfp4`
  (#600), MoE-prefill `gemm_grouped_nvfp4` (#601), and hd=256 prefill coverage
  (#603, blocked on #566).
- **CUDA 13.3 native images** (#520); dependency pins CUTLASS v4.5.1, GTest
  v1.17.0, nlohmann/json v3.12.0, httplib v0.46.1.
- **Server-level degeneration suite** `tools/analysis/degen_suite.py` (#508).

## [0.9.1] - 2026-05-27

### Critical fixes

- **FP8 prefill degeneration on sm_120** (#446) — cuBLAS 13.4 `cublasLtMatmul`
  returns `CUBLAS_STATUS_NOT_SUPPORTED` for FP8 E4M3 GEMMs at non-aligned M on
  consumer Blackwell. The prior `cublasGemmEx` fallback silently produced garbage
  (no per-tensor scales), corrupting the KV cache and causing decode degeneration
  (repetition loops, immediate EOS) on **all GGUF models**. Fix: FP8 prefill
  auto-disabled on sm_120; FP16 weight cache used instead. FP8→FP16 dequant
  fallback added as defense-in-depth. cuBLAS algo benchmarking now validates
  return status during warmup.
- **Server hallucination at turn boundaries** (#442) — thinking models at high
  temperature could hallucinate `Human`/`<think>` turn markers, leaking internal
  reasoning. Fixed with stop-sequence detection.
- **CUDA graph crash on Nemotron-H** (#443) — Mamba2 SSM layers auto-detected
  and excluded from CUDA graph capture.

### New model support

- **Gemma-4 dense (31B)** (#444) — weight mapping fix: `mlp.{gate,up,down}_proj`
  was unconditionally routed to shared expert slots, breaking dense Gemma-4 models.
- **Phi-4-reasoning-plus NVFP4** (#429) — fused qkv_proj/gate_up_proj support.
- **Nemotron-Labs-3-Elastic-30B-A3B NVFP4** — newer QAD quant, ~70 tok/s decode.

### Performance

- **dp4a dense prefill for Q4_K/Q5_K** (#436) — computes directly from quantized
  blocks (0.55 B/elem) instead of FP16 weight cache (2.0 B/elem) at small M.
- **Q4_K_M GGUF support** (#431, #432, #414) — dequant fallback + FP8 D2H fix +
  fused MoE dp4a. Qwen3-30B Q4_K_M: pp512=3616, tg256=271.
- **CUTLASS NVFP4 dispatch fix** (#428) — zero-copy MoE expert registration
  eliminated 15 GiB D2H copy on Qwen3.6-35B.

### Dependencies

- CUTLASS v4.5.0 → v4.5.1 (#447)
- cpp-httplib v0.45.0 → v0.46.0 (#448)

### Other fixes

- FP16 cache VRAM overcommit on dense Q4_K_M (#435)
- ForwardPassTest + MoE + quant tests segfault on weight registry (#437)
- CUDA teardown errors in Model destructor (#439)
- Q5_K forward pass test NaN from cross-test cuBLAS state contamination (#445)

### Earlier unversioned work (pre-0.9.1; was a stray second "Unreleased" section)

- **Gemma-4 FP8 prefill carve-out removed** — the 2026-05-09 measurement
  showing -5..-19% prefill on Gemma-4 vs FP16 has substantially closed
  with intermediate prefill work (PRs #177, #181). Re-measured 2026-05-15
  on Q4_K_M: pp128 +1.0%, pp512 -0.9%, pp833 -4.2%, pp2048 **+7.3%** —
  neutral with long-context advantage. FP8 also halves the activation
  cache. Coherence bit-exact on chat prompts. Closes the last entry in
  the "Gemma-4 remaining carve-outs" roadmap section.
- **Gemma-4 NVFP4 decode cache for Q*_K source weights** — drops the
  "per-layer head_dim not yet supported" carve-out at `engine.cpp:864-866`.
  The per-tensor convert→quantize loop in `executor_pre_dequant.cu` handles
  mixed (N, K) shapes correctly since each `wcache_.nvfp4` entry carries its
  own dimensions. Gemma-4-26B-A4B-it-Q4_K_M: pp512 1713 → 2394 tok/s (+40%),
  tg256 176 → 197 tok/s (+12%). Coherent on chat prompts; pre-existing Q4_K_M
  code-gen drift is orthogonal. `make verify-fast` green.
- **Chunked prefill for INT4 KV** (sub-byte gather). New `paged_kv_gather_int4_to_fp16`
  kernel mirrors the existing FP16/FP8/NVFP4 gather variants: symmetric 4-bit
  packed nibbles + per-head FP16 scale (matches `write_kv_cache_int4_kernel`).
  `Engine::supports_chunked_prefill_()` now allows `--kv-int4`. INT4's
  pre-existing long-context quality regression (-22% decode at 20K ctx;
  output degenerates on long prompts) is independent of chunked prefill —
  short-prompt smoke is fine (`The capital of France is` → `Paris.`), long-
  prompt chunked vs single-chunk produce equivalent (equally-degenerate)
  output. Closes the INT4 entry in the "Sub-byte KV cache dtypes" out-of-scope
  list.
- **Chunked prefill for Gemma-4** (SWA + dual head_dim 256/512). `attention_cublas_prefill`'s
  three softmax kernels now accept a `sliding_window` parameter; the mask zeros
  positions outside `[abs_row - sliding_window + 1, abs_row]`. Gemma-4 SWA layers
  route through cuBLAS instead of the naive FP32 workaround when the
  `attn_scores` buffer fits the FP16 S-matrix (naive remains the fallback for
  hd=512 globals at n > attn_scores cap, where `flash_attention_prefill_tc`'s
  ~280 KB static tile overflows sm_120's 100 KB opt-in dynamic smem). `Engine::supports_chunked_prefill_()`
  now allows Gemma-4 (per-layer dispatch covers the heterogeneous shapes
  correctly). Validated on Gemma-4-26B-A4B-it-Q4_K_M: 2823-token chunked
  prefill at 1508 tok/s, decode bit-exact to single-chunk; perf gates green
  (decode -0.35%, prefill +4.65% vs baseline).

## [0.9.0] - 2026-05-10

NVFP4 hits its production stride: prefill goes from 1.2k to 13k tok/s on
Qwen3-Coder-30B-A3B-NVFP4 (×10.5), NVFP4 KV cache lands as a Klasse-A
context unlock (16k → 40k tokens same VRAM, ×3.9 compression), and the
BitDecoding TC paged-decode port reaches FP16 parity. New architectures
(NemotronH hybrid Mamba2+MoE+Attention, multimodal Qwen3.6-VL NVFP4,
zero-config SafeTensors auto-detect, native SentencePiece parser) plus
chunked-prefill correctness across full-attention + hybrid models close
the long-context cliff. Build target moves to `sm_120a` for the full
RTX 5090 feature set; CUDA 13.2 modernization (TMA-style memcpy, `add.f32x2`)
ships. Sixty-plus PRs since v0.8.0.

### Highlights

- **NVFP4 MoE prefill fast-path** (#160) — Qwen3-Coder-30B-A3B-NVFP4 pp512
  1241 → 13046 tok/s (×10.5). Direct-from-NVFP4 grouped GEMM with cached
  problem shapes; previously fell through to dequant→cuBLAS per chunk.
  Cross-model effect: Qwen3.6-NVFP4 / Gemma-4-NVFP4 / Qwen3-30B-A3B-Modelopt
  prefill all double or better.
- **NVFP4 KV cache** (#108, #125) — opt-in `--kv-nvfp4` (or
  `imp.conf:kv_cache.dtype="nvfp4"`); 4 bits/element + per-block scale
  brings 16k → 40k tokens at the same VRAM, ×3.9 compression vs FP16.
  Vectorized PTX `cvt.rn.f16x2.e2m1x2` decode path closed the dequant
  gap (+25.6%, parity with FP16 baseline 147 tok/s on Qwen3-8B Q8).
- **BitDecoding TC paged decode (Phases 0-3)** (#142, #145, #146, #147,
  #148, #149) — WMMA Q.K dot dispatch + block-softmax + FP16 residual
  buffer + multi-seq + splitk path + graph-safe. Final state: parity
  with FP16 baseline (193 vs 193 tok/s) on Qwen3-4B Q8 NVFP4-KV; was
  50 tok/s before. Default opt-in (`bitdecoding_residual_tokens=0`) —
  NVFP4-MoE / dual-head_dim regressions don't justify a flip yet.
- **NemotronH hybrid Mamba2+MoE+Attention NVFP4** (#104, #109) — new
  `NemotronHForCausalLM` arch loads end-to-end; 4-file KV-cache-sizing
  patch fixes the multi-chunk hang on long-context. tg128 42 →
  319 tok/s (+650%) after dynamic NVFP4 MoE reserve sizing — no env
  var needed.
- **`sm_120f` → `sm_120a` build target** (#105) — full RTX 5090 feature
  set (architecture-specific instructions). Historical C7600 `ptxas`
  workaround for `sm_120f` is obsolete on CUDA 13.2.1+.
- **1024 → 4096 prefill cliff closed** (#110) — n≤1024 cap removed,
  S-matrix buffer 256 → 1024 MiB. Qwen3-4B Q8_0 pp=4096 +28%, Qwen3-8B
  Q8_0 +18%, Llama-3.2-3B +24%. Cliff now sits at 4096→4112.
- **Chunked prefill correctness** (#130) — prefill chunks ≥2 now correctly
  read past chunks' K/V from the paged cache. New `paged_kv_gather_*`
  kernels + rectangular `attention_cublas_prefill(q_offset)`. Previously,
  `prefill_chunk_size > 0` produced silently-wrong logits for full-attention
  models and full degeneration for SWA models like Gemma-4.
- **Chunked prefill on hybrid GDN+MoE / Mamba2+MoE archs** (#156) —
  Qwen3.5/3.6, Nemotron-H: prompts where `total_input > effective_chunk`
  were previously rejected with `RequestStatus::CANCELLED`. Two-part fix —
  Mamba2 plain conv1d kernel now reads trailing context from `conv_state`
  at the chunk boundary, and `Engine::supports_chunked_prefill_()` carve-out
  is gated by attention-shape uniformity so HF-loader-populated
  `n_kv_heads_per_layer` arrays (uniform across attention layers) don't
  trip the heterogeneous-shape exclusion.

### Added

- **NVFP4 KV cache storage path** (#108) — `kv_cache.dtype="nvfp4"`,
  paged attention kernel reads block-scaled NVFP4 directly.
- **NemotronH hybrid arch support** (#104, #109) — Mamba2 + MoE + Attention.
  Dynamic NVFP4 MoE reserve replaces the static 1 GiB clamp; reserve is
  computed from the model's attention layout (`per_token_kv × 16K + 256 MiB
  safety, clamped [256 MiB, 1 GiB]`).
- **Native SentencePiece (`.model`) parser** (#128) — drops the Python
  fallback for Mistral-family tokenizers.
- **Multimodal Qwen3.6-VL NVFP4 loader** (#152) — all HF
  `Qwen3.6-NVFP4` repos ship VL/Omni base; loader strips the multimodal
  prefix on text-only weights.
- **Zero-config SafeTensors auto-detect** (#116) — observability + Phase-2
  audit follow-throughs; no `--arch` needed for supported repos.
- **Server: tools + JSON-schema coordination** (#103, #112, #119) —
  preamble pass-through for reasoning models, schema preamble close,
  tool-coordination gaps closed.
- **Server: opt-in `--log-requests` JSONL** (#155) — per-request log line
  written when the flag is on; off by default.
- **Native SentencePiece + AWQ acquisition recipe** (#128, #129) —
  AU2 lit up; AU3 acquisition path documented.
- **Prom/Grafana stack** (#130) — alongside the chunked-prefill fix.
- **`Engine::resolve_prefill_chunk_size_()`** with sentinel `-1` =
  "use per-arch default" (512 for full-attention + FP16/FP8 KV, 0
  otherwise). Default `Config::prefill_chunk_size` flips from `0` to `-1`.
- **`tests/perf_baseline_chunked.json`** — perf baseline for chunked default
  with looser 5%/8% gates.
- **New unit tests**: `test_kv_gather` (3), `test_attention_chunked` (3),
  `test_chunked_prefill` (5 e2e), `test_nvfp4_paged_residual` splitk launch.
- **`make verify-chunked`** target — perf gate against the chunked baseline.
- **CUDA 13.2 modernization** (#131) — `cudaMemcpyWithAttributesAsync`,
  `add.f32x2` two-element FP add intrinsic.
- **`tools/analysis/sass_omma_audit.sh`** + BitDecoding ROI scripts
  (#139) — re-runnable SASS / OMMA audit harness for sm_120a.
- **GHCR release pipeline** (#101) — Docker image published on tagged
  release; manual `workflow_dispatch` available for ad-hoc images.
- **HAS_GPU_RUNNER repo variable** (#127) — CI Test job gates on the
  variable rather than a hard-skip.
- **CI: ccache** + path-aware cache keys + base image bump 13.2.0 →
  13.2.1 (#122, #127). Auto-merge on owner PRs (#123).

### Changed

- **Build target is now `sm_120a`** (#105) — was `sm_120f`. Architecture-specific
  feature set unlocked.
- **`prefill_chunk_size` default sentinel** (#130) — `-1` = per-arch default.
  Single-chunk is now default for SWA / Gemma-4 long-context recall (#114,
  #117); multi-chunk gated on attention-shape uniformity for hybrids.
- **`attention_cublas_prefill` signature** — now takes `int q_offset` (0 =
  square path, byte-equivalent to prior behavior).
- **`causal_softmax_inplace_kernel` / `_fp32_inplace_kernel`** generalized
  to `(S, q_len, kv_len, q_offset, causal)`.
- **`make verify-fast` smoke** pins `--prefill-chunk-size 0` to keep
  `perf_baseline.json` apples-to-apples.
- **imp-cli `--prefill-chunk-size 0`** now correctly forces single-chunk
  (was silently dropped due to `>0` guard).
- **Auto `max_seq_len` for hybrid models** (#157) — corrected; soft-cap
  default lifted to 16K.

### Performance

- **NVFP4 MoE prefill fast-path** (#160) — see Highlights.
- **NVFP4 MoE GrpGemm cache** + opt-in gate+up fusion infrastructure
  (#161) — +4-7% on Qwen3-Coder-30B-A3B-NVFP4 prefill from cache;
  fusion is opt-in and does not yet beat baseline (NVFP4 prefill
  landscape memo + investigation log included).
- **GDN α+β fused GEMV decode** (#153) and **4-way input fusion**
  (`ssm_in` + `gdn_gate` + α + β, #154) — Qwen3.5 / Qwen3.6 GDN decode
  speedups end-to-end.
- **Vectorized FP4 dequant** in paged KV decode (#125) — +25.6% on
  Qwen3-8B-Q8 with `--kv-nvfp4`; closes the gap to FP16.
- **nsys-driven prefill/decode wins + MoE `fp32_down` pre-alloc** (#150,
  #151) — long-context GDN unblocked; cross-model graphs CI gate.
- **Default mem-pool retain + `cudaGraphExecUpdate` re-capture** (#149)
  — chunked prefill on NVFP4 KV cache is now graph-safe.

### Fixed

- **`d_pf_block_tables_` undersize** (#134) — sized from `max_blocks`,
  not `blocks_per_seq`.
- **Gemma-4 FP8-prefill carve-out** (#137) — corrected reason
  (perf, not correctness; cuBLASLt FP8 algo is slower at Gemma-4 shapes,
  not buggy). Output is bit-identical.
- **`llm-compressor` zero / non-finite tensor_scale + input_scale**
  (#113) — defensive guard prevents NVFP4 zero-norm collapse on
  llm-compressor-quantized exports.
- **MoE NVFP4 expert_gemm uses cached buffer** (#115) — non-gated arch
  path was bypassing the contiguous per-expert NVFP4 buffer.
- **Graph-safe `gemm_nvfp4` dequant fallback** (#121) —
  `set_nvfp4_dequant_workspace()` + capture-guard in
  `ensure_dequant_buffer`. Stage-1 of spec-decode/MTP prereqs.
- **Server: drop 512-token prefill chunking default** (#114) — fixed
  Gemma-4 long-context recall.
- **Server: single-chunk prefill default** (#117) — Gemma-4 long-context
  recall pass-rate 4/11 → 11/11 across 128–3000 token doc-length sweep.
- **NVFP4 chunked prefill on KV cache** (#149) — `attn_scores_` buffer
  capacity now sized correctly; chunked decode no longer aborts on
  long context.

### Hardening / audit

- **SafeTensors + NVFP4 audit (F1-F8)** (#126) — F1 model header guard,
  F2 missing-tensor messaging, F3 NVFP4 tensor_scale finite-check,
  F4 `input_scale` visibility, F5 multimodal prefix, F6 `arch_norm_offset`,
  F7 `RMSNorm 1+W`, F8 `A_log → -exp(A_log)` SafeTensors path.
- **CUDA 13.2 modernization** (#131) — `cudaMemcpyWithAttributesAsync`
  replaces stream-attribute dance for L2 streaming hints; `add.f32x2`
  intrinsic for two-element FP adds in inner loops.
- **CMake: drop stale ptxas C7600 diagnostic comment** (#133) — fixed
  on CUDA 13.2.1.

### Repository / tooling

- **Skills committed** (#124) — imp-specific Claude Code skills under
  `.claude/skills/`.
- **Audit cleanup 2026-05-10** (#159) — closed-out memos archived,
  open items consolidated.
- **`docs/performance.md` numbers refreshed** (#158).
- **`docs/roadmap.md` housekeeping** (#132, #135, #136, #138) — CUDA 13.2
  modernization items marked shipped, FP8-KV Gemma-4 entry corrected,
  research-grade KV per-item verdicts.

### Known issues (carry-over)

- **NVFP4 MoE prefill ceiling** at ~16k tok/s warm vs vLLM single-seq
  18.5k = 1.42× gap. Investigation memo + landscape (#161); next steps
  documented under `nvfp4_moe_prefill_landscape_2026_05_10`.
- **Spec-decode / MTP** still off on NVFP4 decode-cache models — graph-safe
  dequant fallback (#121) is Stage-1; full MTP wiring remains a 2-3 week
  item.
- **CUTLASS NVFP4 sm_120 non-determinism** under graph-replay (skip-guard
  retained for `llm-compressor` exports). User-facing output is OK; only
  the graph-replay determinism test trips.
- **Prefill throughput** still shows up to 2.6× variance between
  container restarts due to cuBLAS autotuning. Compare decode-only for
  reliable A/B.

---

Public-release readiness pass: documentation rewrite, hygiene gate
(`scripts/check-release.sh`), removal of dev-internal scratch files,
filename / endpoint corrections.

## [0.8.0] - 2026-05-03

NVFP4-prequant SafeTensors hits production: Mistral-3.2 / Gemma-4 /
Qwen3.6 / Qwen3-Coder all coherent on single-turn, sampling, multi-turn
and short long-context. FP8 KV warmup calibration fixed for Llama and
GDN families. CUDA Graphs lit up for prequant SafeTensors. Forty-plus
PRs since v0.7.0.

### Server + tools (PR #97)

- **Native function calling for Gemma-4 + Qwen3.6** — root-cause was a
  tokenizer bug, not just missing parsers. `encode_spm` / `encode_gpt2` /
  `encode_gemma4` now run a longest-match pre-split pass against
  CONTROL-flagged added tokens before BPE. Multi-character markers like
  `<|tool_call>` (Gemma-4 token id 48) were being BPE'd as raw UTF-8
  bytes — the model never saw the trained marker in its prompt's
  tools-rendering and answered with markdown JSON code blocks instead
  of the native protocol. Fixed: token 48/49 round-trip as their
  assigned id. Added `parse_tool_calls_gemma()` for Gemma's non-JSON
  syntax (`<|tool_call>call:NAME{key:value}<tool_call|>` with
  `<|"|>...<|"|>` string escapes), and extended `parse_tool_calls_chatml()`
  to branch on body shape so Qwen3.6's XML-styled
  `<function=...><parameter=...>` payload parses too. End-to-end
  verified on Gemma-4 Q4_K_M (`finish_reason=tool_calls`, 19 tokens
  completion) and Qwen3.6-NVFP4 (`finish_reason=tool_calls` with
  reasoning_content alongside).
- **Faster cold start (24s → 18s on Qwen3.6 NVFP4)** — skip MTP / vision
  -only SafeTensors shards when neither is wired up (~5s, 2.4 GiB of
  mmap + header parse + page-cache pressure avoided), MAP_POPULATE +
  MADV_WILLNEED on weight mmaps, pinned staging ring 2x64 MiB →
  4x128 MiB, Pass-2 expert upload re-arms cudaMemGetInfo cache so
  per-tensor checked_cuda_malloc skips ~15k sync calls on 128-expert
  MoE, concurrent SafeTensors shard parse (3 shards in parallel
  threads), exposed `name_is_skipped()` to deduplicate the shard-skip
  filter and translate_name's skip rules.
- **Server fixes (Open WebUI on Qwen3.6-NVFP4)** — UTF-8 boundary walk
  in reasoning stream (German umlauts came out as `f��r` because the
  7-byte tail-overlap landed mid-multibyte), drop leaked stop tokens
  (`<|im_end|>` / `<|endoftext|>`) before the `is_last` gate, restrict
  "[Reasoning truncated]" notice to `finish == "length"`, post-`</think>`
  grace 4 → 16 tokens, repetition_penalty default 1.0 → 1.05 to break
  multi-turn loop degeneration, workspace skips FP8 / MXFP4 scratch
  for paths we won't use (~6.4 GiB VRAM headroom on Qwen3.6 NVFP4 GDN).
- **Open WebUI tools enabled in docker-compose** — DuckDuckGo web
  search (no API key), Pyodide code interpreter (browser-side, no
  sandbox service), URL fetch, native function calling toggleable
  per message via the chat-input icons.

### Fixed

- **FP8 KV warmup-calibration bug** (#89) — `Engine::warmup()` ran a forward
  pass with synthetic BOS tokens; the FP8 write path's online calibration
  treated this as the FIRST prefill, locked `kv_scales_[layer]` to a
  too-small absmax, and never recalibrated. Real generation then overflowed
  FP8 dynamic range on Llama-3.2-3B Q8_0 and Qwen3.5-4B GDN Q8_0 (output
  degenerated within ~30 tokens, e.g. `" France, and, 2008, 201, 201, …"`).
  Fix: `Engine::warmup()` drops the `kv_calibrated_` flags at end-of-warmup;
  the FP8 write path promotes the scale monotonically via `std::max` so
  the warmup observation survives if it's already wider, and real prefill
  widens it further when needed. Long generation (100 tokens) on
  Llama-3.2-3B FP8 KV now produces a clean factually-correct list of world
  capitals.
- **NVFP4 prequant CUTLASS prefill cache** (#88) — Phase 0 promotes set
  `Tensor.qtype = NVFP4` directly on the main weight tensors but Phase 3b
  (CUTLASS cache build) only iterated the legacy `wcache_.nvfp4` map.
  Prequant SafeTensors prefill therefore fell through to `gemm_nvfp4`
  dequant→cuBLAS, allocating ~40 MiB FP16 scratch per layer per prefill —
  graph-incompatible AND noisy on SmoothQuant-calibrated Mistral-3.2-NVFP4.
  Phase 0b loop registers all dense + `out_proj_` prequant tensors in
  `cutlass_nvfp4` directly. Standard pp512/tg256 bench post-fix:
  Mistral-3.2-NVFP4 tg 81→101, Qwen3.6-NVFP4 tg 117–142→217,
  Gemma-4-NVFP4 tg 157–180→213, **Qwen3-Coder-30B-A3B-NVFP4 tg 51→272**
  (`--no-cuda-graphs` no longer needed). Mistral-3.2-NVFP4 long-context
  Lorem×11 numerical-hash garbage → coherent text.
- **NVFP4 prequant MoE decode fast-path** (#85) — Qwen3.6-NVFP4 went 8.34 →
  117–142 tok/s (~14–17×); Gemma-4-NVFP4 went ~42 → 157–180 tok/s (~4×).
  Three bugs: `can_decode_fast` whitelist did not include NVFP4-prequant
  models; `cache_moe_native_nvfp4` had to be added to build the contiguous
  per-expert NVFP4 buffer for SafeTensors per-expert layouts; per-layer
  free of per-expert allocations (32 GiB VRAM ceiling on 35B-A3B).
- **Six Qwen3.5/3.6-NVFP4 SafeTensors loader bugs** (#81) blocking coherent
  decode: (1) RMSNorm `1+W` convention now honoured via
  `UploadCtx::arch_norm_offset`, (2) GDN head layout HF-grouped vs
  GGUF-tiled with kernel `grouped_layout` flag, (3) `partial_rotary_factor`
  read from both top-level and nested `rope_parameters`,
  (4) `rope_theta` from nested `rope_parameters.rope_theta`,
  (5) `A_log → -exp(A_log)` transform applied to BF16/F16 SafeTensors path
  only, (6) `fp32_scan` y_buf populated outside `debug_forward`. Per-layer
  correlation vs GGUF Q4_K_M now ≥0.997 across all 40 layers; output
  matches the GGUF oracle for the standard verification prompt.
- **Qwen3.5 GDN Q8_0 α/β qtype mismatch** (#59) — `upload_weight` pre-dequanted
  Q8 → FP16 without updating `qtype`. Dispatcher mis-interpreted bytes →
  state collapse (` my my my…`).
- **MXFP4 GDN-fallback dequant** (#58) — replaced buggy CPU path with GPU kernel.
- **MXFP4 FP16-fallback VRAM oversubscription diagnostic** (#60) — clear error
  message for the Qwen3.5-27B-MXFP4 IMA-on-load case (was silent).
- **Qwen3.5-MXFP4 `A_log` from `blk.X.ssm_dt.weight`** (#61).
- **MoE expert-offload auto-pick** (#54) — defaults try 10 % overhead first
  before falling back to 30 %. Qwen3-Coder-30B Q6_K 77 → 234 tok/s.
- **Mistral-3.2-NVFP4 `use_default_system_prompt`** (#78) — honour the
  tokenizer-config flag and skip the 600-token jinja default system prompt.
  "I am the capital of France?" → "Paris".
- **Server `<channel|>` swallowing answer body on Gemma-4** (#39).
- **Gemma-4 byte-fallback on common names** (#37).
- **Server `reasoning_content` for chat-template-injected `<think>`** (#86).
- **`verify` auto re-execs in `imp:test` when host CMake is missing** (#70) —
  unblocks `make verify-fast` for clean-host workflows.

### Added

- **KV-cache safety default flip** (#51) — default KV dtype is now FP16; FP8
  is opt-in via `--kv-fp8` / `imp.conf:kv_cache.dtype="fp8"`. Fixes Mistral,
  DeepSeek, and Qwen3.5-GDN out of the box on first decode.
- **Auto-deterministic cuBLAS when FP8 KV active** (#52) — pins cuBLAS algo
  selection to avoid quant-dequant noise → softmax NaN. Necessary fix; not
  sufficient for all archs (see docs/roadmap.md "FP8 KV stride bug").
- **CUDA Graph coverage expansion** (#53) — speculative-verify graphs, SigLIP
  vision graph, default mem-pool retain, `cudaGraphExecUpdate` re-capture.
- **SM120 FMHA optimisation pass — Project B Stage 4** (#55, #56) — float4
  tile loads + HW FP4 conversion. **+11–13 % prefill** on Qwen3-4B Q8_0 at
  pp=8192. Stage 5 (`mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`) layouts
  verified byte-exact, integration is the next open Project B item.
- **NVFP4 SafeTensors loader from llm-compressor** (Phase 1, #63; Phase 2
  Item 1 Mistral3, #64; Phase 2 Item 2 Gemma-4 extras + per-row gemv
  bypass, #65). Mistral3-NVFP4 decode tg ≈ 81 tok/s post Phase 2 Item 1.
  Gemma-4-NVFP4 (llm-compressor) decodes coherent end-to-end at ~34 tok/s
  with default flags after #65 routes M>1 expert GEMV through `gemm_nvfp4`
  dequant→cuBLAS (legacy serial path's per-row `gemv_nvfp4_kpar` loop
  produced wrong output at Gemma-4 expert dimensions; M=1 decode path is
  unchanged).
- **Qwen3.6-NVFP4 SafeTensors plumbing** (Phase 1 #71) — load-only.
  Decode lit up later via #85.
- **JSON config plumbing** (#74, #77) — `generation_config.json` sampling
  defaults, `special_tokens_map.json`, Mistral V3 tokenizer-config flags.
- **Tokenizer-config `use_default_system_prompt=false` honoured** (#78) — see
  Fixed.
- **Type-system + config refactor** (#72) — unified `QType`, `Tensor` sidecars,
  `imp.conf` (TOML, ~50 former `IMP_*` env vars now keys). New top-level
  `imp.conf.example`. CLI `--set kv_cache.dtype=fp8` for per-run overrides.
- **NVFP4 collapsed load-time scratch** (#73) — single `Model` map.
- **FP32 attention S-matrix + Qwen3.5 QK-norm split** (#66) — improves
  numerical headroom on long-context attention.
- **Diagnostic env vars for NVFP4 + attention** (#79) — reproducer support
  for the long-context NVFP4 bug.
- **Anthropic `/v1/messages` endpoint** (Phase 1 non-streaming #35,
  Phase 2 streaming #36) — synthetic SSE stream over the OpenAI handler.
- **Storage-planner enumerates shared-expert FFN + top-level embeddings/LM
  head** (#38, #40) — fixes silent miss in MoE memory budget.
- **Strengthened GDN coherence test** (#48) — detects recurrent-state collapse.
- **Strengthened Gemma-4 NVFP4 e2e gate** (#68) — Paris coherence assertion.
- **Synthetic `gemv_kpar` M>1 per-row-loop bug repro test** (#69).
- **Split `imp-tests` into 8 per-module binaries** (#57) — speeds up filtered
  test runs.
- **`tools/analysis/` PTX survey scripts** (#67) — re-runnable cvt / MMA /
  async-TMA / atomics / SFU / cluster surveys for `sm_120f` after CUDA upgrades.

### Changed

- **Default KV dtype is FP16** (#51) — see Added. Was implicit auto-FP8.
- **`imp.conf` is now the configuration interface** (#72) — ~50 `IMP_*` env
  vars retired; sectioned TOML keys (`runtime.cuda_graphs`, `kv_cache.dtype`,
  `attention.fp8_fmha`, …). Loading precedence: `--config` → `$IMP_CONFIG` →
  `./imp.conf` → `~/.config/imp/imp.conf` → embedded defaults. CLI overrides
  via `--set section.key=value`.

### Repository / build hygiene

- **Untracked `build-docker/` and `bringup_artifacts/`** (#82) — debug dumps
  no longer in VCS.
- **Removed tracked binaries + stale Gemma-4 debug snapshots** (#83).
- **Removed obsolete top-level docs + stale benchmarks dir** (#84).

### Performance baseline refresh (2026-04-29)

`tests/perf_baseline.json` refreshed (#80). Numbers reflect a
RelWithDebInfo build with full GPU boost engaged (P1, 2880 MHz, 456 W).
Refresh via `scripts/gen_perf_baseline.sh`.

### Known issues (carry-over)

- **FP8 KV cache** still breaks Llama-3.2 / Mistral-Small-3.1 / DeepSeek-R1-Distill
  out of the box even with the determ-cuBLAS gate. Default is FP16; opt-in
  per model after testing. See docs/roadmap.md.
- **NVFP4 long-context regression** on Mistral-3.2-NVFP4 at ~500+ raw tokens
  is **partially resolved** by PR #88 (CUTLASS NVFP4×NVFP4 prefill);
  numerical-hash kernel garbage is gone. Residual model-behaviour issue
  on long English prose remains. PR #79 ships diagnostics; PR #78 ships
  the `use_default_system_prompt=false` workaround.
- **CUDA graphs are now safe by default for prequant SafeTensors** (PR #88).
  The previous `--no-cuda-graphs` requirement on Qwen3-Coder-30B-A3B-NVFP4
  is **removed**: the dequant→cuBLAS fallback that allocated FP16 scratch
  per prefill (graph-incompatible) doesn't fire anymore. Decode jumped 51 →
  272 tok/s on Qwen3-Coder NVFP4 by enabling graphs.
- **Prefill throughput** shows up to 2.6× variance between container restarts
  due to cuBLAS autotuning. Compare decode-only for reliable A/B.

---

## [0.7.0] - 2026-04-23

Big correctness + platform release: the long-context dispatch cliff is gone,
Gemma-4 and the Qwen 3.5/3.6 GDN family now produce clean output on Blackwell,
CUDA 13.2.1 with stream priorities and mem-sync domains is live, and the
StreamingLLM smart-KV mode is available.

### Fixed

- **FP8 FMHA long-context cliff at n>1024** (#33) — `fmha_sm120_fp8_kernel` placed
  `S_tile` only `Bkv*head_dim` bytes past `KV_fp8`, but the K-as-FP8 / V-as-half
  slot is reserved for the full `Bkv*head_dim*sizeof(half)` bytes. V row `Bkv/2+`
  overwrote the P values the PV MMA was about to read → NaN on every attention
  layer above n=1024 (cuBLAS dispatch boundary). Invisible to prior benchmarks
  because `pp=512/1024` always stayed on cuBLAS and decode uses paged attention.
  All tested models (Qwen3-4B/8B, Qwen3.5-4B/9B, Llama-3.2-3B, Mistral-24B,
  Qwen3-32B) now coherent at n≥1025.
- **Qwen 3.5/3.6 GDN fused-kernel launch_bounds** (#30) — `__launch_bounds__(HD, 2)`
  miscompiled at HD=128 (register pressure with `H_reg[128]` and 2 blocks/SM).
  Dropping to `(HD, 1)` fixes Qwen3.5-4B/9B Q8_0 coherence and improves
  Qwen 3.6 tg256 from 36 → 57 tok/s.
- **Qwen 3.5 partial-RoPE pair offset** (#30) — sister fix: partial-RoPE pair index
  was `pair_idx + head_dim/2` instead of `pair_idx + rope_pairs`. Both fixes are
  needed for correct output.
- **Qwen 3.6 h_state FP32 preservation** (#28) — engine auto-downgraded
  `ssm_state_dtype` from FP32 to FP16 for all SSM models, but the GDN scan writes
  FP32. Each layer's scan overflowed 1 MB into the next layer's `conv_state` /
  `h_state` region, producing NaN at L38 on Qwen 3.6. Also switched L2 norm to
  PyTorch-style `rsqrtf(fmaxf(sum_sq, 1e-12))` for near-zero-head stability.
- **Gemma-4 SWA long-context degeneration** (#21) — fixed regression where prompts
  >1024 tokens on global layers emitted garbage via the broken FMHA fallback.
- **Gemma-4 rope_freqs** (#20) — per-layer `rope_freqs` were ignored on global layers;
  llama.cpp uses them with `n_rot=hd`. Fix cuts L13/L14 drift from 11-15 % to <2 %.
- **Gemma-4 host-resident MoE** (e879bcd) — fused gate_up split on host, batch buffer
  preserved. Fixes silent output corruption when experts are CPU-offloaded.
- **Gemma-4 Q4_K_M CUDA-graph decode** (873f1d7) — split-K pipeline kernel only issued
  one 16 B `cp.async` per load, missing half the data for HEAD_DIM=512 on global
  layers. Loops `cp_async_ca_16` in 8-half chunks. tg256 Q4_K_M 55 → 183 tok/s
  (×1.21 vs llama.cpp 151). Also +12 % on Qwen3-4B MXFP4.
- **Qwen 3.5 GDN L2-window CUDA errors** (275807c) — `cudaStreamAttributeAccessPolicyWindow`
  with `num_bytes > cudaDevAttrMaxAccessPolicyWindowSize` (128 MiB on RTX 5090)
  silently poisoned the stream. Clamped in `set_l2_streaming` +
  `set_l2_persist_kv`.
- **Gemma-4 ≥3120 token limit was VRAM, not architecture** (2026-04-20) — default
  ceiling lifted 3120 → ~7881 tok; `--min-kv-tokens 14000` reaches 11242 tok. Root
  cause: max_seq_len ordering bug + defensive 80 % cap.
- **Gemma-4 decode FP32 router + half rope_dim on global layers** (5a1e844) —
  MoE routing FP16 accumulation caused expert mis-pick at L29. Also fixed full
  rope rotation being applied instead of the partial-RoPE schedule.
- **Async decode loop correctness** (3b766bc) — four latent bugs in the async
  decode path that only surfaced with real long generations.

### Added

- **StreamingLLM smart KV cache** (#26) — attention sinks + sliding window; keeps
  long-conversation coherence without unbounded VRAM growth.
- **Weight-storage refactor** (#27) — `TensorKind` + `StoragePlanner` +
  `gemm_dispatch` (phases 0-5). Collapses 21-param dispatch to 5 params, legacy
  overload retired, `beta=1` supported. No functional change, -1200 LoC
  churn absorbed cleanly.
- **CUTLASS 3.x NVFP4 Grouped GEMM scaffold** (#22) — path for sm_100+ FP4 grouped
  with fused MoE quantize; default ON for all batch sizes after the gate+up
  shared-quantize opt (decode 51 > legacy 37 on Qwen3-Coder-30B-A3B NVFP4).
- **CUDA 13.2.1 base images** (#16) + **stream priorities, mem-sync domains,
  cluster spread** (#17).
- **Qwen 3.6 `ModelArch::QWEN36_MOE` scaffold** (#23) — GDN + MoE hybrid.
- **GDN reference infrastructure + Qwen 3.6 cache preservation** (#25) — shared
  helpers for GDN debug dumps, multi-turn state preservation.
- **IMP_DEBUG_RAW meta-flag** (#29) — single switch that turns off CUDA graphs,
  PDL, host-MoE, and other sources of non-determinism for reference-diff runs.
- **IMP_EXPERT_OVERHEAD_PCT hint** (#32) — runtime emits the right env-var
  suggestion when it disables CUDA graphs due to insufficient VRAM headroom.
- **IMP_GEMMA4_CUDA_GRAPHS, IMP_FORCE_HOST_EXPERTS=N, IMP_NO_MMVQ,
  IMP_NO_MMVQ_Q8_0** — debug overrides surfaced during the Gemma-4 stabilization.
- **`tools/analysis/layer_diff.py`** (#20) — .npy-based per-layer tensor diff
  between imp and llama.cpp for drift analysis.
- **CUDA graph diagnostics** (#11-#14) — `IMP_GRAPH_DIAG` / `IMP_GRAPH_DUMP`,
  device-side stop-reason trace in `post_decode_step_kernel`,
  `cudaDeviceGraphMemTrim` on capture lifecycle.
- **Regression tests** — Gemma-4 e2e suite (7633e1a), `Gemma4GraphsTest` for
  the CUDA graph path (dd10244), `FmhaFP8Test.Qwen35LikeHD256_GQA41_SeqMultiTile`
  for the long-context fix (#33).
- **cpp-httplib 0.40.0 → 0.42.0** (4295c05).

### Changed

- `imp_version()` and `project(… VERSION)` now return **0.7.0**.
- Gemma-4 CUDA graphs enabled by default for decode fast-path (no more D2H in
  the routing path on that arch).
- Gemma-4 benchmark docs refreshed (#24) with the quality caveat for Q4_K_M on
  complex code-gen prompts — Q5_K_M / Q8_0 recommended when output quality
  matters.

### Deprecated / Removed

- Nothing user-visible. Internal legacy `_inline_quant` GEMV and stale
  `gen_reference` stub removed (e558f10).

### Known Issues

- Qwen3-Coder-30B-A3B NVFP4: `--no-cuda-graphs` still required for coherence on
  the MoE routing path (general-MoE D2H routing memcpy is incompatible with
  capture; Gemma-4 is excepted via the decode fast-path).
- Prefill throughput shows up to 2.6× variance between container restarts due
  to cuBLAS autotuning algorithm selection — compare decode-only for reliable
  A/B testing.
- FP8 FMHA path (n>1024 prefill) is ~30 % slower than cuBLAS at the dispatch
  boundary on small dense models (Qwen3-4B: 27 k → 19 k tok/s at 1024→2048).
  Output is correct; optimization is future work.
- MXFP4 GGUFs use the imp-proprietary tensor-type 31, which llama.cpp reads as
  the removed `Q4_0_4_4`. Cross-tool perplexity comparison is therefore not
  possible without a standard-format MXFP4 export.

---

## [0.6] - 2026-04

Previous release. Highlights that shipped under this tag:

- NVIDIA Model Optimizer NVFP4 prequant SafeTensors loading
  (Qwen3-Coder-30B-A3B-FP4 verified).
- imp-server SafeTensors support + `resolve_model_auto()` format detection.
- Chat-template array format (HuggingFace convention).
- Jinja2 macro support — fixed Qwen 3.5 "ignores prompt" symptom.

## [0.5.1], [0.4.1], [0.4], [0.2]

Pre-0.6 tags retained for reference. See `git log` for details.
