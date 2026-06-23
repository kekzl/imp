# Changelog

All notable changes since v0.6. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
