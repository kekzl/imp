# Test Trustworthiness Audit — Phase 1 (Gap Analysis)

Date: 2026-06-04 · Commit base: post-#523 main · Author: Claude Code (session audit)

**Charter:** not more tests — tests that catch real bugs. This audit inventories
what exists, classifies it by evidentiary strength, and ranks the riskiest
untested paths. No test code was written for this phase.

**Why this audit is credible right now:** in the 24h before it was written, the
suite was green while (a) the FA2/fp8 prefill attention made **9 of 12 zoo
models prompt-blind** (13/13 FA2 parity tests passed — synthetic ±0.12 data,
5% tolerance, no accumulation), (b) Nemotron had been **positionally blind
since integration** (RoPE applied to NoPE attention — no positional-correctness
test exists), (c) the NVFP4 mode-2 quantizer collapsed on Gemma outlier
weights (only round-trip-via-own-code tests existed), and (d) three stacked
think-budget bugs lived in a fully untested state machine. Every one of these
is a "wrong from day one or day N, never caught" class — exactly what
tautological tests cannot see.

---

## 1. Coverage inventory per subsystem

Status legend: **DIRECT** (unit-tested in `tests/`), **E2E-only** (only
exercised when a local model + GPU is present — never in CI, which has no GPU
runner), **UNTESTED**.

| Subsystem | Tested (DIRECT) | E2E-only | Untested (highest-value gaps) |
|---|---|---|---|
| core (tensor/config/logging) | tensor shapes/strides, config parse, tensor-kind tables (all structural) | — | `Tensor::reshape`/`slice` numerics never validated against data movement |
| compute / attention | FMHA fp8 + FA2 + MXFP4 + TC vs CPU ref (weak data, see §2); paged FP16/FP8 partial | paged decode in real decode loop | **paged INT4/INT8 kernels, attention_blackwell fallback, attention_dispatch routing, cuBLAS-vs-FMHA cross-path agreement** |
| compute / other kernels | RoPE, RMSNorm, softmax, reduce, activation, hadamard vs CPU fp32 (good oracles, synthetic data) | — | **rope_attn_disabled (NoPE) branch**, sampling correctness (only "token in vocab") |
| quant | NVFP4 E2M1/E4M3 host-decode refs (the one true class-A anchor), grouped-NVFP4-smallM vs CPU | mode-2 from-scratch conversion | **Q4_K/Q6_K/Q8_0 dequant vs reference; quantizer outlier behavior (tensor_scale/micro-scale floor); quantized GEMM vs fp16 GEMM tolerance bands; MMVQ/dp4a numerics** |
| memory / KV | FP8 KV calibration vs CPU ref; block alloc/free patterns (structural) | paged eviction in real runs | **KV equivalence (fresh vs prefix-cache-hit logits), eviction+refill output stability, INT8/INT4/NVFP4-KV accuracy bands, device/vram allocator, storage_planner** |
| model / loaders / tokenizer | tokenizer BPE/SPM merge chains, Jinja2 engine (good), chat templates, loader field parsing (structural) | weight_upload, weight_map | **GGUF fault injection (truncated/malformed → clean error, not UB); tokenizer unicode round-trip (the NUL-byte class); hf_hub** |
| exec | a few kernel-dispatch smokes | pre_dequant phases 1–4, executor forward, MoE dispatch | **pre_dequant phase 3 (mode-2) in isolation; executor_ffn; fused QKV paths** |
| runtime | scheduler/batch-builder (structural), green_ctx alloc patterns, json/schema constrainers (now whole-token) | engine init, CUDA graphs, AsyncGraphLoop in real decode | **engine_sampling_stop (think budget/`started_in_think`/`think_exit_idx`/text-tail), warmup token-type detection (CONTROL vs USER_DEFINED), constraint preamble gate, ConditionalRunner, request lifecycle** |
| vision | — | gemma-3/4 VL via manual runs | **everything** (encoder, image processor, projector tail) |
| api / server | C-API generate parity (stub-backed), engine relaunch (VRAM + re-init) | — | **handlers.cpp streaming text pipeline (the max_stop_len/NUL class), reasoning split, anthropic transform, stop handling — zero unit tests** |
| api / HTTP contract (Python) | schema/SSE/tool/error shape vs mock server (CPU-capable) | golden runs vs real model | streaming-reassembles-to-non-streaming equality; logprobs shape against real engine |

Subsystem-level aggregate (component count basis): ~45% has some direct test,
~30% only E2E (≈ never in CI), ~25% untested.

## 2. Classification of existing tests

~165 test files (143 C++/CUDA across 8 binaries, ~11 Python API, ~11
standalone/bench). Full per-file table in the appendix of the inventory run;
the distribution:

| Class | Definition | Share | Examples |
|---|---|---|---|
| **A — independent reference** | external/closed-form/CPU-fp32 oracle computed independently | **~13%** | RoPE, RMSNorm, softmax, GEMM-vs-CPU-loop, Jinja2, tokenizer merges, MoE top-k, FP8-KV calibrate, NVFP4 E2M1/E4M3 host refs |
| **A− — real oracle, weak data/tolerance** | CPU reference exists but synthetic benign inputs and/or unjustified tolerance | **~19%** | all FMHA variants (`0.02f*((i*7+3)%13-6)` patterned fills, 5% rel tol), paged attention, quant integration |
| **B — tautological** | imp compared against imp (other kernel, own round-trip, prior output) | **~39%** | CUTLASS-vs-CUTLASS, NVFP4 round-trip via own quant+dequant, KV write/read-back, embedding lookup, forward-pass not-NaN, weight dispatch |
| **C — structural/smoke** | shapes, parse-ok, no-crash, HTTP 200 | **~28%** | loaders, allocators, registry lookups, mock-server API tests, benches |

**The headline number: ~13% of the suite can catch a kernel that was wrong
from day one.** The A− band is the most dangerous: it *looks* like class A
(there is a real reference!) and passes review, but the data regime hides the
failure modes that matter. Proof: `FmhaFA2Test` 13/13 green during the FA2
catastrophe; adding the *engine-realistic shapes* still passed; only realistic
*data distribution + per-layer accumulation* would have failed.

Notable specifics:
- `tests/golden/` values lack committed generators (magic constants); there is
  no `tests/refs/` with reproducible Python reference scripts.
- Python API tests run against `mock_server.py` — good for CPU CI, but the
  mock encodes the *documented* contract, not the real handler code; the
  NUL-terminator bug lived in exactly that gap.
- No compute-sanitizer integration anywhere (and note: **compute-sanitizer
  cannot attach on this WSL2 host** — "WDDM debugger interface" failure,
  verified 2026-06-04. The CTest target must exist but run on a native-Linux
  / CI GPU runner).
- No spec-decode exactness test (MTP test is smoke-only), no KV-equivalence
  test, no MoE-determinism probe, no GGUF fault injection, no unicode
  round-trip.

## 3. Top-10 risk ranking — P(silent bug) × blast radius

Ranked with the evidence of this week's real failures. "Blast" = how much of
the product is wrong when this is wrong.

| # | Untested path | P(silent) | Blast | Evidence / rationale |
|---|---|---|---|---|
| 1 | **Prefill attention numerics under realistic data + cross-path agreement** (FA2 ≡ fp8-FMHA ≡ cuBLAS/materialized ≡ fp32 ref, real magnitude/distribution, multi-layer accumulation budget) | proven | every token of every hd=128 model | FA2 e4m3 noise: 9/12 models prompt-blind, all parity tests green. Cross-path agreement is the killer assert: the engine has 3 prefill paths that MUST agree — nothing checks it. |
| 2 | **NVFP4 quantize pipeline vs independent reference on adversarial weights** (outliers, per-tensor scale floor; mode-2 from-scratch vs from-FP16) | proven | all NVFP4 decode (the product's core) | Gemma outlier collapse → NaN logits → `<pad>` argmax. Only round-trip-via-own-code existed. Needs quantize→dequant→GEMV vs fp16 with N(0,1)+outlier fixtures. |
| 3 | **E2E greedy regression locks vs llama.cpp/HF per architecture** (fixed prompt+seed → frozen tokens, sequence first verified externally, then locked) | proven | whole model families, silently | Would have caught: Nemotron NoPE (positionally blind since integration), Phi-4 RoPE-NeoX, FA2 regression, gemma mode-2. The single highest-leverage addition. |
| 4 | **Engine think/stop state machine** (`engine_sampling_stop`: budget recount, `started_in_think`, text-tail `</think>`, grace period; warmup token-type CONTROL vs USER_DEFINED) | proven | every reasoning-model request (the dominant zoo) | 3 stacked bugs found in one evening, all in untested code. Pure host logic — cheap CPU tests. |
| 5 | **Server streaming text pipeline** (per-token flush math, stop-sequence holdback, UTF-8 boundaries, SSE reassembly == non-streaming result, reasoning split) | proven | every streamed response | `max_stop_len=0` → NUL terminator in every delta; non-stream/stream divergence. Mock-server tests structurally green throughout. |
| 6 | **Paged decode attention quant variants** (FP8/INT4/INT8 paged kernels vs fp32 ref; per-KV-dtype tolerance bands) | high | every decoded token when those KV modes are on | Decode twin of #1; zero numeric oracle today. FP8-KV is one config flag away from default. |
| 7 | **KV-cache equivalence** (fresh prefix vs prefix-cache hit within tolerance; eviction+refill output-stable) | high | correctness of the whole caching feature | Prefix cache ships off-by-default *because* its determinism is unvalidated — the test IS the enabler. |
| 8 | **Q4_K/Q6_K/Q8_0 dequant + MMVQ/dp4a GEMM vs reference** | medium-high | all GGUF serving | Everything downstream consumes these; only INT4/INT8 toy refs exist. llama.cpp's dequant is the natural oracle. |
| 9 | **Spec-decode exactness** (spec greedy ≡ non-spec greedy, token-for-token) | medium (parked feature) | silent wrong outputs if ever enabled | Acceptance is mathematically exact; current test asserts "token in vocab". An equality assert costs nothing and guards re-activation. |
| 10 | **GGUF/tokenizer robustness** (header fault injection → clean error; unicode/special-token encode∘decode round-trip incl. NUL, ZWJ, surrogates) | medium | loader UB / server output corruption | The NUL-byte class came from exactly this blind spot (special-token rendering paths). |

Honorable mentions (below top-10 but cheap): MoE determinism probe (quantify
the atomics spread, N=20 runs, assert bounded; bit-exact assert once a
deterministic mode exists), `attention_dispatch` routing table test (which
path serves which (hd, seq, dtype) — the #493 regression was a routing
change), allocator/storage_planner accounting, vision pipeline golden.

## 4. Structural fixes the suite needs (Phase 2 preamble)

1. **`tests/refs/` with committed Python generators** (PyTorch/transformers
   fp32 + llama.cpp harness) producing versioned `.npz`/`.json` goldens —
   no magic constants. Every class-A test states source + tolerance + why.
2. **Tolerance policy**: fp16 ≤ 1e-2 rel (justify per-op), fp8 ≤ 5e-2 with
   accumulation budget per layer count, NVFP4 ≤ 1e-1 single-op but **E2E
   token-level locks pick up the slack** — single-op tolerances provably do
   not compose, so end-to-end locks are mandatory, not optional.
3. **CTest split**: `unit` (CPU, no GPU — parsers, tokenizer, server text
   pipeline, constraint FSMs, think state machine, contract tests against
   real handlers not just mock) vs `gpu` (sm_120a-gated). New GitHub workflow:
   build + CPU-unit on every PR (exists partially today — extend); GPU lane
   documented for a self-hosted runner.
4. **compute-sanitizer CTest target** (memcheck/racecheck/initcheck/synccheck)
   — must run on native Linux/CI GPU runner; **documented as non-functional on
   the WSL2 dev host** (verified). Until a runner exists, a manual
   `make sanitize` target with instructions.
5. **Realistic-data fixtures**: real Q/K/V slices (committed, small) captured
   from Qwen3-8B layer 0 + synthetic outlier-augmented weights as standard
   kernel-test inputs, replacing the periodic `%13` fills.

## 5. Proposed Phase-2 implementation order

1. `tests/refs/` scaffolding + tolerance policy doc (enables everything).
2. Risk #1: attention cross-path agreement + fp32 ref under realistic data.
3. Risk #2: NVFP4 outlier fixtures (quantize→dequant→GEMV vs fp16).
4. Risk #3: E2E greedy locks (llama.cpp-verified, then frozen) — Qwen3-8B,
   Qwen3-MoE, Gemma-3, Nemotron-H, Phi-4, LLaMA; CI-skipped without model,
   mandatory in local verify.
5. Risk #4+#5 CPU tests (think state machine, server streaming pipeline) →
   straight into the CI `unit` lane.
6. Risks #6–#10 in order; sanitizer target + CI split alongside.

— End of Phase 1. No test code written; awaiting owner review before Phase 2.
