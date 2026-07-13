# imp — Goal Definition

## Mission

**imp shall be the best single-GPU *agentic* AI inference engine on NVIDIA RTX 5090 (sm_120) — the fastest, most capable backend for running coding agents, tool-using assistants and reasoning loops on one workstation card.**

The foundation is unchanged and non-negotiable: **the fastest single-stream (batch=1) decode of any engine on this chip**, across the architectures we support. Low single-stream latency is what makes an agent feel instant — it is the definition of "best" below, and nothing agentic is allowed to erode it.

What "agentic" adds *on top of* raw speed (specified in "Agentic surface" below, gated in the release bar):

- **Tool calling & constrained output** that never breaks the JSON / schema contract.
- **Long-context loops** — large agent transcripts, prefix-cached multi-turn, KV budgets sized for 64k+ working context.
- **Reasoning / thinking** as a first-class, separable output channel.
- **Moderate concurrency** — an agent harness fans out parallel sub-agents; serving *tens* of concurrent requests on one GPU is a goal, not an afterthought (see "What imp is NOT").
- **Reliability under sustained agentic load** — clean cancel, per-request metrics, no host syncs on the hot path.

Single-user-class hardware, single GPU, latency-first. Not a datacenter throughput competitor to vLLM/SGLang. The agentic weapon for the workstation.

---

## Definition of "best"

Ranked, non-negotiable order:

1. **Decode throughput (tok/s) at batch=1** on RTX 5090 — primary metric. Must lead llama.cpp, vLLM, SGLang, ExLlamaV3, MLC-LLM on every supported architecture and quant combination we ship.
2. **Prefill throughput (tok/s) at batch=1** on RTX 5090 — must match or beat llama.cpp on dense **NVFP4/SafeTensors** (GGUF prefill is best-effort, ceiling is architectural — see release bar 3); must close the MoE gap to vLLM (re-measured ~1.4× single-seq 2026-05-31; remaining levers: prefill attention + grouped-GEMM occupancy scheduler #558).
3. **Time-to-first-token (TTFT)** at realistic prompt lengths (512, 2048, 8192, 32k) — must be competitive with vLLM despite our batch=1 focus.
4. **VRAM efficiency** — must fit larger models than competitors at equivalent quality (NVFP4, FP8 KV, paged cache). 32 GB on a 5090 should serve everything up to ~70B dense at usable quality.
5. **Quality** — perplexity and downstream eval parity with llama.cpp at the same quant. No silent quality regressions for speed.

These five define raw **engine** quality and are anchored on single-stream latency. Agentic **capability** — the other half of the mission — is specified in "Agentic surface" below and gated in the release bar. Concurrent throughput for agent fan-out is a goal (see "What imp is NOT"), but it is a *secondary* metric: it may never be bought by regressing single-stream decode.

Anything below this bar is a bug.

---

## Target hardware (in priority order)

1. **RTX 5090** (sm_120, GB202, 32 GB GDDR7) — the hero target. Every optimization decision is made for this chip first.
2. **RTX PRO 6000 Blackwell** (sm_120, 96 GB) — same arch, more VRAM. Free win if 5090 is fast.
3. **Consumer Blackwell siblings** (5080, 5070 Ti, etc.) — same sm_120, covered by the `compute_120f` PTX fallback in the fatbin, lower priority for tuning.

Everything else is **unsupported by design**. There is no Hopper (sm_90a), Ada, Ampere, or datacenter-Blackwell (sm_100) path in the tree — the engine is built against `sm_120a` exclusively (see README "Consumer Blackwell only"). Not a goal.

---

## Target models (hero set)

These models must be **best-in-class on 5090**. No exceptions, no excuses.
Hero status requires staged local weights, a green degeneration battery, and
decode numbers in `BENCHMARKS.md` (realigned 2026-06-06, issues #549/#550 —
heroes that had never produced a local number moved to the extended set
below; gpt-oss-20b closed the last hero gap on 2026-06-06, #547 / PRs
#572–#574):

| Model | Quant | Why |
|---|---|---|
| Qwen3-4B / 8B | Q8_0 (8B also NVFP4) | Daily driver dense, fast iteration |
| Qwen3-14B | Q6_K, NVFP4 | Sweet spot for 5090 — the north-star model |
| Qwen3-Coder-30B-A3B | NVFP4 | Hero MoE |
| Qwen3.6-35B-A3B | NVFP4 | Hybrid (GDN + MoE) daily driver, MTP head |
| Gemma-4 26B-A4B (text + vision) | NVFP4 | Multimodal + MoE hero |
| Nemotron-H | NVFP4 | Hybrid (Mamba2 + Attn + MoE) flagship |
| gpt-oss-20b | MXFP4 (SafeTensors) | First-class MXFP4 path — **supported since 2026-06-06** (#547, PRs #572–#574): experts converted to NVFP4 at load, Harmony channels, tg ≈ 315–345, pp512 ≈ 16–19k |

If a hero model regresses against any competitor on the primary metric, that is a release blocker.

**Extended set** — supported architectures, validated opportunistically, NOT
release-blocking (promotion back to hero requires staged weights + battery +
benchmark numbers):

| Model | State (2026-06-06) |
|---|---|
| DeepSeek-R1-Distill-7B/14B | Never benched locally. Arch is Qwen2/Llama (covered). |
| DeepSeek-V2-Lite (MLA) | **Supported** — first Multi-head Latent Attention arch (#802 materialized Stage A, #803 absorbed latent-KV decode, opt-in). Validated locally at bf16 (28 GB, experts host-offloaded → graphs disabled); same-corpus PPL within ~3% of HF bf16 (imp 6.43 vs HF 6.25, 534-tok) after the 2026-07-07 YaRN rope-mscale fix. Real V2/V3/R1 MLA checkpoints now load via this path. |
| Gemma-3 27B | 12B Q4_K_M + 4B-VL validated locally; 27B never staged. |
| Phi-4 14B | NVFP4 (reasoning-plus) validated locally; GGUF Q6_K never staged. |
| Mixtral 8x7B | Arch in the enum, chat-template test only; never staged. |

---

## What "best on 5090" requires

Concrete technical commitments — these are means, not ends, but progress on the mission is measured against them:

### Compute path
- **NVFP4 must remain the default fast path** on sm_120. FP16/BF16 paths exist for correctness, never as the recommended config. The Blackwell consumer FP16/BF16 throughput nerf is a hardware fact; NVFP4 is the way out and we lean into it harder than anyone.
- **MXFP4 FMHA** (already novel — first such impl) must stay ahead and expand to more shapes. The +6.7–7.9% over FP8 FMHA on Qwen3 is the baseline, not the ceiling.
- **FA2 prefill family** (register-resident FA2 default-on, FP16-QK FA2 for short prefill, FP8 FMHA fallback) is the prefill attention path on sm_120 — FP8×FP8 cuBLAS prefill is `NOT_SUPPORTED` on this chip, so FA2 carries the commitment. Keep it ahead and expand shapes (hd≠128 declines today).
- **Paged decode attention** (FP16/FP8/INT8/INT4/NVFP4 KV) + the CUDA-graph decode loop is the decode workhorse on sm_120. Tune for every hero model's head dim.
- **Grouped GEMM dequant for MoE prefill** — gap to vLLM single-seq has substantially closed. Qwen3-Coder-30B-A3B-NVFP4 (cold-median 5×5 trials, 15 s cooldown, 2026-05-23):
  - **pp512 = 17,521 tok/s** (σ wide: 16k-19k spread across trials — cuBLAS-algo-state still drifts at this kernel size despite cold-median methodology)
  - **pp2048 = 18,573 tok/s** (σ tight: 18.3k-18.9k = 3 % spread — exceeds vLLM 0.20.2's pp512 single-seq number)
  - **tg128 @ ctx=512 = 273.24 tok/s** (σ 0.09 — rock solid)
  - **tg128 @ ctx=2048 = 268.46 tok/s** (σ 0.10)

  Compared to vLLM 0.20.2 single-seq pp512 = 18,500 tok/s: **gap = 1.056× (5.6 %)**, was 1.14-1.32× pre-#374. The MoE-prefill skip-gather (#374) + downstream improvements moved us from 14,562 → 17,521 pp512 (+20.3 %). pp2048 actually exceeds vLLM's single-seq pp512 baseline. vLLM's 25.5 k multi-seq is continuous-batching mode, not a fair single-seq comparison — imp is batch=1 by design.

  *Update 2026-05-31/06-06:* the fresh audit re-measured the real single-seq gap at **~1.4×** (the 1.056× snapshot above sat inside the 2.6× cuBLAS-restart variance band). The grouped-GEMM compute itself is near roofline (the "20× grouped-GEMM gap" premise is refuted); the remaining halves of the gap are **prefill attention** (FA2 partial) and the **grouped-GEMM launch/occupancy** (single wave at ~24% occupancy, latency-bound at small per-expert M — persistent/stream-K scheduler, tracked in #558).

### Memory
- **Paged KV cache** (block 16) with LRU, prefix caching — keep and extend.
- **FP8 / INT8 / NVFP4 KV cache** — NVFP4 decode cache is already shipping; expand to prefill where quality allows.
- ~~TurboQuant integration~~ — **retired 2026-05-17** (dead-ends archive); the CLI flags survive only as deprecated aliases. KV squeezing beyond the shipped FP8/INT8/NVFP4 cache options is not a tracked commitment.

### Latency
- **CUDA Graphs everywhere on decode** — already done, never regress.
- **PDL (Programmatic Dependent Launch)** — keep aggressive.
- ~~TurboDraft (L2-resident speculative decoder)~~ — **parked.** The "~25-30% K=1 acceptance" from the 2026-05-30 diagnosis was a kernel bug, not a precision wall: the MTP attn-output gate used silu where the head expects sigmoid; correcting it (#804) lifts acceptance to 85%+ on Qwen3.6. Spec-decode *generation* via this head is still parked for a different reason — the GDN-hybrid MTP model carries irreversible recurrent state through verify and the economics are net-negative. It returns to the table only with a **non-recurrent** MTP model or a trained draft model (multi-week, not committed).
- **No host syncs on the decode hot path.** Ever. CI gates this.

### Surface
- **OpenAI- and Anthropic-compatible HTTP server** stays first-class. `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, SSE streaming, tool calling, logprobs, `/tokenize` — already done. Maintain compliance with the test suite as both wire formats evolve.
- **C library API** stable enough for embedding.
- **CLI** for benchmarking and interactive use.

---

## Agentic surface

The other half of the mission. An engine can be fast and still be useless to an agent if it drops tool calls, can't hold a long transcript, or stalls under concurrent sub-agents. These are tracked commitments, not nice-to-haves — each is gated in the release bar:

- **Tool calling & constrained decoding.** OpenAI + Anthropic tool-call wire formats; `response_format=json_schema` with a constrained-decode FSM that is *guaranteed* to emit valid, terminating JSON (e.g. the digit-run cap, #761). A broken JSON contract under any sampler state is a release blocker, not a quality nit.
- **Reasoning / thinking as a separable channel.** `reasoning_content` split from `content`, `think_budget` honoured (0 disables), gpt-oss Harmony channels parsed (#768). Thinking must be controllable per request and never leak into the answer.
- **Long-context agent loops.** Prefix cache default-ON for multi-turn (#763), `cache_control` pinning, KV budget defaults sized for agentic working sets (auto max_seq_len to 64k, NVFP4 KV-fraction fix, #771), `kv_cache.dtype=auto` honouring FP8 hints where quality allows (Qwen3, #704). The target is a coding-agent session that stays coherent across a full task, not a one-shot prompt.
- **Concurrency for sub-agent fan-out.** Per-request spec/vision/sampling state so heterogeneous concurrent requests batch together without an engine pause (per-request vision #774, per-request spec toggle #770). The per-seq decode consumers are batched (sampler #745, batched-M / tensor-core lm_head #746/#748) — concurrent decode 472→767 tok/s @16. Tens of concurrent agent requests on one 5090 must stay responsive (TTFT, ITL) without starving the single-stream path.
- **Reliability under sustained load.** Clean request cancel, ITL/cancel/queue metrics (#770), bounded decode bursts, fail-fast on bad input — the server must survive an agent that opens, abandons and retries streams for hours.
- **Multimodal agents.** Vision (gemma-3/4-VL) routed through the normal batched path so image requests interleave with text instead of pausing the engine (#774).

---

## What imp is NOT

Defining this is as important as defining the mission. These are explicit non-goals — saying yes to them dilutes the mission:

- **Not a multi-GPU engine.** Tensor/pipeline parallelism is out of scope. Single GPU, period.
- **Not a datacenter throughput engine.** vLLM and SGLang own batch=64+ continuous-batching serving on racks; we don't chase aggregate tok/s as the headline number. But **moderate agentic concurrency is explicitly in scope** — an agent harness fanning out *tens* of parallel sub-agents on one 5090 is a supported, tuned workload (batched sampler #745 + batched-M lm_head #746/#748 already lift concurrent decode 472→767 tok/s @16). The line: optimize single-stream latency first, then make concurrent agent fan-out efficient — never trade single-stream latency for aggregate throughput.
- **Not a training framework.** Inference only.
- **Not a mobile / embedded engine.** Workstation-class GPUs only.
- **Not a CPU engine.** GPU only. No AVX kernels, no Metal, no Vulkan.
- **Not a model zoo.** Architectures land when they justify the maintenance cost. The hero list above is curated, not exhaustive.
- **Not a research playground.** Every experimental kernel either lands as the default or is removed.

---

## Benchmarking discipline

- **llama-bench methodology** is the canonical comparison. pp512 + tg128 at minimum, plus realistic long-context (pp8192, tg512 @ 16k ctx).
- **Comparisons run on the same machine**, same driver, same CUDA version, same GGUF/SafeTensors weights where possible.
- **Every commit that touches a hot path** must report bench delta vs main on at least one hero model. Regressions need explicit justification.
- **Reproducibility:** bench scripts checked in, weight checksums recorded, results committed to a tracked file.

---

## Release bar

A release is shippable when, on RTX 5090:

1. All hero models pass correctness tests: perplexity within 0.5% of llama.cpp reference at the same quant **with documented owner-approved speed/quality trades excluded**; each such trade must be opt-out via config and listed here. Current trades:
   - `gemm.nvfp4_lm_head_gdn` default-ON (#483): +2.2% PPL for +11.4% decode on GDN hybrids.
   - `gemm.fp8_ssm_proj` on **GGUF** hybrids default-ON (#962): +1.8% PPL (201-token corpus) for +21% decode on Qwen3.6-35B UD-Q4_K_M — E4M3 stacked on the Q8_0 lattice; the native-NVFP4 branch of the same flag is PPL-flat (#949) and is not a trade.

   - `gemm.nvfp4_lm_head` default **"auto"** (#982, resolved 2026-07-13): the LM-head NVFP4 decode cache follows the measured net rule — ON for native BF16/F16 heads (+8-16% decode, +2.2% PPL, the long-standing accepted trade) and for small dense GGUF heads (d_model ≤ 4096: 4B +6.6% decode/+3.8% PPL, 8B +5.8%/+2.6% — decode win exceeds PPL cost); OFF for larger or MoE GGUF heads where the 2026-07-12 sweep measured the reverse (14B +1.9%/+2.1%, 30B-A3B +3.7%/+5.0%). Cost: north-star (14B Q6_K) −~1.9% decode, accepted for default PPL parity; `"on"`/`"off"` override.
   - `gemm.fp8_attn_proj` default "auto" = full q/k/v/o FP8 decode sidecar on **gpt-oss** (#984, 2026-07-13): +12.1% decode (349.7→392.1 tok/s). Not a teacher-forced-PPL trade by construction — the sidecar is decode-only (M=1 GEMV) and an nsys-verified `--perplexity` run executes zero FP8 kernels; decode-path quality gated via greedy-identity at 100 tokens and coherent 512-token generations across prompts. Opt-out `"off"`, conservative `"qo"` middle mode.

   First systematic cross-engine measurement 2026-07-12 (`docs/audit/ppl_parity_2026_07_12.md`): with the LM-head opt-out imp is at parity (−0.8%…+0.2%) with llama.cpp on every comparable GGUF hero.
2. Decode tok/s leads llama.cpp by ≥5% on every hero model.
3. Prefill tok/s ≥ llama.cpp on every dense **NVFP4/SafeTensors** hero. GGUF prefill is explicitly best-effort: the Q4K-MMQ experiment (2026-05-28) showed imp's GGUF prefill ceiling is architectural (ties cuBLAS at ~4.3% of peak; llama.cpp's MMQ leads 1.3-2.4×) — the old "≥ llama.cpp on every dense GGUF hero" bar was permanently violated as written and is dropped (realignment 2026-06-06, #550).
4. Prefill tok/s ≥ 70% of vLLM single-seq on every MoE hero (measured ~1.4× gap 2026-05-31; the remaining levers are prefill attention and the grouped-GEMM launch/occupancy scheduler, #558).
5. No host syncs on the decode hot path (CI-enforced).
6. OpenAI **and Anthropic** API compliance suites green.
7. **Agentic surface green:** tool-call + `json_schema` constrained-decode batteries pass (valid, terminating JSON under degenerate sampler state); reasoning/`think_budget` and gpt-oss Harmony parsing correct; prefix cache + long-context KV defaults coherent across a multi-turn agent session; vision interleaves with text.
8. **Concurrency holds:** tens of concurrent agent requests stay responsive (TTFT/ITL within target) with clean cancel, and concurrent fan-out does **not** regress single-stream decode.
9. README benchmarks updated, commit hash of competitors recorded.

---

## North-star single number

**Qwen3-14B Q6_K decode tok/s at batch=1, ctx=2048, on RTX 5090.**

This number goes up over time. It never goes down. If a refactor makes it go down, the refactor is wrong, no matter how clean the code looks.

Measurement methodology (always cold-median, never single-shot): 5 independent `imp-cli --bench` invocations × 5 reps × 15 s cooldown between trials, take the median. Resists cuBLAS-algo-state drift over long sessions (`memory/bench_sustained_load_cublas_algo_drift_2026_05_23.md`).

Current: **157.71 tok/s default flags** (May 23 2026, cold-median methodology — 5 trials × 5 reps × 15 s cooldown, σ = 0.16 tok/s across samples; same code as the May 22 measurement, just less cuBLAS-algo-cache-state noise). The 150 milestone is hit by default with margin.
Previous: 150.1 tok/s (May 22 2026, single-shot, post PRs #362 + #364 + #367). 121.4 tok/s (May 2026, +25.5% vs llama.cpp c830f99).
Next milestone: **175 tok/s** — requires multi-week kernel-fusion work, not tuning. The roofline sweep (2026-05-30) showed decode is 87% NVFP4 GEMVs running at 66-70% HBM with a 4-bit-dequant co-limit (L1TEX 91%); occupancy raises and KPAR/MR rerouting are measured dead ends, and the LM_HEAD quantization unlock already shipped (#479/#483). The previously listed "FP8 prefill cache coverage" lever **does not exist on this hardware** — FP8 prefill is cuBLAS `NOT_SUPPORTED` on sm_120 (realignment 2026-06-06, #550).
Stretch: 200 tok/s would need speculative decoding, currently parked: the Qwen3.6 MTP head now accepts at 85%+ after the sigmoid-gate fix (#804), but spec-decode *generation* dead-ends on the GDN-hybrid model's irreversible recurrent state through verify (net-negative economics) — it needs a non-recurrent MTP model; there is still no MTP head for Qwen3-14B, and draft-model integration is multi-week, not committed.
