<!--
layer: L1
audience: operators
verified: 2026-09-22
commit: 9cbb8004
-->

# imp - Goal Definition

## Mission

imp is the best single-GPU **agentic** AI inference engine on NVIDIA RTX 5090 (`sm_120`): fastest, most capable backend for coding agents, tool-using assistants and reasoning loops on one workstation card. Single GPU, latency-first, not a datacenter throughput competitor to vLLM/SGLang.

Foundation, non-negotiable: **the fastest single-stream (batch=1) decode of any engine on this chip**, across every supported architecture.

- Nothing agentic may erode it.
- On top of raw speed (release-bar gated, see "Agentic surface"): unbreakable tool-call/JSON contracts, long-context multi-turn loops, reasoning as a separable channel, moderate concurrency (tens of requests on one GPU), reliability under sustained load.

## Definition of "best"

Ranked, non-negotiable: anything below this bar is a bug.

- Concurrent throughput for agent fan-out is secondary: never bought by regressing single-stream decode.

| # | Metric | Bar |
|---|---|---|
| 1 | Decode tok/s at batch=1 on RTX 5090 | Primary metric. Lead llama.cpp, vLLM, SGLang, ExLlamaV3, MLC-LLM on every shipped arch+quant |
| 2 | Prefill tok/s at batch=1 | >= llama.cpp on dense NVFP4/SafeTensors (GGUF best-effort, release bar 3). MoE gap to vLLM closed and reversed: pp4096 +4 %, pp2048 +27 % (2026-06-13); Coder-30B pp512 ~19.4k vs vLLM ~18.5k (2026-07-22). Grouped-GEMM dated detail (Coder-30B-NVFP4, cold-median 5x5, 2026-05-23): pp512 17521 tok/s, pp2048 18573, tg128@512 273.24 (σ 0.09), tg128@2048 268.46 (σ 0.10); the #374 skip-gather moved pp512 14562->17521 (+20.3 %, was 1.14-1.32x behind vLLM pre-#374); vs vLLM 0.20.2 pp512 18500 = gap 1.056x snapshot, re-measured 2026-05-31/06-06 at the real ~1.4x (1.056x sat inside a 2.6x cuBLAS-restart variance band); compute near roofline, gap was prefill attention + grouped-GEMM occupancy (#558, closed, `docs/roadmap.md`) |
| 3 | TTFT at 512/2048/8192/32k prompt | Competitive with vLLM despite batch=1 focus |
| 4 | VRAM efficiency | Fit larger models than competitors at equal quality; 32 GB serves up to ~70B dense. 80-120B-class MoE (small active set, hot experts resident, cold host-side): a target, not yet measured |
| 5 | Quality | PPL / downstream eval parity with llama.cpp at the same quant; no silent regressions for speed |

## Target hardware

RTX 5090 (`sm_120`, GB202, 32 GB GDDR7) is the hero target, every decision made for it first.

- RTX PRO 6000 Blackwell (96 GB) and the 5080/5070 Ti siblings share `sm_120`/`compute_120f` and inherit wins, lower tuning priority.
- Everything else unsupported by design; what `sm_120a` has/lacks: [`internals/ARCHITECTURE.md`](internals/ARCHITECTURE.md).

## Target models (hero set)

Leading on 5090, no exceptions (realigned 2026-06-06, #549/#550; gpt-oss-20b closed the last gap, #547/#572-#574).

- Requires staged local weights, a green degeneration battery, decode numbers in `BENCHMARKS.md`.
- A hero regressing against any competitor is a release blocker.

| Model | Quant | Why |
|---|---|---|
| Qwen3-4B / 8B | Q8_0 (8B also NVFP4) | daily driver dense |
| Qwen3-14B | Q6_K, NVFP4 | sweet spot, the north-star model |
| Qwen3-Coder-30B-A3B | NVFP4 | hero MoE |
| Qwen3.6-35B-A3B | NVFP4 | hybrid GDN+MoE daily driver, MTP head |
| Gemma-4 26B-A4B (text+vision) | NVFP4 | multimodal + MoE hero |
| Nemotron-H | NVFP4 | hybrid Mamba2+Attn+MoE flagship |
| gpt-oss-20b | MXFP4 | experts converted to NVFP4 at load, Harmony channels, tg ~315-345, pp512 ~16-19k |

`check-release.sh` stage 9 runs `make bench-competitive` `RELEASE_BAR=1`, failing any hero leading llama.cpp by < 5 %.

- Enforced over **5 of 7**: Coder-30B-A3B and Nemotron-H are NVFP4-only, llama.cpp has no NVFP4 path on `sm_120`, no shared-quant comparison exists; gate prints `N/7 contested`.
- Before 2026-08-21 enforced over 2 only (`perf_baseline.json`, `perf_baseline_north_star.json`); `docs/audit/DEBT_LEDGER_2026_08_21.md` (h).

**Extended** - validated opportunistically, not release-blocking (2026-06-06): DeepSeek-R1-Distill-7B/14B (never benched, Qwen2/LLaMA arch); DeepSeek-V2-Lite MLA (supported, #802/#803, bf16 28 GB experts host-offloaded, PPL ~3 % of HF: imp 6.43 vs 6.25, 534-tok, post 2026-07-07 YaRN fix); Gemma-3 27B (12B Q4_K_M + 4B-VL staged, 27B not); Phi-4 14B (NVFP4 staged, GGUF Q6_K not); Mixtral 8x7B (chat-template test only, never staged).

## What "best on 5090" requires

Means, not ends.

| Area | Commitment |
|---|---|
| Compute | NVFP4 stays the default fast path (FP16/BF16 correctness-only). MXFP4 FMHA +6.7-7.9 % over FP8 FMHA on Qwen3 is the baseline not ceiling. FA2 prefill family carries the prefill commitment (FP8xFP8 cuBLAS prefill `NOT_SUPPORTED` on `sm_120`). Paged decode attention (FP16/FP8/INT8/INT4/NVFP4 KV) + CUDA-graph decode is the workhorse, tuned per hero head dim. Grouped-GEMM MoE prefill: gap to vLLM closed, see table above |
| Memory & latency | Paged KV cache (block 16), LRU, prefix caching: keep, extend. NVFP4 decode KV ships, expand to prefill where quality allows; TurboQuant retired 2026-05-17, CLI flags survive as deprecated aliases. CUDA graphs everywhere on decode: done, never regress. PDL: keep aggressive. No host syncs on the decode hot path, ever, CI-enforced |
| Speculation | TurboDraft (L2-resident draft) parked: "~25-30 % K=1 acceptance" (2026-05-30) was a kernel bug (MTP attn-output gate used silu where sigmoid was expected), fixed (#804) to 85 %+ on Qwen3.6; generation stays parked, GDN-hybrid MTP carries irreversible recurrent state through verify (net-negative), needs a non-recurrent MTP model |
| Surface | OpenAI/Anthropic HTTP server first-class: chat/completions, `/v1/messages`, SSE, tool calling, logprobs, `/tokenize`. C library stable enough to embed. CLI for bench + interactive use |

## Agentic surface

Gated in the release bar.

| Surface | Commitment |
|---|---|
| Tool calling & constrained decoding | OpenAI + Anthropic wire formats; `response_format=json_schema` FSM guaranteed valid, terminating JSON (digit-run cap, #761); a broken contract under any sampler state is a release blocker |
| Reasoning channel | `reasoning_content` split from `content`, `think_budget` honoured (0 disables), gpt-oss Harmony parsed (#768) |
| Long-context loops | Prefix cache default-ON multi-turn (#763), `cache_control` pinning, auto max_seq_len to 64k (#771), `kv_cache.dtype=auto` honours FP8 hints (Qwen3, #704) |
| Concurrency, fan-out, reliability | Per-request vision/spec/sampling state, heterogeneous requests batch together (#774, #770); batched sampler + tensor-core lm_head (#745/#746/#748) moved concurrent decode 472 -> 767 tok/s @16; clean request cancel, ITL/cancel/queue metrics (#770), bounded decode bursts, fail-fast on bad input |
| Multimodal | Vision (Gemma-3/4-VL, Qwen3-VL) through the normal batched path (#774); Qwen3-VL adds dynamic resolution + DeepStack (#1163-#1180), one image per request |

## What imp is NOT

Explicit non-goals. Reasoning and reopen conditions: [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md).

| Not | Detail |
|---|---|
| Multi-GPU engine | tensor/pipeline parallelism out of scope, single GPU only |
| Datacenter throughput engine | vLLM/SGLang own batch=64+ racks; moderate agentic concurrency (tens of sub-agents on one 5090) is in scope, never traded against single-stream latency |
| CPU engine | GPU-only forward pass; CPU-resident cold-expert exception measured and withdrawn (14.0 ms/token host vs 4.7-8.9 ms streaming to GPU, see [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md)); 80-120B ambition ships GPU-side: expert cache, own slot pool (#1370/#1374/#1376) |
| Training framework, mobile/embedded, model zoo, research playground | inference only, workstation-class GPUs only; architectures land when they justify maintenance cost (hero list curated not exhaustive); every experimental kernel lands as default or is removed |

## Benchmarking discipline

llama-bench methodology: pp512 + tg128 at minimum, plus pp8192/tg512 @ 16k ctx; same machine, driver, CUDA version, weights where possible. Every hot-path commit reports a bench delta vs `main` on >= one hero, regressions need justification; bench scripts checked in, weight checksums recorded, results committed.

## Release bar
1. All heroes pass correctness: PPL within 0.5 % of llama.cpp at the same quant, documented opt-out trades excluded:

   | Flag | Default | Trade | Source |
   |---|---|---|---|
   | `gemm.nvfp4_lm_head_gdn` | ON | +2.2 % PPL for +11.4 % decode, GDN hybrids | #483 |
   | `gemm.fp8_ssm_proj` GGUF hybrids | ON | +1.8 % PPL for +21 % decode, Qwen3.6-35B UD-Q4_K_M; native-NVFP4 branch PPL-flat (#949) | #962 |
   | `gemm.nvfp4_lm_head` | auto (`"on"`/`"off"` override) | ON native BF16/F16 + small dense GGUF heads (d_model<=4096): +6-16 % decode/+2.2-3.8 % PPL. OFF larger/MoE GGUF (reverse sign, `is_dense=false` arm; `is_gdn_hybrid` escapes to ON). Gemma-4-26B-A4B priced 2026-08-21: +7.4 % decode/+9.0 % PPL (245.31->263.44 tok/s, 251.23->273.90 PPL, 6465-token corpus), losing by its own rule; gpt-oss-20b MXFP4 same day: +10.2 %/+18.2 % (413.02->455.24 tok/s, 105.86->125.12 PPL), losing worse. Gemma-4 measured 258.96 before `63df2d30`, 245.24 after (this trade, not a regression) | #982 |
   | `gemm.fp8_attn_proj` gpt-oss | auto | +12.1 % decode (349.7->392.1); decode-only GEMV (M=1), not a PPL trade by construction, an nsys-verified `--perplexity` run executes zero FP8 kernels; opt-out `"off"`, `"qo"` middle mode | #984 |

   2026-07-12 cross-engine measurement: with the LM-head opt-out imp is at parity (-0.8...+0.2 %) with llama.cpp on every comparable GGUF hero (`docs/archive/ppl_parity_2026_07_12.md`).
2. Decode tok/s leads llama.cpp by >= 5 % on every hero.
3. Prefill tok/s >= llama.cpp on every dense NVFP4/SafeTensors hero. GGUF best-effort: Q4K-MMQ experiment (2026-05-28) showed the ceiling is architectural (ties cuBLAS ~4.3 % of peak, llama.cpp MMQ leads 1.3-2.4x); old GGUF bar dropped (2026-06-06, #550).
4. Prefill tok/s >= 70 % of vLLM single-seq on every MoE hero - exceeded: +4-27 % (2026-06-13, re-verified 07-22, Coder-30B pp512 ~19.4k). #558 closed, `docs/roadmap.md`.
5. No host syncs on the decode hot path (CI-enforced).
6. OpenAI and Anthropic API compliance suites green.
7. Agentic surface green (constrained-decode under degenerate sampler state, reasoning/Harmony parsing, prefix cache + long-context KV multi-turn, vision interleaves with text) and concurrency holds (tens of requests, TTFT/ITL in target, clean cancel, no single-stream regression).
8. README benchmarks updated, competitor commit hash recorded.

## North-star single number

**Qwen3-14B Q6_K decode tok/s at batch=1, ctx=2048.** Goes up over time, never down.

- Methodology: 5x `imp-cli --bench` x 5 reps x 15 s cooldown, cold-median, resists cuBLAS-algo-state drift over long sessions (`memory/bench_sustained_load_cublas_algo_drift_2026_05_23.md`).
- `--bench-pp 2048` required (`--bench-pp 16` reads 169.00, `--bench-pp 512` reads 164.83 on the same build, shorter contexts, not gains).

| When | tok/s | Note |
|---|---:|---|
| 2026-07-26 (current) | 149.47 | 3x10-rep cold, spread <0.03 %, `ngram=false`; unchanged from 149.54 (07-15); delta vs May is the accepted `nvfp4_lm_head` trade (#982) |
| history | 157.71 / 150.1 / 121.4 | 2026-05-23 cold-median pre-speculation pre-#982 / 2026-05-22 single-shot post #362/#364/#367 / 2026-05 (+25.5 % vs llama.cpp `c830f99`) |

Gate re-pinned spec-OFF 2026-07-15: dense bench drafts ~99.9 % accept, so spec-ON tg measured the restart-volatile spec-verify GEMMs (11 % swing); pure decode stable <1 %. Default (spec ON) reads above spec-OFF since the verify-chunk NVFP4 overlay (#998/#1001), not pinnable at 3 %.

Next: **175 tok/s**, multi-week kernel fusion.

- Roofline (2026-05-30): decode 87 % NVFP4 GEMVs at 66-70 % HBM, 4-bit-dequant co-limit (L1TEX 91 %).
- Occupancy/KPAR/MR rerouting are dead ends; the LM-head quantization unlock already shipped (#479/#483).
- "FP8 prefill" does not exist on this hardware: `NOT_SUPPORTED` on `sm_120` (#550).

Stretch: 200 tok/s needs speculative decoding, parked - Qwen3.6 MTP accepts 85 %+ (#804 sigmoid fix) but generation dead-ends on GDN-hybrid irreversible recurrent state through verify; needs a non-recurrent MTP model. No MTP head for Qwen3-14B; draft-model integration multi-week, uncommitted.
