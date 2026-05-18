# Roadmap

Open work and known limitations. Shipped work lives in [`CHANGELOG.md`](../CHANGELOG.md).

This is a single-author single-target experiment, so "roadmap" is more "current focus" than "schedule." Items here are ordered by impact, not by ETA.

## Known limitations

### ~~Gemma-4 carve-outs~~ — all removed

All three Gemma-4 carve-outs are now gone:

- **FP8 KV cache** — PR #91 (2026-05-01). The "dual head_dim 256/512 needs per-layer-aware kernels" hypothesis was a red herring; the KV write/read kernels handle per-layer head_dim correctly via `Q.shape[3]` template dispatch. Real bugs were (a) FP8 calibration reading the workspace's allocated shape (`max_hd=512`) instead of the live shape (`hd=256` on SWA layers, junk in trailing 256 cols) and (b) warmup-derived absmax poisoning the high-water-mark scale on Gemma-4's `output_norm` outliers (max=588).
- **NVFP4 decode cache for Q*_K source** — PR #186 (2026-05-15). The per-tensor convert→quantize loop in `executor_pre_dequant.cu` already handled mixed (N, K) shapes correctly; the disable was overly defensive. Removing it on Q4_K_M / UD-Q4_K_M: pp512 1713 → 2394 tok/s (**+40%**), tg256 176 → 197 tok/s (**+12%**).
- **FP8 prefill** — 2026-05-15. The 2026-05-09 -5..-19% slowdown was real at the time but mostly closed by intermediate prefill work (PRs #177, #181). Re-measured Q4_K_M: pp128 +1.0%, pp512 -0.9%, pp833 -4.2%, pp2048 **+7.3%** — neutral with long-context advantage. FP8 also halves the activation cache size. Users wanting max prefill at medium pp can opt out via `[attention] fp8_prefill = "never"`.

Default KV dtype is FP16; FP8 is opt-in via `--kv-fp8` (or `kv_cache.dtype = "fp8"` in `imp.conf`). Coherent on Qwen3 dense, Qwen3.5/3.6 GDN, Llama-3.2, and Gemma-4 (post PR #91).

### Chunked prefill scope (full-attention + hybrid GDN/Mamba2 + Gemma-4; FP16/FP8/NVFP4/INT4 KV)

Default `prefill_chunk_size = 512` for full-attention models (Qwen3, Llama, Mistral), hybrid GDN+MoE / Mamba2+MoE models (Qwen3.5/3.6, Nemotron-H), and **Gemma-4** with FP16, FP8, NVFP4, or INT4 KV cache. Past chunks' K/V are read from the paged cache via `paged_kv_gather_*` and concatenated with the current chunk before a rectangular `attention_cublas_prefill` with `q_offset`-aware causal masking + `sliding_window`-aware mask (added 2026-05-15 for Gemma-4 SWA layers; the same path now replaces the naive FP32 workaround for Gemma-4 SWA in non-chunked prefill too). INT4 dequant gather added 2026-05-15 (symmetric 4-bit + per-head FP16 scale; INT4 KV's pre-existing long-context quality regression is independent of chunked prefill). PR #114 mitigation (default `prefill_chunk_size = 0`) is replaced by `Engine::resolve_prefill_chunk_size_()` which clamps to 0 for out-of-scope archs.

**Out-of-scope** — stay at `prefill_chunk_size = 0` via per-arch default; explicit `--prefill-chunk-size N` is logged + clamped to 0:

- Gemma-3 (SWA, no test model in repo — kernel work is identical to Gemma-4, just unverified)
- Llama-4 (MoE + SWA)
- TurboQuant / TurboQuant Lite KV dtypes (QJL-sketch storage; would need a sketch-aware gather)

Each excluded class is a separate larger work item.

### ~~`d_pf_block_tables_` undersized for prompts ≥ max_seq_len~~ — FIXED #134

When a single prompt exceeds `max_seq_len`, the engine's pre-allocated device buffer `d_pf_block_tables_` (sized `max_seq_len / block_size`) overflowed during `cudaMemcpyAsync`. Fixed in PR #134: `d_pf_block_tables_` is now sized from `max_blocks` (the total KV cache pool count), so a single request's block_table can grow to the entire cache without overflowing.

### ~~NVFP4 SmoothQuant input_scale (Mistral-3.2 NVFP4)~~ — RETIRED 2026-05-18

Phase 1 diagnostic (full findings: `docs/superpowers/plans/2026-05-18-nvfp4-smoothquant-phase1-findings.md`) closed this roadmap item on **all relevant NVFP4 checkpoints**, including the one the design memo gated on.

**Local corpus** — 6 NVFP4 models scanned via SafeTensors header + engine audit-log run. `input_scale` is **100 % scalar (numel=1)** on every Linear of every checkpoint; none of the 6 recipe.yamls ships a `SmoothQuantModifier` (all are pure `QuantizationModifier: scheme: NVFP4`). The per-Linear scalar is llm-compressor's `input_global_scale` activation-absmax anchor, not a SmoothQuant `1/s` carrier — engine's existing "intentionally NOT applied" stance at `executor_pre_dequant.cu:431` is the correct behavior. The 2026-05-07 DIVIDE/MULTIPLY refutation on Gemma-4-NVFP4 is now structurally explained (no SmoothQuant correction existed to apply, since no SmoothQuant was calibrated into the model).

**Mistral-3.2-NVFP4** (the one SmoothQuant-calibrated checkpoint in scope) — resolved via HF range-fetch of the recipe.yaml + first 16 MB of each safetensors shard (no full 15 GB download required for the answer; full download running as artifact). The recipe.yaml carries an explicit `SmoothQuantModifier` with mappings `[q_proj, k_proj, v_proj] → input_layernorm` and `[gate_proj, up_proj] → post_attention_layernorm` — **exactly** the structure design memo §4 sub-case (b) predicted, with `diag(1/s)` migrated into the upstream RMSNorm weights at calibration time. The 280 on-disk `input_scale` tensors are all `(1,)` scalar (sub-cases (a)/(c) refuted on the SmoothQuant model too); the 80 `(5120,)` layernorm weights (40 layers × 2 norm types) are the SmoothQuant migration targets. imp already loads + applies those layernorm weights via the standard path — nothing engine-side to add.

PR #78's `use_default_system_prompt=false` workaround for Mistral-3.2-NVFP4 long-context drift stays load-bearing, but its root cause is **not** missing SmoothQuant absorption — it's the NVFP4-activation-noise-grows-with-`||X||` issue tracked in `memory/nvfp4_long_context_regression_2026_04_28.md`. That fix is a separate, much larger workitem (dynamic NVFP4 activation quantization end-to-end + a real NVFP4×NVFP4→FP32 GEMM in the dense path) outside this roadmap entry's scope.

**Phase 2-4** (Option-B `smooth_activations` pre-pass kernel + regression tests) **never needed** — no checkpoint in the foreseeable workload would trigger a non-null `s_inv` path. Diagnostic patch shipped to `src/model/weight_upload.cu` (per-Linear `scalar=N per_channel=M` split + `ndim/shape/numel` samples under `diagnostics.audit_nvfp4_scales=true`) as the lasting deliverable for any future SmoothQuant-calibrated NVFP4 checkpoint that might land.

**Design memo:** `docs/archive/plans-2026-05/nvfp4_smoothquant_input_scale_design_2026_05_17.md` (predicted sub-case (b); confirmed empirically).

### Qwen3.5-27B MXFP4 fails at load — Phase A1+A2 shipped (#244); A3 gated on two external blockers

12 GiB of MXFP4 weights plus the 48 GiB FP16 fallback oversubscribes 32 GB of VRAM. PR #60 converted the original IMA into a clean `IMP_LOG_ERROR` pre-flight refusal at `src/graph/executor_pre_dequant.cu:1532-1574`. **Design memo:** `docs/plans/qwen35_27b_mxfp4_host_dequant_design_2026_05_17.md`.

**Phase A1+A2 shipped via PR #244 (2026-05-17):** new `attention.mxfp4_fp16_cache_policy` config field (default `legacy` = pre-PR behavior; `pruned` skips MoE `expert_*_packed` + LM head `out_proj_` from the FP16 fallback cache). Code at `src/graph/executor_pre_dequant.cu:1521-1571`. Validated on **Qwen3.5-4B MXFP4** (dense, the only locally-available MXFP4 model — the original `Qwen3.5-27B-mxfp4.gguf` mentioned in older memos is no longer on disk): FP16 cache shrinks 8020 → 6807.50 MiB (−1.18 GiB = LM head only, no MoE in dense 4B); pp64 +39 %, tg32 +11 %. The MoE expert prune is wired but not exercised by any local MXFP4 model. No production-default flip (`legacy` remains the default until a Qwen3.5-27B end-to-end test runs).

**Phase A3 (verify on 27B) gated on two independent external blockers — neither in imp's control:**

1. **No public Qwen3.5-27B MXFP4 GGUF exists.** `unsloth/Qwen3.5-27B-GGUF` ships only standard quants (Q3/Q4/Q5/Q6/Q8/IQ4/UD variants), no MXFP4. The HF MXFP4 sources for this model (`olka-fi/Qwen3.5-27B-MXFP4`, `mrhuseyn4/qwen3.5-27b-mxfp4`, `kaitchup/Qwen3.5-27B-MXFP4A16`) are all SafeTensors, and imp's SafeTensors loader explicitly refuses MXFP4 (`src/model/safetensors_loader.cpp:1095`: "imp does NOT have a SafeTensors MXFP4 decode path yet. Convert to GGUF for actual MXFP4 support."). Resolving this requires either a community GGUF publication or an in-repo BF16→MXFP4 GGUF conversion pipeline (multi-hour + tooling).
2. **N=48 alpha/beta MXFP4 GEMV NaN.** Separate kernel bug — even after a working load, GDN decode through alpha/beta MXFP4 GEMV produces NaN logits (`tok=-1`). Out of scope for the loading fix per design memo §1.1. Tracked in `memory/qwen35_27b_mxfp4_ima_2026_04_25.md` §"What remains".

Phase B (host-dequant + storage planner) remains deferred — only worth funding if Path A's pruned policy proves insufficient on the actual 27B once both blockers above clear.

**Workarounds:** Qwen3.5-9B Q8_0 (cleanest), Qwen3.5-35B-A3B Q4_K_M (closest to 27B-scale).

### Gemma-4 Q4_K_M code-gen drift

Q4_K_M decodes coherent for chat but degenerates on complex code-gen prompts (Fibonacci → backtick loop). Cause is accumulated FP16 drift over 30 layers. Practical fix: use Q5_K_M or Q8_0 when output quality matters.

### MoE expert offload disables CUDA Graphs

Decode fast-path (`src/graph/executor_forward_moe.cu:524`) handles all device-resident MoE quants — Q6_K, Q8_0, Q4_0, Q4_K, Q5_K, Q2_K, Q3_K, Q5_1, NVFP4 — fully device-side (no D2H memcpy of routing or expert offsets), so CUDA Graphs capture cleanly. Verified A/B 2026-05-07: Qwen3-Coder Q6_K tg128 117 → 232 tok/s (+97%), Gemma-4 Q4_K_M tg128 65 → 179 tok/s (+177%).

The remaining limitation is **host-offloaded experts**: when the model + KV doesn't fit in VRAM, `experts_on_host_=true` triggers per-layer H2D staging via `expert_cache_` LRU at `executor_forward_moe.cu:1256-1278` (dequant path) and `:1413-1426` (fused-GEMV path), inserting a host pointer dereference + `cudaMemcpyAsync` per expert per token. `engine.cpp:1158-1165` disables CUDA Graphs in that mode. Tip: bumping `IMP_EXPERT_OVERHEAD_PCT` from 30 to 10 trades VRAM headroom for full on-device experts and unlocks +97% to +234% decode (Qwen3-Coder Q6_K is the real workload that actually triggers host-offload today). Generalising the LRU prefetch to be device-side / async-pipelined would restore Graphs while keeping host-offload available. **Design memo:** `docs/plans/moe_host_offload_graphs_design_2026_05_17.md` (4-6 weeks for full Phase 1-5).

**Phase 1 spike complete (2026-05-17, memo `moe_host_offload_phase1_findings_2026_05_17.md`): PROCEED.** Synthetic A/B on Qwen3.6-35B-A3B Q4_K_M via `moe.force_host_experts=N`: baseline 210 tok/s → Graphs-OFF 73.75 tok/s (+8.80 ms/tok graph penalty) → force_host=20+Graphs-OFF 28.57 tok/s (+21.45 ms/tok host-offload incremental, total 35.01 ms/tok). nsys trace shows **PCIe is only ~14% of the host-offload penalty** (3.0 ms/tok of 21.45 ms; 6.5 GB/s effective WSL2 H2D bandwidth, not Gen5-bound). The remaining 18 ms/tok is host-side LRU bookkeeping + Graphs coalescing loss — the exact surface Phase 2-5 targets. Projected ceiling with all five phases shipped: ~128 tok/s = **+348%** over current host-offload, comfortably above the memo's 80% decision gate. Phase 1 also surfaced a separate QW8 NVFP4 da_cache over-strict abort at `executor_pre_dequant.cu:2483` that fatals on three legitimate paths (`--no-nvfp4` GGUF MoE, `force_host_experts`, partial NVFP4 budget on Q4_K_M MoE); fix in PR #232.

**Phase 2-5 shipped, +10% measured (2026-05-17):**

- **Phase 2 (#233)** — device-side LRU mirror at `[n_layers × 3 × n_experts]` mirroring host LRU state. Bookkeeping only, no perf change. Foundation for kernel-side slot resolution in Phase 5.1.
- **Phase 3 (#235)** — per-layer slot pool partitioning + lazy host_expert_addrs + canonical host_packed_ptrs tables. Per-layer isolation lets Phase 4 prefetch L+1 without evicting L. Hit-rate regression −22pp baked in (uniform partition wastes slots on non-MoE GDN layers) — recoverable via Phase 5.1 prefetch overlap, otherwise mitigated by `IMP_EXPERT_OVERHEAD_PCT=10`.
- **Phase 4 (#236, fixed in #237)** — async prefetch on a dedicated `prefetch_stream_`, top-K most-recent from per-layer access ring. Original PR claimed +43% which was a silent bug artifact (per-(layer, proj) byte size mismatch caused all prefetch H2Ds to fail with "invalid argument", the dispatch fallback re-loaded the same expert, the bug counted as a "savings"). PR #237 corrects to per-proj byte size. **Honest perf at `moe.prefetch_top_k=3` is +10% over Phase 3 baseline** (30.80 → 33.78 tok/s on the Qwen3.6-35B Q4_K_M force_host=10 setup). Opt-in via `moe.prefetch_top_k`.
- **Phase 5 (#238) — blocked on dispatch refactor.** Empirical attempt to drop the graph-disable guard at `engine.cpp:1158` surfaced two structural blockers: (a) host-driven `get_or_load()` captures cudaMemcpyAsync nodes with fixed (src host ptr, dst slot) pairs that don't adapt to per-token routing — captured-graph replay would silently corrupt output; (b) `cudaStreamWaitEvent(compute_stream, prefetch_done_[layer], 0)` inside the capture window fails with "dependency created on uncaptured work in another stream" even under `cudaStreamCaptureModeRelaxed`. Shipped `moe.allow_graphs_under_offload` flag (default false) as research scaffolding + findings memo `docs/plans/moe_host_offload_phase5_findings_2026_05_17.md`. The flag does not currently produce useful graphs.
- **Phase 5.1 path (multi-week, deferred)** — refactor every gemv variant (`gemv_q6k`, `gemv_q8_0`, `gemv_dp4a_kpar_*`, dequant→cuBLAS fallback) to take `(d_lookup_ slice, slot-pool base)` and compute `slot_idx = d_lookup_[proj * n_experts + expert]` at runtime. Once dispatch is kernel-driven, the captured-graph compute nodes adapt to per-token routing without re-running host code and blocker (a) dissolves. Realistic 2-4 weeks; the original 3-5 day estimate in the design memo was off.

Net for the user today: `IMP_EXPERT_OVERHEAD_PCT=10` remains the right advice (+97-234% via the existing graph fast-path). The Phase 4 prefetcher is opt-in and worth +10% for the rare case where the workload actually needs host-offload (model+KV genuinely doesn't fit at the 10% overhead). The infrastructure to do better (mirror, host_addrs, prefetch stream) is shipped and waiting for Phase 5.1.

### Reasoning models + JSON schema — preamble pass-through

Reasoning models (Qwen3.6, DeepSeek-R1, Gemma-4-thinking) emit `<think>...</think>` before every response. Strict JSON / JSON-Schema enforcement starting at token 0 masks the `<think>` opener, leaving the model with no valid token to sample. Auto-detected via the tokenizer (presence of `<think>` + `</think>` special tokens) and handled by `PreambleGate` (`src/compute/preamble_gate.h`): the gate lets all tokens pass until the close marker, an `{` / `[` is observed, or a budget cap is hit, then strict enforcement kicks in.

Non-reasoning models (Llama-3.2 etc.) get the same gate in budget-only mode (8-token slack) so markdown-fence preambles like ` ```json ` and short verbal openers ("Sure! ") pass through cleanly.

When a request sets both `tools` and `response_format=json_schema`/`json_object`, the engine-side `PreambleGate` enters tool-aware mode. It bypasses the schema mask through the entire tool-call body (delimited by single-token tags for ChatML/Hermes/Mistral/Gemma, or `<function=`/`</function>` char-prefix/suffix for Llama3) and stays unmasked for the rest of the generation, supporting parallel tool calls. If the model emits free-text JSON instead, the schema mask kicks in normally on the first `{`/`[`. Tool argument validation continues to flow through each tool's own `parameters` schema (post-hoc, not in-stream).

## Performance work

### ~~Closing the TurboQuant–FP8 gap~~ — RETIRED 2026-05-17 (Phase 5, PR #251)

**TurboQuant retired in favor of `--kv-mxfp4` (MXFP4-KV) per the design memo's Path A endgame.** Net deletion: **−2828 LOC across 33 files** (PR #251). `--kv-turboquant` / `--kv-turboquant-lite` remain as deprecated CLI aliases that emit a one-shot `IMP_LOG_WARN` and fall back to MXFP4-KV. `IMP_DTYPE_TURBOQUANT(_LITE)` C-API enumerators are preserved as ABI-stable aliases routing to `QType::MXFP4_KV`. `src/quant/turboquant_fp4.cuh` (UE8M0/FP4 helpers) is kept — MXFP4-KV depends on it.

The full historical narrative below is preserved as audit trail.

---

TurboQuant currently runs ~23 % behind FP8 on Qwen3-8B Q8_0 decode (191 vs 248 tok/s) end-to-end — but the kernel-level gap is **3.3-4.1× FP8 per attention call**, much larger than the end-to-end number suggests (weight-bandwidth-boundedness compresses the visible gap). Closing it would need to drop QJL and switch to MXFP4 K directions with group micro-scales. **Design memo:** `docs/archive/plans-2026-05/turboquant_fp8_gap_design_2026_05_17.md` (651 lines).

**Phase 1 microbench complete (2026-05-17, `docs/archive/superpowers-2026-05/plans/2026-05-17-turboquant-phase1-findings.md`):**

| Metric | pp=512 | pp=4096 | Threshold (§5) | Verdict |
|---|---:|---:|---:|---|
| QJL fraction of TQ kernel time | **54.7 %** | **60.3 %** | ≥ 15 % | ✅ PASS by >3× |
| NVFP4 vs FP8 ceiling gap        | **20.5 %** | **26.8 %** | ≤ 5 %  | ❌ FAIL by ~5× |

Path A is bottleneck-targeted (QJL XNOR+popcount + Q-side sketch precompute is the dominant cost, not the marginal 6-10 % the design memo §2.3 bracketed) but it **won't reach FP8 parity** — even after rewriting TurboQuant as NVFP4-KV-with-INT4-V, the path inherits NVFP4's ~20-27 % residual gap to FP8 from the K-norm extra FP16 load + INT4 V dequant + scale-pool indirection. The right framing is **"retire TurboQuant in favour of NVFP4-KV, not optimise it"** — the big win is the −2000 LOC code retirement, not headline perf parity with FP8. End-to-end decode tok/s improvement is bounded by weight-bandwidth-boundedness per `bitdecoding_long_context_eval_2026_05_14.md`; the 23 % roadmap gap likely closes to ~8-12 %, not zero.

Path B (per-token QJL tuning) **shelved** — its 3-5 % recovery ceiling vs QJL's actual 54-60 % dominant cost makes it the wrong shape. `--kv-turboquant-lite` remains retirable as part of Path A.

**Phase 2 NIAH ran (2026-05-17, `docs/archive/superpowers-2026-05/plans/2026-05-17-turboquant-phase2-findings.md`):**
- **4 K context, all 4 configs (FP16 / FP8 / TQ-QJL-on / TQ-QJL-off via `IMP_TQ_SKIP_QJL=1`):** 100 % retrieval. Δ(TQ_off − TQ_on) = 0 pp → formally PASS by the design memo's ±5 pp threshold, but vacuous signal (Qwen3-8B aces NIAH at 4 K regardless of KV dtype).
- **16 K context, FP16 / FP8:** 53 % / 67 %. **Both TQ configs:** 0 % — the engine rejects 15 547-token prompts with `Prefill error: out of memory` because TurboQuant has no chunked-prefill support (single-chunk cap = 4096 BPE tokens, `src/runtime/engine.cpp:1997`).

The 16 K engine limit is itself a Phase 2 finding: **TurboQuant cannot reach long context on the current engine.** Path A (drop QJL, retire TQ to NVFP4-KV-with-INT4-V shape) inherits chunked prefill from NVFP4 — so Path A is the *unblocker*, not a regressor, for long-context TQ workloads. Combined with Phase 1's perf finding, both phases point at the same conclusion: **retire TurboQuant in favor of NVFP4-KV, don't optimise it**.

**Phase 3 shipped end-to-end:** Slice 1 (PR #248, `ScaleDtype` template scaffolding), Slice 2 (PR #249, `--kv-mxfp4` end-to-end wiring), Slice 3 (this branch, NIAH re-run + bugfix).

**Slice 3 NIAH re-run (2026-05-17, `docs/archive/superpowers-2026-05/plans/2026-05-17-mxfp4-kv-slice3-findings.md`):**
- First run found MXFP4-KV producing **degenerate "the the the" loops, 0 % NIAH retrieval at 4K**. Looked like a Path A dead end.
- Root cause: encoder-decoder scale mismatch in `write_kv_cache_mxfp4_kv_kernel` — nibbles quantized with `inv_sc = 1/sc_exact`, scale stored as UE8M0-rounded `sc_byte`. For E4M3 the mismatch is ~1.5 % (NVFP4 tolerates it); for UE8M0 (power-of-2 only) it's up to 2 × per group, compounded over 32 layers ⇒ doom.
- 5-LOC fix: quantize to UE8M0 first, derive `inv_sc` from the UE8M0-decoded scale, then quantize nibbles round-trip-consistently. TurboQuant's MXFP4 K write kernel already did this pattern correctly (`executor_kernels.cu:1187-1191`); Slice 2 missed the precedent.
- Post-fix 180-prompt matrix on Qwen3-8B Q8_0:

| Config       | 4 K          | 16 K        | vs NVFP4 (16K) |
|---           | ---:         | ---:        | ---:           |
| FP16 (gold)  | 100.0 %      | 60.0 %      | −6.7 pp |
| FP8          | 100.0 %      | 60.0 %      | −6.7 pp |
| **NVFP4**    | **100.0 %**  | **66.7 %**  | 0 anchor |
| **MXFP4-KV** | **100.0 %**  | **60.0 %**  | **−6.7 pp** (1/15 prompts; per-depth identical except d=0.75) |
| TQ (QJL on/off) | 100.0 %   | 0.0 %       | engine limit (Phase 2) |

Δ = -6.7 pp is single-prompt-of-15 variance, not a systematic regression. **Path A's "MXFP4-KV as thin variant of NVFP4" framing validated.**

**Decision: PROCEED to Phase 5** (TurboQuant retirement, ~−2000 LOC per design memo §5 Phase 5):
1. Deprecate `--kv-turboquant` → alias to `--kv-mxfp4` (or `--kv-nvfp4`) with one-shot `IMP_LOG_WARN`.
2. Remove `--kv-turboquant-lite` with `IMP_LOG_ERROR` + fallback.
3. After one release: delete `src/quant/turboquant.{h,cu}`, `src/compute/attention_paged_turboquant.cu` (1108 LOC), the three TQ KV-write kernels in `executor_kernels.cu`, the sketch_pool path in `kv_cache.cu`, and `tests/test_turboquant.cu`.

Slice 4 (two-level scaling on MXFP4-KV) was planned as a precision reserve but the Slice 3 bugfix made it unnecessary for correctness — defer indefinitely unless a future workload demands it.

### `pp=512` on large dense models

Qwen3-32B Q4_K_M and Mistral-24B Q6_K sit at ~0.5–0.6× llama.cpp at `pp=512` (Qwen3-32B Q4_K_M: 1888 tok/s, RTX 5090). nsys profile (2026-05-15, `tools/analysis/profile_pp512_large_dense.sh`):
- 4.4 % FP16 compute / 3.4 % memory-bandwidth utilization — launch-overhead + dequant-overhead bound.
- 25 % GPU time in `dequant_q4k_kernel` (FP16 cache doesn't fit), 64 % in cuBLAS GEMMs.
- 23 % host time in sync `cudaMalloc`/`cudaFree` (939+930 calls).

imp already ships a Q4_K × Q8_1 kernel (`src/compute/ggml_mmvq.cu::mmvq_kernel`) but it's a *warp-per-output-element batched-GEMV*, not a tiled GEMM. Measured crossover on Qwen3-32B Q4_K_M (`tools/analysis/bench_q4k_mmvq_crossover.sh`): mmvq wins at M ≤ 16 (e.g. M=8: 92 vs 45 tok/s), cuBLAS wins above M=16 (M=512: 1802 vs 251 tok/s). mmvq saturates at ~250 tok/s regardless of M because each output element gets its own warp with no TILE_M × TILE_N weight/activation reuse.

A **direct tiled Q4_K_M GEMM kernel** (`src/compute/mmq_q4k.cu`, Phase A dp4a) was prototyped in PR #189 — microbench beat mmvq by 2.0-2.4× across M=32..512 (tile `<16,32,1,1>`, 512 thr, SMEM bank-conflict-padded). On Gemma-3-12B Q4_K_M end-to-end: **wins +13-56 % at M=2..16**, **loses to FP16-TC cuBLAS at M ≥ 32** — dp4a peak (~50 TFLOPS) vs FP16-TC peak (~838 TFLOPS) is a 16× ceiling gap that tile tuning cannot close. **The Phase A v1 kernel + the v2 HMMA followup were both retired** in the Day-30-60 streichliste sweep (PR #193 → recovery PR #199, -2 726 LOC): v2 because its e2e regressed -4% on Qwen3.6-35B (MoE under MIN_M=64 + fp16_cache hits skip v2 + per-call dispatch overhead — see option 1 below); v1 because it dispatched in `[2, 16]` only and the win was too narrow to justify the maintenance cost. Current main has neither — high-M Q4_K_M flows through dequant→cuBLAS for everything. Multi-stream dequant↔GEMM overlap was refuted by measurement (GPU busy ratio ≈ 100 %).

Closing the high-M gap requires porting the inner loop to a Tensor-Core MMA. Two paths exist on sm_120:

1. **FP16 HMMA** (`mma.sync.m16n8k16.f16`) with on-the-fly Q4→FP16 dequant. **Attempted and retired**: shipped across 7 phases as `src/compute/mmq_q4k_v2.cu` on a feature branch, microbench reached **4.87× v1 dp4a** at M=512 (kernel-only). End-to-end on Qwen3.6-35B Q4_K_M was **-4% pp** in production because MoE keeps experts under the `MIN_M=64` v2 threshold, the FP16 weight cache (`wcache_.fp16`) hits skip v2 entirely, and Phase 1 dispatch overhead is paid per call. Retired in PR #193. Re-eval pending a dense Q4_K_M model that bypasses both the MoE-min-M and the fp16_cache hot path. Memo: `mmq_q4k_v2_phase2_shipped_2026_05_16.md`.
2. **INT8 IMMA** (`mma.sync.m16n8k32.s32.s8.s8.s32`, ~838 TOPS) with Q4_K dequant→INT8 reordering — **explored across PRs #254–#269 and DEFERRED 2026-05-18.** Wrap-up memos: `docs/superpowers/plans/2026-05-18-q4k-imma-phase2b-ceiling.md`, `docs/superpowers/plans/2026-05-18-q4k-imma-phase3-refuted.md`.

   - **Phase 1** (PR #254): raw MMA microbench confirmed **931 TOPS** on sm_120a (3.82× FP16 HMMA), no throttle.
   - **Phase 2A** (PR #255): Q4_K → symmetric-s8 reorder kernel.
   - **Phase 2B** (PR #267): production tile kernel — BLOCK_M=64 N=32 K=32, 4 warps/CTA with WRM·WRN=2·2, 2-stage cp.async. **Plateaus at ~40 TOPS** (4.3 % of raw MMA peak); 3-stage cp.async + ldmatrix.x4 refuted at saturation.
   - **Phase 2C** (PRs #263 + #268 + #269): infrastructure (`WeightCaches::q4k_imma` + `gemm.q4k_imma_enabled` knob), high-level `mmq_q4k_imma_gemm` entry, production dispatch handler. Default off.
   - **Phase 3** (this date, refutation memo): end-to-end A/B on **Gemma-3-12B-it Q4_K_M** at pp2048: IMMA **3.8× slower** (1697 vs 6418 tok/s) than the default dequant→cuBLAS path. Decode unchanged. Per-dispatch IMMA at 40 TOPS = 1.5 ms; cuBLAS dequant→FP16 + GEMM at 244 TFLOPS = 0.26 ms ⇒ ~6× per-call ratio, compounded across ~336 Q4_K dispatches per prefill.

   **Decision: DEFERRED.** All artifacts retained (kernel + dispatch handler + knob) so future researchers can re-bench when one of these conditions appears:
   - Larger FFN shapes (N ≥ 8192, K ≥ 4096) where the 40 TOPS plateau region widens
   - Fundamental kernel restructure (persistent CTAs + stream-K, or CUTLASS template instantiation)
   - A workload where dequant→cuBLAS isn't reachable (fp16_cache disabled + dequant_scratch unavailable — rare)
   - Activation-quant cost fuseable into the prior layer's epilogue

Memos: `mmq_q4k_phase_a_2026_05_15.md` (v1 dp4a sweep), `mmq_q4k_v2_hmma_design_2026_05_15.md` (v2 HMMA blueprint), `q4k_mmvq_crossover_2026_05_15.md` (original mmvq-vs-cuBLAS measurement), `docs/plans/q4k_imma_design_2026_05_17.md` (INT8 IMMA design), `docs/superpowers/plans/2026-05-18-q4k-imma-phase1-findings.md` (Phase 1 PROCEED), `docs/superpowers/plans/2026-05-18-q4k-imma-phase2b-ceiling.md` (Phase 2B 40 TOPS ceiling), `docs/superpowers/plans/2026-05-18-q4k-imma-phase3-refuted.md` (Phase 3 e2e refuted).

### Speculative decoding — investigated and shelved

EAGLE-3, self-speculative, DFlash, PPM-based TurboDraft, and n-gram speculation were all investigated. None paid off on a single RTX 5090: decode is bandwidth-bound, and the variants tested either failed to amortise weight reads (EAGLE-3, self-spec at 56–50% of baseline) or had unacceptable acceptance rates (PPM 0% on real text). Spec-decode CLI flags were removed in `7380ea8`.

**MTP (Multi-Token Prediction) — Phase 2.2 shipped, Phase 3.5 deferred.** PRs #171/#172/#174/#175 (2026-05-14) shipped the full MTP scaffolding for Qwen3.6: weights load, FC projection + dual pre-norms, full transformer block (Qwen3-Next attention with attn_output_gate + per-head qk-norm + mrope + KV scan + GQA), 256-expert top-8 MoE with shared expert + sigmoid gating, final_norm + lm_head. CLI `--mtp-spec-decode K` enables telemetry; `IMP_MTP_PATTERN_LOG=1` for per-prediction diagnostics.

K=1 acceptance settles at **22-30 %** on Qwen3.6-A3B-NVFP4 with strict argmax-match. Phase 3.5 batched-verify is deferred because the ROI math doesn't pay below ~50 % accept (2× decode cost vs +0.22 tokens/cycle = 40 % slower than no spec-decode).

**Audit (2026-05-17, B1 in `mtp_audit_no_bug_2026_05_17.md`):** complete code-review of `src/runtime/mtp_forward.cu` (936 LOC) against DeepSeek-V3 / Qwen3.6 MTP spec found a 15-row structural-match checklist with all rows ✓ — **no engine bug.** The often-cited "FastMTP paper reports 70 % vanilla MTP K=1" is **the wrong baseline**: FastMTP measured MiMo-7B-RL (Xiaomi reasoning-tuned, MTP head jointly-trained with main, BF16 throughout), while imp measures Qwen3.6-A3B-NVFP4 (Alibaba's post-hoc MTP head, BF16-uploaded-as-FP16 head + NVFP4 main). Three independent multiplicative drivers explain the gap: (1) head quality (DeepSeek-V3 jointly-trained reports 85-90 %, Qwen3.6 is "vanilla quality" with no Alibaba training compute → 40-60 % standalone ceiling), (2) NVFP4 quantization drift between training and inference (5-15 pp), (3) strict argmax acceptance criterion. **22-30 % under these conditions is literature-consistent, not pathological.**

Re-eval triggers: better MTP head ships for Qwen3.6/Gemma-4/Llama-4 (e.g. FastMTP-recipe-trained); imp adds a model with proven jointly-trained MTP (DeepSeek-V3 with MLA, Qwen3-Next official, MiMo-7B with Xiaomi arch); HF-reference numerical comparison finds an imp divergence; NVFP4-aware MTP training recipe lands.

### FFN contextual sparsity — investigated and shelved

Idea (Vector 1 of the 2026-05-17 "Break the Memory Wall" research note): exploit the fact that `silu(gate[i]) * up[i]` is near-zero for a large fraction of intermediate-dim rows in trained transformers (the *contextual* sparsity that Deja Vu / PowerInfer / LLM-in-a-Flash all chase). If `|silu(g)*u|` is below some threshold, the corresponding column of `w_down` is multiplied by ~0 and contributes ~0 to the output — its 34-byte Q8_0 weight block could be skipped entirely (no HBM read).

**Probe (`src/compute/ffn_sparsity_probe.{h,cu}`, opt-in via `ffn.sparsity_probe`):** instrumentation kernel that counts, for each of 5 hard-coded thresholds `{0.005, 0.01, 0.02, 0.05, 0.1}`, the fraction of intermediate-dim rows under threshold per layer per token. Per-layer counters accumulate across the process and flush to stderr on engine destruction.

**Skip kernel (`src/compute/ffn_sparsity_mask.{h,cu}`, opt-in via `ffn.sparsity_threshold > 0`):** per-Q8-block packed-bit mask + mask-aware Q8_0 down_proj GEMV (kpar-layout, 4 warps per row). At `threshold = 0` the mask is all-1 and the kernel is bit-identical to `gemv_q8_0_q8_1_residual`.

**Result (2026-05-18, memo `ffn_sparsity_kpar_refuted_2026_05_18.md`):** Qwen3-8B Q8_0 probe confirms **25–52 % contextual sparsity** across layers — the *theoretical* upside is real and matches the literature. But the *measured* end-to-end gain from the masked kpar GEMV is **+0–1 % at realistic thresholds**, capped at **+11 % even at `threshold = ∞`** (i.e. skip every block). The kpar GEMV is warp-cooperative: 4 warps × 32 lanes share K, so the wallclock = slowest-thread time. Realistic skip patterns are spatially uncorrelated → every warp ends up doing roughly the same number of dp4a calls, regardless of how many *total* blocks the mask zeros out. The HBM bandwidth saving exists but is masked by the per-block branch + serialized scale-fetch overhead. **DEFER.** Code shipped as opt-in research artifact (both flags default off ⇒ zero hot-path cost) so the probe stays reproducible. A row-parallel rather than warp-cooperative skip layout might recover the win; not pursued today.

### FMHA cluster-launch — investigated and shelved (M5)

A Blackwell-style cluster-launch variant of the FMHA prefill kernel (DSMEM K-broadcast across a 2-CTA cluster, modelled after `paged_attention_cluster_kernel<HD>`) was implemented across three slices in May 2026: Slice 1 helper (#198), Slice 2 FP16 cluster kernel + dispatch (#200), Slice 2.2 FP8 variant (#202). End-to-end A/B across 4 NVFP4 MoE models (Qwen3.6-35B, Coder-30B, Gemma-4-26B, Qwen3-30B-Modelopt) on 2026-05-17 found the perf signal **noise-dominated**: ±20% same shape, opposite signs across re-runs of the same config; cuBLAS / thermal / scheduler variance drowned any cluster effect. Cluster output is **bit-identical to legacy** (`ClusterMatchesLegacy*` tests: max_abs = 0), so the kernel is sound; the maintainability cost (a parallel kernel + dispatch + dual test files) doesn't pay back. Default flipped OFF in #204 via `attention.no_fmha_cluster=true`; code retained as opt-in for future hardware where the signal might emerge. Memo: `m5_slice2_cluster_refuted_2026_05_17.md`.

### GEMM dispatch unification (R5) — DONE

The 21-parameter `gemm_dispatch_impl` god-dispatcher has been **retired**. The strategy-keyed `GemmKernel` registry (`src/graph/gemm_kernel_registry.h`) is the unconditional dispatch path. Slice 8.6 (final) closed the cross-axis refactor by migrating the QW7 dual-cache CUTLASS MXFP4 probe into the CUTLASS_NVFP4 handler, deleting the legacy switch (~247 LOC), and hoisting `mmvq_scratch_get_or_grow` into its own TU (`src/graph/gemm_scratch.{h,cu}`).

| Slice | Tier / scope | Status |
|---|---|---|
| 1 | FP16 interface + proof | shipped (#197) |
| 2 | FP8 prefill cache hit, M>1 | shipped (#209) |
| 3 | NVFP4 GEMV (M==1) | shipped (#210) |
| 4 | NVFP4 GEMM dequant fallback (M>1) | shipped (#211) |
| 5 | CUTLASS_NVFP4 dense | shipped (#212) |
| 6 | MXFP4 (GEMV + GEMM) | shipped (#213) |
| 7 | GGUF dp4a/mmvq, M==1 (8 qtypes) | shipped (#214) |
| 8 | Flip default ON + coverage audit | shipped (#215) |
| 8.1 | FP8 cache-miss → dequant fallback | shipped (#217) |
| 8.2 | Fused gemv fallback for Q6_K, Q8_0 | shipped (#220) |
| 8.3 | Q4_1 dead-code purge | shipped (#221) |
| 8.4 | `fp16_cache` for non-F16 weights (covered by Slice 1, no-op) | shipped (#225) |
| 8.5 | Raw-quant large-M dequant→cuBLAS catch-all | shipped (#225) |
| 8.6 | QW7 dual-cache CUTLASS MXFP4 migration + final `gemm_dispatch_impl` delete + `mmvq_scratch` TU hoist + flag retired | shipped |

QW7 decision: the dual-cache CUTLASS MXFP4 probe was **migrated, not deleted**. It only fires under explicit `--mxfp4-prefill` opt-in (off by default; documented in `docs/performance.md`), so production-trace evidence wasn't available; deleting would silently break the opt-in path. The probe now lives inside the CUTLASS_NVFP4 handler — when both `cutlass_nvfp4` and `cutlass_mxfp4` caches hit the same `weight.data`, the handler tries MXFP4 CUTLASS first and falls back to NVFP4 CUTLASS on `gemm_mxfp4_cutlass_sm120` failure. Same drop-through semantics as the legacy switch.

`gemm.use_kernel_registry` config field deleted (was an escape hatch during Slice 1-7 migration; redundant once the legacy path is gone).

### MoE prefill graph capture (M3 Phase 4) — DONE

Decode-side CUDA Graphs default-on gives +180-234% on NVFP4 MoE decode. The prefill-side equivalent **shipped 2026-05-17 via PR #218** after the M3 Phase 4 A/B sweep confirmed Blocker B (the CUTLASS 3.x grouped-GEMM hang under `cudaStreamCaptureModeGlobal`) is gone now that `graph_capture_mode = "relaxed"` is the default (#196). The sweep ran 3 NVFP4 MoE models × 4 capture-mode configs × 3 trials (36 datapoints, raw in `/tmp/imp-bench-results/`): no hang on any combination, decode tg flat ±0.3% across all configs. `runtime.prefill_graph` is now default `true`; opt-out via `--set runtime.prefill_graph=false`. Memos: `prefill_graph_blockers_2026_05_14.md`, `prefill_graph_cublaslt_blocker_2026_05_15.md`.

## Research interest

These are upstream features that would unlock real wins but haven't been integrated yet.

### CUDA 13.2 / CCCL 3.2 features

- ~~**Grouped GEMM with CUDA Graphs + device-side shapes**~~ — re-tested 2026-05-08 against cuBLAS 13.4.0.1 (`tools/analysis/probe_cublaslt_grouped.cu`): zero algorithms returned for FP16/BF16/FP8/NVFP4 on sm_120. Grouped layout API still marked Experimental in `cublasLt.h` 13.4 and only supported on datacenter Blackwell (SM100/B200), not consumer SM120. Re-run probe on each new cuBLAS release.
- ~~**`cub::DeviceTopK`**~~ — already wired in production (`src/compute/sampling.cu:834`, `cub::DeviceTopK::MaxPairs` for the `top_k > MAX_TOP_K=128` path with a small follow-up `DeviceRadixSort` over just the top-k results for top-p ordering).
- ~~**`cub::DeviceSegmentedReduce`**~~ — re-evaluated 2026-05-08, no applicable use case in imp. The 66× speedup claim applies to host-launched many-small-segments patterns (e.g. CUB benchmarks reducing thousands of fixed-size rows in one call). imp's per-head reductions are all already inside their owning kernel as warp-/block-level shuffle reductions (RMSNorm, attention softmax, MoE gate norm) — fused, optimal, and unrelated to DeviceSegmentedReduce's regime.
- ~~**`cudaMemcpyWithAttributesAsync`** (NUMA hint use case)~~ — shipped #131 at the recurring H2D paths (`src/memory/layer_offload.cu`, `src/runtime/vision_pipeline.cpp`) with `srcAccessOrder=Stream` + `srcLocHint=HostNumaCurrent`. The L2-persistence-hint use case (prefix-cache pinning without batched API) is still open.
- ~~**`add.f32x2` native PTX** (Blackwell)~~ — investigated #131. ptxas on consumer Blackwell (sm_120) accepts the legal PTX op but **decomposes it into 2× scalar FADD at SASS** — the vectorized hardware path is only exposed on datacenter Blackwell (SM100/B200). Helper `imp::add_f32x2` lives in `src/compute/ptx92_utils.cuh` for forward-compat with future toolkits / hardware. No SASS-level instruction-count reduction achievable on RTX 5090.

### PTX ISA 9.2

- ~~**`st.async.b128`** — 16-byte async stores for KV cache writeback~~ — **REFUTED 2026-05-17 at PTX availability gate.** ptxas (CUDA 13.2.1) rejects all `st.async` variants on sm_120: `shared::cluster.b32` → "Modifier rejected"; `shared::cluster.v4.b32` (b128) → "Modifier rejected"; `global.b32` → ptxas C7907 Internal compiler error. Adjacent paths that *do* assemble (`cp.async.bulk.tensor` TMA-store, raw `cp.async.bulk.global.shared::cta.bulk_group`) all require SMEM staging from registers — the same structural disadvantage that already sank TMA bulk in FMHA (`fmha_tma_lever_refuted_2026_05_14.md`, cp.async 0.31×–0.79× faster). Microbench skipped. Memo: `st_async_b128_unavailable_2026_05_17.md`. Re-eval triggers: ptxas upgrade lowering `st.async` for sm_120, port to SM100/B200, or KV-write becoming latency-bound (currently ~3–5 % of decode budget).
- **`cp.async.bulk` with `.ignore_oob`** — eliminates bounds-checking in TMA descriptors for variable seq lengths. **Sub-1 % expected on sm_120** since TMA itself is empirically slower than `cp.async` on consumer Blackwell; only relevant if a future workload re-enables TMA bulk on sm_120.
- **`cvt .bf16x2` ↔ narrow** (`.e2m1x2`, `.e4m3x2`) — packed FP4/FP8 pair conversion, 2× throughput. **Neutral on imp today**: FP4 cvt via `.b8`/`.b16` already works (`ptx_cvt_survey_2026_04_26.md`) and is wired in `nvfp4_quant_hw.cu` / `turboquant_fp4.cuh` / `kv_gather.cu`. The PTX 9.2 lever is the *packed `.bf16x2`* form — only useful with a BF16-native KV-write hot path, which imp doesn't have (KV cache is FP16-native).
- ~~**`.scale_vec::4X` with `.ue8m0`** for MXFP4 MMA~~ — **shipped PR #56**, QKT path default-on via `attention.fmha_blockscale = "auto"` (`attention_fmha_mxfp4_sm120.cu:591`, `gemm_grouped_nvfp4_smallM.cu:87`, `mxf4nvf4_qkt_validate.cu:127`). Per-16-element UE4M3 SFA/SFB. The PV-side Phase 3 follow-up was investigated and deferred (see below).
- **`mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`** — QKT path shipped in PR #56 (commit `208f25b`, default-on via `attention.fmha_blockscale = "auto"`); per-16-element UE4M3 SFA/SFB feed real `q_scales_fp8` / `k_scales_fp8`. Measured +1.8% Qwen3-4B MXFP4 at HD=128 (Phase 1 MMA is only ~15% of FMHA wall time, so 2.5× raw MMA → small visible delta). **Phase 3 (FP4 PV) — investigated and deferred (2026-05-17):** Phase 3a microbench (#240) showed single-level FP4 PV catastrophically fails on post-softmax (p99=797 % rel err). Phase 3b two-level accumulator (#241) reduces p99 to 260 % (still borderline, abs err bounded ≤ 0.48). Phase 3b.5 investigation found sm_120's sparse MMA is fixed 2:4 pattern, doesn't match our ~2:16 residual sparsity — realistic 2L-A speedup is ~2× not 4.6×, so e2e gain on HD=128 is **+2-3 %** at best, below the multi-week kernel work threshold. **DEFER** until an HD=256 MXFP4 model lands (PV becomes a larger FMHA fraction → more upside). Microbench infrastructure (`tests/bench/fp4_pv_bench.{h,cu}`) stays useful for re-eval. **Design memo:** `docs/plans/fp4_pv_phase3_design_2026_05_17.md`. **Findings memos:** `fp4_pv_phase3a/3b/3b5_*_2026_05_17.md`.

### Attention kernel research

- **Sawtooth Wavefront Reordering** ([arxiv:2601.16032](https://arxiv.org/abs/2601.16032)) — algorithmic L2-locality technique for FA: alternate K/V scan direction across consecutive Q tiles per SM (Q_0 forward, Q_1 backward, Q_2 forward …) to reduce reuse distance. **HW-agnostic surface**: WMMA + Shared Memory + standard global loads, no SM100-only primitives (no tcgen05/wgmma/cluster/DSMEM/TMA bulk) → cleanly portable to sm_120. Paper reports **+60 % causal / +13 % non-causal CuTile FA on GB10** (HD=64, seq=128K). **GB10→GB202 scaling argument** caps the upside: GB10 ~24 MiB L2 vs GB202 **96 MiB** L2 (3–4× larger) ⇒ working_set-vs-L2 ratio is 3–4× smaller on the 5090, so the L2-miss bottleneck Sawtooth fixes is structurally smaller. **Imp regime estimate**: 0 % on decode (single Q tile, no alternation possible); 0 % on prefill at ctx_len < 48 K (working set fits in L2); **+15–30 %** on prefill at ctx_len ≥ 64 K, HD=128, FP16 KV. The 64 K+ workload is rare on 32 GB consumer (Qwen3-8B-NVFP4 + 64 K KV ≈ 24 GB, leaves 8 GB headroom — viable but niche). **No open-source code** in paper — needs re-implementation in `attention_fmha_*_sm120.cu` / `attention_paged_*.cu` prefill path (~1 week). **Defer** below the three active levers (NVFP4 SmoothQuant Phase 1 diag, Qwen3.5-27B MXFP4 Phase A1, INT8 IMMA Phase 1 microbench). Re-eval trigger: a real user workload that runs prefill at ≥ 64 K context regularly.

### Long-context KV memory

The "decode is bandwidth-bound, sub-byte KV quant regresses decode" framing was partially obsolete after Lever 2 NVFP4 KV (PR landed 2026-05-07/08): NVFP4 storage with vectorized-PTX dequant is at parity with FP16 decode at 3.9× compression. BitDecoding (below) attempted to close the remaining gap via Tensor-Core dispatch on the dequantized math; both kernel phases shipped but failed to move decode tok/s at production context lengths — the original framing turned out to be right after all: **imp's decode is weight-bandwidth-bound, not attention-math-bound, on consumer Blackwell**.

- **K2 MLA (DeepSeek)** — latent vector replaces full K/V, ~–93% KV VRAM, 5.76× max throughput. **DEFERRED**: gates on adding DeepSeek-V2/V3 architecture support to imp; no MLA-arch model in scope today. Bonus paper [arxiv:2502.14837](https://arxiv.org/abs/2502.14837) proposes retrofitting MLA into non-MLA pretrained models — worth tracking but its own research project. Re-eval when imp adds DeepSeek-V2/V3 support or a calibration-only MLA recipe ships.
- **K5 Token eviction (H2O)** ([arxiv:2306.14048](https://arxiv.org/abs/2306.14048)) — Heavy-Hitter Oracle eviction by attention-score power-law; 5–20% retention, ≤20× memory. **POSSIBLE BUT QUALITY-RISKY**: well-documented retrieval-task degradation since 2023 (RULER, NIAH, multi-hop QA). Build only if VRAM-pressure becomes the bottleneck on contexts that NVFP4 KV alone can't fit. Re-eval if a successor with retrieval-quality fix lands (Q-Hitter, SnapKV, PyramidKV are candidates). **Design memo:** `docs/plans/k5_h2o_eviction_design_2026_05_17.md` (defer indefinitely; the engineering cost of any quality-preserving successor matches K8 CPU offload which has zero quality risk).
- **K8 CPU offload** — async prefetch for cold tokens, enables 100K+ context. **Design memo:** `docs/plans/k8_cpu_offload_design_2026_05_17.md`. Recommendation: defer with Phase-1 spike (2-day nsys profiling). **Important correction:** PCIe is NOT trivially un-bottlenecked as the roadmap framing implied — at full-attention semantics over a 100K cold tail, the kernel touches thousands of cold blocks per token (~40 ms vs the 5 ms decode budget). K8 only pays off in attention regimes that naturally skip the cold tail (SWA, H2O, retrieval).

### KV cache compression research

- **BitDecoding** ([arxiv:2503.18773](https://arxiv.org/abs/2503.18773), HPCA 2026) — **kernel work complete, deferred on empirical grounds.** Tensor-Core decode on dequantized NVFP4 KV; paper claims 8.6× over FP16 FlashDecoding-v2 on Blackwell, 3× E2E on Llama-3.1-8B 128K. imp's current NVFP4 KV (Lever 2) gets the VRAM win on CUDA cores → parity-only on decode tok/s; BitDecoding was supposed to close that gap. **What shipped:**
  - **Phase 1** (Q·K via WMMA) — PR #144 (2026-05-09), opt-in via `kv_cache.bitdecoding_qk = true`. SASS audit confirms 24 HMMA in TC kernel vs 0 HMMA in legacy scalar.
  - **Phase 2** (P·V via WMMA) — PR #145 (2026-05-09). Both attention GEMMs on Tensor-Core. Kernel: `src/compute/attention_paged_nvfp4_tc.cu` (WMMA P·V at lines 367-411 + Phase 3 residual replay at 447-590).
  - **Phase 3** (residual FP16 cache for newest N tokens) — PR #147/#148, opt-in via `kv_cache.bitdecoding_residual_tokens = N`.

  **Empirical result (2026-05-14 long-context eval, memo `bitdecoding_long_context_eval_2026_05_14.md`):** 0% gain across `pp ∈ {512, 4K, 8K, 16K}` on Qwen3-4B/8B and Gemma-4. The kernel routes correctly through the TC path; attention math just isn't the bottleneck at consumer-Blackwell scale — decode is bound on weight reads, not on Q·K / P·V. Stays opt-in, default OFF. Re-eval becomes useful only if a workload appears where attention math dominates decode wall-time (e.g. very long ctx + tiny weights). Design memo for revisits: `docs/plans/bitdecoding_phase2_design_2026_05_17.md`. Ref impl: [OpenBitSys/BitDecoding](https://github.com/OpenBitSys/BitDecoding).
- **DeltaKV** ([arxiv:2602.08005](https://arxiv.org/abs/2602.08005)) — residual-based sparse KV compression: encode tokens as residuals against retrieved historical references; 29% KV memory, 2× throughput. **DEFERRED**: bundled with [Sparse-vLLM](https://github.com/CURRENTF/Sparse-vLLM) (near-fork, not a library); imp would need to re-architect the dense paged KV layer. Marginal win over the NVFP4 + BitDecoding stack at high engineering cost. Re-eval if imp pushes past 256K context where DeltaKV's long-range similarity advantage grows, or if a paged-vLLM-compatible reference appears.

Detailed per-item evaluation lives in `kv_research_grade_eval_2026_05_09.md` (memory file).
