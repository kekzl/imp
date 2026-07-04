# imp Mission Journal — Autonomous "best engine on RTX 5090" run

> Append-only. Survives context resets. Newest entries at the BOTTOM of each section,
> but the **RESUME HERE** block at the top is always kept current.

---

## RESUME HERE (always current)

**Session:** 2026-06-18. Work lands via PRs off main (branch then `gh pr create --base main`).
**Phase:** Post-NVFP4-limit campaign. main green at `40803986` (PR #736). The 05-30 → 06-13 arc
(40+ PRs) was logged in `CHANGELOG.md` + PR descriptions, not in this journal — see the catch-up
entry at the bottom of the Log for the strategic summary. 06-18 added 3 wins (load/serving) and
exhausted the decode frontier — see the 2026-06-18 Log entry.
**On resume:** read this block + `CHANGELOG.md [Unreleased]` + the `perf_baselines_detail_2026_06_11`
memory. Rebuild `make build`. Trust `tests/perf_baseline.json` + `BENCHMARKS.md` for numbers
(SHA-anchored); decode is the A/B signal (prefill 2.6× restart variance, host days ±8-15%).
NOTE: force-push + `git reset --hard` are BLOCKED in this env — land via fresh branch + merge,
not rebase+force. Profile MoE decode at **pp512** not pp128 (split-K only engages at ctx≥512).

**WHERE THINGS STAND (NVFP4 = the priority path):** imp wins dense + MoE decode best-in-class
vs llama.cpp AND vLLM, wins TTFT everywhere, and wins MoE pp2048 vs vLLM. The whole prior
cross-engine PPL gap turned out to be tokenization, now closed for 4 families (#657, byte-identical
to HF). Decode sits at the HBM roofline; FA2 prefill is instruction-mix-bound near its practical
ceiling. The **single remaining competitive gap is pp4096 NVFP4 prefill ~1.19-1.25× behind vLLM**,
and every *bounded* lever for it was empirically closed in the 06-12 campaign.

**OPEN WORK (most big levers are refuted — see roadmap "Investigated and shelved"):**
1. **pp4096 NVFP4 prefill ~1.19-1.25× vs vLLM.** Cross-Tile (+9/+15% refuted), Grouped-GEMM tile
   axis, chunk-4096, and fp8-QK (format-intrinsic, #511/#681) are all closed. Only surviving idea:
   scaled fp8-KV storage with f16 compute (vLLM's actual win) — see #2.
2. **kv-fp8 storage default-on.** −768 MiB VRAM, +0.83% PPL, the −35% MoE tax was diagnosed and
   removed (#682). Blocked on building long-context quality gates per model family before honoring
   `kv_cache_quant_algo=FP8` by default.
3. **GGUF-only gaps (deprioritized — recommend NVFP4):** Q4_K_M prefill MMQ (evidence-refuted, see
   roadmap), MoE/hybrid GGUF decode (dp4a, ~−13%).
4. **Open model bugs:** Qwen3.5-27B/4B MXFP4 (blocked — community files never verified in any engine,
   27B not even local); Qwen3.6-27B-VL NVFP4 (wontfix — ~30 GB OOMs a 32 GB card). gpt-oss GGUF MoE
   NaN FIXED (#690); Gemma-4 Q4_K_M code-gen drift no longer reproduces (verified 06-13, see roadmap).
5. **Small cleanups:** ms_ref loader slab (last VRAM-audit item); spec-decode default-on needs an
   engagement heuristic (opt-in tg128 −15% on short / draft-poor prompts).

**GPU:** RTX 5090, 32607 MiB, water-cooled (never warm — no cooldown waits between benches). Host
`nvidia-smi` + host nsys work (WSL2; nsys recipe in memory nsys-host-to-container, `--no-cuda-graphs`
+ `--trace=cuda`, --user root, /tmp/nsys_out chmod 777). Healthy load = ~2850 MHz SM / 13801 MHz mem
/ ~500 W — sample DURING the bench before trusting a cross-day decode delta. Warm clocks >1s before
timed reps (built-in `Warmup...` is too few iters); nsys with CUDA Graphs ON hides captured kernels.

**Canonical references (not this journal — it lagged the 06-xx campaign):**
- Perf gate: `tests/perf_baseline.json` (3% decode / 5% prefill). Reproducible numbers (SHA-anchored):
  `BENCHMARKS.md`. User-visible changes: `CHANGELOG.md`. Current focus: `docs/roadmap.md`.
- Detailed campaign index lives in the agent's private memory (`perf_baselines_detail_2026_06_11` et al.).
- North-star: Qwen3-14B Q6_K decode @ctx2048 = 157.71 tok/s.

---

## Model Inventory (verified `/home/kekz/models/` 2026-05-29)

GGUF: gemma-3-12b-it-Q4_K_M, gemma-4-26B-A4B-it-{Q8_0,UD-Q4_K_M}, gemma-4-31B-it-Q4_K_M,
  Qwen3-14B-Q6_K, Qwen3-8B-Q8_0, Qwen3.5-4B-mxfp4, qwen3.6-35B-A3B-gguf.
NVFP4 SafeTensors: Gemma-4-26B-A4B-it-NVFP4, Gemma-4-31B-IT-NVFP4, Nemotron-3-Nano-30B-A3B-NVFP4,
  Nemotron-Labs-3-Elastic-30B-A3B-NVFP4, Phi-4-reasoning-plus-NVFP4, Qwen3-14B-NVFP4,
  Qwen3-30B-A3B-NVFP4-Modelopt, Qwen3.6-35B-A3B-NVFP4, Qwen3-8B-NVFP4-cortecs,
  Qwen3-Coder-30B-A3B-Instruct-FP4, Qwen3-30B-A3B-Q4_K_M.
Also: $REPO/models/ has Q8_0 verify baselines.

---

## Scoreboard (current snapshot)

imp, 2026-05-29, reps=8, ctx≈512, CUBLAS_WORKSPACE_CONFIG=:4096:8. tok/s. (partial — sweep running)

| Model | Family | Quant | pp512 | tg128 (decode) |
|---|---|---|---|---|
| Qwen3-8B | dense | Q8_0 | 7971 | **274.4** |
| Qwen3-8B | dense | NVFP4 | 21581 | 239.6 |
| Qwen3-14B | dense | Q6_K | 4965 | **155.4** |
| Qwen3-14B | dense | NVFP4 | 17961 | 147.6 |
| Qwen3-30B-A3B | moe | NVFP4 | 16099 | 170.7 |
| Qwen3-30B-A3B | moe | Q4_K_M | 3851 | **276.5** |
| Qwen3-Coder-30B-A3B | moe | NVFP4 | 17139 | 170.9 |
| Qwen3.6-35B-A3B | hybrid | NVFP4 | 8373 | 151.2 |
| ...rest pending | | | | |

**llama.cpp MEASURED on THIS machine (2026-05-29, commit 19e92c3, CUDA 12.8, sm_120, -fa 1,
-ngl 999, -r 5). Same GGUF weights as imp:**

| Model | Quant | imp pp512 | llama pp512 | pp winner | imp tg128 | llama tg128 | tg winner |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | Q8_0 | 7971 | 14068 | llama 1.77× | 274.4 | 159.7 | **imp +72%** |
| Qwen3-14B | Q6_K | 4965 | 6367 | llama 1.28× | 155.4 | 113.7 | **imp +37%** |
| Qwen3-30B-A3B | Q4_K_M | 3851 | 9232 | llama 2.40× | 276.5 | 316.7 | llama +15% |
| Qwen3.6-35B-A3B | Q4_K_M | 3625 | 6290 | llama 1.73× | 157.9 | 228.6 | **llama +45%** |
| Gemma-4-26B | Q4_K_M | 4278 | 10457 | llama 2.44× | 252.0 | 212.2 | **imp +19%** |
| Gemma-3-12B | Q4_K_M | (bogus) | 8850 | — | (bogus) | 142.3 | — |

**REFRAME — the real picture (this overrides stale "imp wins decode +24-86%" memory):**
- imp DOMINATES dense GGUF decode (+37-72%). Already best-in-class there.
- imp LOSES ALL GGUF prefill 1.28-2.44× (worst: MoE/Gemma Q4_K_M ~2.4×). dequant→cuBLAS vs
  llama MMQ. Consistent, biggest gap. Hard (custom MMQ kernel, weeks) — memory q4k_prefill.
- imp LOSES MoE/hybrid GGUF decode: Qwen3.6-35B −45% (158 vs 229!), Qwen3-30B −15%. NEW finding.
- NVFP4 path is UNCONTESTED by llama.cpp (no NVFP4 support); imp NVFP4 prefill 16-24k crushes
  llama GGUF. NVFP4's only competitor is vLLM (broken on sm_120 MoE-NVFP4 per research). So
  GOAL.md's "NVFP4 = recommended path" is well-supported; making NVFP4 decode/MoE great (LEAD-1/2)
  is the highest-leverage, most-defensible work. Need measured vLLM NVFP4 to confirm prefill lead.
- Gemma-3-12B Q4_K_M row is bogus (tg=1588, known degeneration early-stop). Re-measure / flag.

---

## Backlog (prioritized gaps) — initial, pre-profiling

**LEAD-1 (decode, all NVFP4 dense): native NVFP4 decodes SLOWER than GGUF on same model.**
Qwen3-8B Q8_0 274 vs NVFP4 240 (−13%); Qwen3-14B Q6_K 155 vs NVFP4 148 (−5%). Same arch.
NVFP4 is the *recommended* fast path per GOAL.md, yet it's slower than GGUF at decode.
Hypothesis: native-NVFP4 checkpoints keep lm_head/embeddings/some-proj in FP16; GGUF path
quantizes output_proj to NVFP4 (pre_dequant_phase3:101). NEEDS nsys kernel-mix diff.

**LEAD-2 (decode, NVFP4 MoE): biggest gap. NVFP4 MoE decode 170 vs Q4_K_M MoE 276 (−38%).**
Qwen3-30B-A3B + Coder-30B both ~170 NVFP4 vs 276 Q4_K_M. GGUF MoE gets a dedicated
Q4_K×Q8_1 dp4a `can_decode_fast` path (executor_forward_moe.cu:92); NVFP4 MoE uses per-expert
FP16-activation GEMVs (gemv_nvfp4_moe_*). NEEDS nsys to confirm cause. Possibly biggest win.

**LEAD-3 (strategic, multi-week): spec-decode / EAGLE / MTP** — 1.5-4× batch=1 multiplier
competitors ship. imp parked at 196 tok/s ceiling. Highest ceiling, highest cost. Revisit.

**LEAD-4 (prefill MoE):** vLLM MoE-NVFP4 broken on sm_120 (Marlin fallback) per research →
imp may already win MoE prefill in practice. Confirm with measured vLLM later.

NOTE: imp may already be best-in-class on most DECODE cells vs llama.cpp. If confirmed, the
product gap is "make NVFP4 (recommended path) ≥ GGUF speed" (LEAD-1/2) + prefill + spec-decode.

---

## Log (timestamped, append-only)

### 2026-05-29 — Iteration 4: GGUF prefill profiled; vLLM bringup dispatched
- **GGUF prefill profile (Qwen3-8B Q8_0 pp512, eager):** CUTLASS f16 GEMM 40.5% + 64x256/64x64
  variants ~17% (≈58% GEMM total) + **dequant_q8_0_kernel 30.3%** (Q8_0→FP16 before the GEMM).
  llama.cpp MMQ matmuls on quantized data directly → skips the 30% dequant AND halves weight
  bandwidth. Structural gap = need a direct quantized-matmul (MMQ) prefill kernel. Prior imp MMQ
  attempts (mmq_q4k_v2) gave −4% e2e for DECODE; PREFILL (M=512) is the untried regime where INT8
  IMMA / CUTLASS mixed-input could win — but it's multi-day. DEPRIORITIZED: NVFP4 prefill (imp
  16-24k) already crushes llama GGUF and is the recommended path; GGUF prefill only affects
  GGUF-only users. Documented for a future dedicated MMQ-prefill effort.
- Dispatched bg agent to stand up **vLLM NVFP4** (dense Qwen3-8B/14B + MoE 30B) for the one
  NVFP4-capable competitor comparison — defines whether imp's NVFP4 prefill lead is real.
- **Gemma-3-12B Q4_K_M degeneration memory is STALE — model works** ("Paris. It's also its
  largest city and a global center for art, fashion, gastronomy, and culture", 126 tok/s). Bogus
  scoreboard tg=1588 was a transient artifact. True: tg128=126, pp512=4181. But imp LOSES it to
  llama.cpp (decode 126 vs 142 −11%, prefill 4181 vs 8850). Profile: all-Q4_K, decode is dp4a FFN
  GEMVs (53%, bandwidth-bound ~roofline) + Q4_K lm_head dp4a. `nvfp4_decode_all` REFUTED here —
  Q4_K is already 4.5 bits, NVFP4 gives 0 bandwidth win ("no eligible weights, all ≤4.5 bits").
  So Gemma-3-12B decode gap = dp4a-kernel efficiency vs llama MMQ-decode (large vocab + sliding
  window) — hard, model-specific, no clean win. Same class as Qwen3-30B Q4_K_M decode (276 vs 317).
- **Decode front status: WON on NVFP4 (recommended path) across the fleet. Remaining decode losses
  are GGUF-Q4_K-only (dp4a vs MMQ) and modest. Remaining clear gap = GGUF prefill (MMQ, multi-day).**

### 2026-05-29 — Iteration 3: LEAD-2 LANDED (NVFP4 MoE decode, zero-copy fast path) ✅
The "15 GiB blocker" was a misread (data is borrowed by CUTLASS = already resident). Diagnostic
proved per-expert NVFP4 packed DATA is contiguous in VRAM; native scales resident but NOT
contiguous. Fix: `cache_moe_native_nvfp4` now BORROWS the contiguous expert data (zero copy) +
copies only the small scales (~1/16) into a contiguous buffer + a tiny per-expert tensor_scales
array, building an NvFP4MoEQuantResult that points at the existing data. This engages the existing
fast `gemv_nvfp4_moe_*` decode kernels (base+stride) instead of the CUTLASS grouped GEMM (which
under-utilizes the GPU at M=1). Added `borrowed` flag to NvFP4MoEQuantResult so free_nvfp4_moe_result
skips cudaFree on borrowed/VRAMAllocator pointers (fixed a teardown double-free). Config
`gemm.nvfp4_moe_decode` (default on); `IMP_NO_NVFP4_MOE_DECODE=1` disables.

**Measured (pp512, reps 6-8) — decode tok/s before → after:**
- Qwen3-30B-A3B NVFP4: 170.7 → **307.5 (+80%)** — now > Q4_K_M 276, ≈ llama.cpp 317.
- Qwen3-Coder-30B NVFP4: 170.9 → **307.8 (+80%)**.
- Gemma-4-26B NVFP4: 160.2 → **258.1 (+61%)** — **beats llama.cpp Q4_K_M 212 by +22%**.
- Qwen3.6-35B-A3B hybrid NVFP4: 150.2 → **228.5 (+52%)** — **now matches llama.cpp 229** (was −31%).

**Correctness ALL GREEN:** greedy output token-for-token IDENTICAL ON(gemv) vs OFF(CUTLASS) on 30B
(numerically equivalent path); all models coherent incl. hybrid (GDN state intact) + Gemma-4 ("Paris");
zero error-scan hits; full GPU test suite PASS; flag A/B clean (307 vs 167); prefill unaffected; no OOM.
**Finding: MoE expert decode (not just FP16 SSM) was the dominant hybrid bottleneck — corrects
qwen3_6_35b_a3b_nvfp4_full_profile.** NVFP4 MoE/hybrid decode is now competitive-to-winning vs llama.cpp.


### 2026-05-29 — Iteration 2: attention lead investigated (non-issue) + LEAD-2 de-risked
**Attention "lead" REFUTED as a real-context win (good — avoided optimizing a non-problem):**
- pp128 profile showed paged_attention_gqa = 25% at 33µs/call. Root: gqa kernel grid =
  (batch × n_kv_heads); Qwen3-30B has 4 KV heads → only 4 blocks on 170 SMs (under-occupancy).
- Diagnostic build confirmed: at ctx128, num_ctx_blocks=5 → max_useful_splits=5/4=1 → num_splits=1
  → split-K declines (correctly — too few KV blocks) → slow gqa kernel.
- BUT at ctx≥512 (scoreboard + real usage), num_ctx_blocks≥32 → split-K fires → attention fast.
- **Lesson: profile MoE decode at pp512, NOT pp128 (short ctx mis-attributes time to attention).**
  The gqa under-occupancy only hurts the first ~256 decode tokens after a very short prompt — low
  real impact, not worth a risky kernel change. Diagnostic reverted; tree clean.

**LEAD-2 (NVFP4 MoE decode) DE-RISKED — the "15 GiB blocker" was a misread:**
- `convert_nvfp4_to_cutlass` does `dst.data = src.packed_data // borrowed, not owned` — the
  standard-packed per-expert NVFP4 DATA is already resident (CUTLASS only adds sfatom scales).
  The ~15 GiB "blocker" in `cache_moe_native_nvfp4` was the cost of a redundant CONTIGUOUS COPY,
  not a fundamental need. Per-expert data pointers already exist (CUTLASS device-args d_B_ptrs).
- So the real fix is a **gemv_nvfp4_moe variant taking per-expert POINTER ARRAYS** (no contiguous
  copy, ~0 extra data VRAM), reading the existing per-expert packed data + native micro-scales.
- DATA MAP (confirmed): per-expert registry handle (CUTLASS_NVFP4 tier) `payload.cutlass_nvfp4.weight`
  = the packed FP4 data (borrowed from native src.packed_data — resident). NATIVE per-row FP8
  micro-scales are at the original expert Tensor's `.scales` sidecar (Phase-0 set `tmp.micro_scales
  = w.scales`, pre_dequant_phase0:363) + `.tensor_scale`. payload.cutlass_nvfp4.sf is sfatom layout
  (WRONG for gemv — need the native .scales).
- IMPLEMENTATION SPEC (next session, ~300-400 LOC, bounded):
  1. Confirm native expert `.scales` sidecars are still resident at decode (diagnostic: non-null +
     plausible). Nothing obvious frees them; Phase-0 promotes, CUTLASS borrows.
  2. At load build device pointer arrays per layer per projection: {packed_data_ptr[e],
     native_ms_ptr[e], tensor_scale[e]} for e in 0..n_experts. Tiny VRAM (~128×3×24 B/layer).
  3. Add gemv_nvfp4_moe_{decode,gate_up,swiglu}_ptrs kernels: index expert via expert_indices →
     pointer array (vs current contiguous base+stride). Reuse the existing dot_micro_block /
     warp_k_loop math from nvfp4_gemm.cu verbatim.
  4. Wire run_moe_decode_fast / can_decode_fast to take this path when experts are CUTLASS_NVFP4
     (native NVFP4) + M==1. Pointer arrays are static → CUDA-graph safe.
  5. Validate at **pp512** (not pp128): decode tok/s + 200-tok coherence + no degeneration.
- Expected: NVFP4 MoE decode ~175 → toward/past Q4_K_M's ~276 (lm_head already NVFP4 via #465).


### 2026-05-29 — Session start
- Re-read GOAL.md, CLAUDE.md, perf_baseline.json. No existing journal.
- GPU idle & cool. Kicked off Docker build.
- Goal restated: batch=1 latency-first, decode tok/s is primary metric; must beat
  llama.cpp/vLLM/SGLang/ExLlamaV3/MLC on every supported arch×quant on sm_120a.

### 2026-05-29 — Build green, correctness baseline green
- `docker build -t imp:test .` → exit 0. Binaries: imp-cli, imp-server, imp-tests, test-*.
- `imp-tests-unit`: 34 passed / 3 skipped. `imp-tests` (GPU): all suites PASS
  (84+139+34+83+73), 0 FAILED; skips are model-path-dependent only. **main is correct.**
- Bench rig validated vs perf gate: Qwen3-8B Q8_0 pp512=7927 (gate 7736, +2.5% cuBLAS var),
  tg128=273.99 (gate 276.59, −0.9%). Measurement trustworthy.

### 2026-05-29 — Competitive research (web, this session)
- **llama.cpp CUDA is the only credible batch=1 DECODE threat.** It beats vLLM at batch=1
  decode on Blackwell (measured 3rd-party: 134 vs 89 tok/s Qwen3.5-9B Q4_K_M vs BF16).
- vLLM wins PREFILL (3.4×) and batched throughput; loses batch=1 decode. vLLM MoE-NVFP4
  backend selection is BROKEN on sm_120 (falls back to Marlin) — opportunity.
- B200-only (NOT sm_120): tcgen05, TMEM, wgmma, TMA warp-specialized grouped GEMM.
  Confirms CLAUDE.md. SGLang's headline FP4 wins are all sm_100/B200.
- ExLlamaV3 (EXL3/QTIP): consumer batch=1 contender but dequant-to-FP16, NO native FP4 TC
  path, sm_120 support is a monkey-patch. imp's native NVFP4 decode cache should beat it.
- **Biggest strategic lever per research = speculative decoding (EAGLE-3/P-EAGLE/MTP),
  1.5–4× batch=1.** Competitors ship it; imp's spec-decode is parked at 196 tok/s ceiling.
  This is the one front where vLLM+P-EAGLE / SGLang+EAGLE-3 could beat imp on Qwen3-class.
- Published llama.cpp 5090 batch=1 decode (InsiderLLM 2026-03-25, Q4_K_M): Qwen3-8B 185.9@4k,
  Qwen3-14B 123.8@4k, Qwen3-30B-A3B 234.3@4k. (To be reproduced on THIS machine.)

### 2026-05-29 — LEAD-1 implemented (NVFP4 FP16-lm_head decode cache)
- Confirmed via safetensors headers: Qwen3-8B/14B/30B NVFP4 all store `lm_head.weight` BF16.
- Confirmed via nsys (eager, --no-cuda-graphs): cuBLAS FP16 lm_head GEMV = 19.3% of decode
  GPU time, 0.78 ms/call, Qwen3-8B NVFP4. 2nd-biggest kernel.
- Fix (commit pending): new `GraphExecutor::nvfp4_decode_cache_fp16_lm_head_()` in
  pre_dequant_phase3 — quantizes an FP16/BF16 output_proj → NVFP4 via
  quantize_fp16_to_nvfp4_async, stores in wcache_.nvfp4[lm.data]; forward_logits
  `lm_nvfp4_secondary` hook reads it (zero forward-pass change). Gated: skips GDN/SSM-hybrid
  models (LM-head NVFP4 degrades recurrent quality). Config `gemm.nvfp4_lm_head` (default on),
  env `IMP_NO_NVFP4_LM_HEAD=1` to disable for A/B. Same transform the GGUF Q8_0 path already
  applies to output_proj (proven quality-safe + passes perf gate).
- Predicted: Qwen3-8B NVFP4 239→~279 (beats Q8_0 274 & crushes llama.cpp 160). Verifying.

### 2026-05-29 — LEAD-1 VERIFIED + landed ✅
A/B Qwen3-8B NVFP4 (identical conditions, reps=10, fix default-ON vs IMP_NO_NVFP4_LM_HEAD=1):
- **Decode: 238.56 → 276.80 tok/s (+16.0%)** — now > Q8_0 (274) & **+73% vs llama.cpp (160)**.
- **Prefill: 25514 → 29949 tok/s (+17.4%)** bonus (last-token logits proj is M=1 GEMV in prefill too).
- LM head log: "quantized FP16 [151936 x 4096] → NVFP4 (333.8 MiB)".
Correctness gates ALL GREEN:
- Long 200-tok gen: eloquent, coherent, zero repetition/degeneration. Error scan empty.
- Factual + ocean prompts coherent. Output diverges from BF16 by 1 benign token-flip (expected).
- Q8_0 GGUF gate model UNCHANGED (pp 7954, tg 273.96 — lm_head log absent, Q8_0 head untouched).
- Full GPU test suite: all PASSED, 0 FAILED.
Same NVFP4-quantize-output_proj transform the production GGUF Q8_0 path already uses → low risk,
confirmed. Applies to ALL non-hybrid native-NVFP4 dense models (Qwen3-8B/14B, Phi-4). Committing.
NEXT: re-sweep NVFP4 dense models to capture the win; then LEAD-2 (NVFP4 MoE decode 170 vs 276).

### 2026-05-29 — Scoreboard sweep launched
- Wrote `scripts/scoreboard.sh` (committable harness): sweeps 14-model matrix, pp512+tg128,
  reps=8, CUBLAS_WORKSPACE_CONFIG=:4096:8 → `docs/scoreboard.tsv`.
- Launched sweep (bg) + llama.cpp-on-sm120 build/bench agent (bg, builds CUDA 12.8 image).
- Early results: Qwen3-8B Q8_0 pp512=7970/tg128=274 · Qwen3-8B-NVFP4 pp512=21581/tg128=239.

### 2026-05-29 — Iteration 4b: GGUF prefill via IMMA scoped — architecturally capped
- The existing INT8 IMMA Q4_K MMQ kernel (`gemm.q4k_imma_enabled`, default off) ONLY engages at
  **M≥1024** (dense, %64). At pp512 it never fires (my first A/B showed no change — M=512).
- At **pp2048** it DOES fire: Gemma-3-12B 4983 → **5375 (+7.9%)** prefill. Real but modest; coherent.
  Variance-sensitive (cuBLAS prefill ±2.6×) — NOT shipped without rigorous cooldown/multi-trial/
  multi-model validation (defer until GPU free; vLLM agent holds it).
- **Phase-2B ceiling doc** (docs/archive/.../2026-05-18-q4k-imma-phase2b-ceiling.md): IMMA caps at
  **40 TOPS = 4.3% of 931 TOPS peak** — scale-apply (FP16→FP32 + FMA between MMAs) serializes and
  under-issues the tensor cores. Reaches ~dequant→cuBLAS parity only; beating llama MMQ (which
  defers/batches scale-apply) needs a FUNDAMENTALLY different kernel. So GGUF prefill = genuine
  multi-day kernel research, not a quick win.
- **CONCLUSION this session: decode WON on NVFP4 (shipped #465+#469). All remaining gaps are
  hard/research (GGUF prefill MMQ architecturally capped; GGUF Q4_K decode dp4a-bound) or pending
  (vLLM NVFP4 prefill comparison, GPU-occupied). Next concrete action gated on vLLM result.**

### 2026-05-29 — Iteration 5: vLLM measured → NVFP4 PREFILL is the next real gap
vLLM 0.21.0 (FlashInfer-CUTLASS NVFP4, sm_120 verified) batch=1, measured this session:
- **DECODE: imp NVFP4 BEATS vLLM on all (with LEAD-1/2 shipped):** Qwen3-8B 277 vs 142 (+95%),
  Qwen3-14B 166 vs 97 (+71%), **Qwen3-30B-A3B MoE 307 vs 203 (+51%)**. LEAD-2 FLIPPED the MoE
  (vLLM beat imp 203 vs 158 PRE-fix). imp NVFP4 decode = best-in-class vs llama.cpp AND vLLM. ✅
- **CORRECTION: vLLM MoE-NVFP4 is NOT broken on sm_120** (vLLM 0.21.0 uses FLASHINFER_CUTLASS MoE,
  not Marlin — the prior research/memory "broken on sm_120" is STALE). imp still beats it on decode.
- **PREFILL: imp NVFP4 LOSES to vLLM** (imp pp2048 vs vLLM ~2267-tok prompt):
  Qwen3-8B 26780 vs 35080 (−24%), **Qwen3-14B 14581 vs 24440 (−40%)**, Qwen3-30B 25800 vs 29450 (−12%).
  imp Qwen3-14B pp512=17961 but pp2048=14581 (DROPS) → attention quadratic cost dominates long prefill;
  vLLM's FlashInfer prefill attention likely the edge. **THIS is the next target: NVFP4 prefill, esp.
  the dense attention prefill path.** (Caveat: re-measure imp with cooldown/multi-trial — cuBLAS/CUTLASS
  prefill ±2.6× — but the gap is consistent + large.)
- Decode mission goal essentially MET on NVFP4 (recommended path). Frontier now = NVFP4 prefill.

### 2026-05-29 — Iteration 5b: NVFP4 prefill gap localized → attention, not GEMM
Qwen3-14B NVFP4 prefill profile (pp2048, eager): CUTLASS NVFP4 GEMM 39% (compute, competitive w/
vLLM) + **attention ~37%** (fmha_sm120_fp8 20% @0.71ms + causal_softmax_inplace 8% + attn GEMMs 9%).
So the vLLM NVFP4-prefill lead is mostly the ATTENTION path: imp uses FP8-FMHA + materialized-S
softmax (at the fmha_prefill_threshold=2049 boundary) vs vLLM's fused FlashInfer. Routing A/B (pp2048):
- force-FMHA (threshold=1): 12806 (WORSE — re-confirms FMHA-rewrite-refuted).
- chunked 256/512: ~16100 vs default 15104 (+6.6%) / vs single-chunk 14179 (+14%). MODEST, NOISY
  (within cuBLAS ±2.6% band); does NOT close the vLLM −34% gap. Default already chunks (not worst-case).
- CONCLUSION: NVFP4 prefill gap = fused-attention-prefill kernel (FlashInfer-class), a multi-day
  kernel effort prior FMHA work found hard. Chunk-tuning is a borderline lead, not shipped (needs
  rigorous multi-trial + risks per-arch-default regressions on other models/contexts).

## SESSION SUMMARY (2026-05-29) — primary mission metric MET
**Batch=1 DECODE (GOAL.md primary metric) on the recommended NVFP4 path is now BEST-IN-CLASS vs
both llama.cpp AND vLLM on every measured dense/MoE/hybrid model** — via two shipped+merged wins:
LEAD-1 (#465, lm_head NVFP4, +8-16% dense) and LEAD-2 (#469, MoE decode, +52-84%; flipped the 30B
MoE from −29% vs vLLM to +51%). Remaining gaps (all secondary-metric / hard kernel research, for
future sessions): NVFP4 prefill vs vLLM (−14-40%, fused-attention kernel), GGUF prefill vs llama
(MMQ, architecturally capped), GGUF Q4_K decode (dp4a, minor), spec-decode (multi-week).

### 2026-05-29 — Iteration 6: server robustness audit (mission §6) — PASS, no bugs
Probed imp-server (Qwen3-8B NVFP4, port 8081) for the §6 completeness bar:
- Malformed JSON / missing messages / empty body / wrong content-type / negative max_tokens → graceful
  HTTP 400 (no crash). Normal chat → 200. Huge max_tokens (1e9) → 200 (clamped to context).
- **Both API dialects work:** /v1/chat/completions AND /v1/messages (Anthropic) → 200.
- **Context overflow → clean 400** with proper OpenAI error format: `{"error":{"message":"Prompt
  exceeds context window (60009 tokens >= 40960 max)","type":"invalid_request_error"}}`.
- **Continuous-batching under concurrent load: 12/12 simultaneous requests → 200, server stable,
  health green throughout.** No crashes, no hangs.
- VERDICT: server robustness/completeness is solid — claimed behavior is TRUE. No fix needed.
  (Not stress-tested: multi-hour sustained-load memory stability, SSE-stream edge cases, true-OOM
  model-load path — candidates for a future deeper soak test.)

### 2026-05-29 — Iteration 7: NVFP4 prefill win SHIPPED (#474, S-matrix cap 256→384)
First crack at the NVFP4-prefill frontier landed. Root cause: the cuBLAS-attention prefill path
(faster than FMHA at the boundary) is gated by the S-matrix buffer (was hard-capped 256 MiB), which
caps the cuBLAS seq per head count — 40-head Qwen3-14B only reached ~1824, so pp2048 dropped to the
slower FMHA. Raised cap to 384 MiB (config `attention.attn_scores_mib`, default 384; sizes to
min(need,cap) so +128 MiB max). **Qwen3-14B NVFP4 pp2048: 14402 → 17517 (+21.6%)**, vLLM gap −40%→−29%.
8B unchanged, Q8_0 gate unchanged, 35B fine, GPU suite green, coherent. cuBLAS path is the validated
default (FP32 S-matrix) → no quality change. THREE wins shipped this session: #465, #469, #474.
Remaining NVFP4 prefill gap (still −29% on 14B) = the FMHA/fused-attention kernel itself for n>cap
(FlashInfer-class, multi-day) — next frontier.

### 2026-05-29 — Iteration 8: prefill FMHA precisely diagnosed (ncu) — needs kernel rewrite
- Confirmed #474's cap=384 default is OPTIMAL: cuBLAS attention wins ≤~2560 tok (pp2048 +21%), but
  FMHA wins BEYOND (pp4096: FMHA 15027 vs forced-cuBLAS 12750 — O(n²) S-matrix materialization
  dominates at long ctx). So the threshold routing is correct; DON'T raise the cap further.
- **ncu on fmha_sm120_fp8 (14B pp4096):** Compute(SM) **14.5%**, DRAM **1.1%**, L1/TEX **75.7%**,
  block-limit=barriers(24), 1.88 ms. NOT compute- or DRAM-bound — bottlenecked on shared-memory
  traffic + __syncthreads. ~85% of tensor-core throughput unused → large headroom, but realizing it
  is a FlashAttention-2-class kernel rewrite (fewer barriers, better smem tiling + cp.async pipeline,
  higher TC occupancy). This is THE long-context NVFP4-prefill lever vs vLLM/FlashInfer, and the
  GGUF-prefill story shares the same attention path. Multi-day, prior FMHA attempts refuted — needs
  a dedicated focused effort, not a tail-of-session attempt.

### SESSION CLOSE STATUS (2026-05-29)
THREE wins shipped+merged: #465 (lm_head NVFP4 decode +8-16%), #469 (NVFP4 MoE decode +52-84%),
#474 (S-matrix cap → NVFP4 prefill +21% on 40-head). imp NVFP4 DECODE = best-in-class vs llama.cpp
AND vLLM across the fleet (primary metric WON). Server robustness audited (§6 PASS). All tractable
single-session wins HARVESTED + verified. Remaining gaps precisely scoped, all multi-day kernel work:
(1) FMHA fused prefill-attention rewrite (ncu: 14.5% compute util, the NVFP4 long-ctx + GGUF prefill
lever), (2) GGUF-prefill MMQ (IMMA capped at 4.3% peak), (3) spec-decode (multi-week). Decode is
near-roofline/maxed. Next session: pick up the FMHA rewrite as a dedicated effort (sm120-cuda-expert).

### 2026-05-29 — Iteration 8b: FMHA root cause PINPOINTED (smem S materialization)
launch_bounds tweak (fp8 FMHA minBlocks 1→2) REFUTED (15051 vs 14995, noise) — not occupancy-bound.
Code read: `fmha_sm120_fp8_kernel` (attention_fmha_sm120.cu) — the header comment claims "register-based
online softmax (no shared-memory materialization of S)" but the impl ACTUALLY materializes the score
tile in shared memory (`S_tile` float[Bq×Bkv], `P_half`) and round-trips it between the QK mma.sync
(f8f6f4 m16n8k32) and the PV WMMA. THAT smem round-trip is the ncu-measured L1/TEX 75.7% bottleneck
(compute only 14.5%). EXACT FIX for the next session: rewrite the fp8 path to keep S/P in registers
(true FA2 online softmax, accumulate row max/sum in regs, no S_tile), which should lift compute util
and close the long-context NVFP4-prefill gap to vLLM/FlashInfer. Also note: header says "WGMMA" but
sm_120 has no WGMMA — it's WMMA m16n16k16 (→ HMMA), another reason for excess smem traffic; the QK
already uses mma.sync f8f6f4 but PV still uses WMMA. Multi-day, well-scoped, flag-gate it.

### 2026-05-29 — Iteration 9: "echtes FA" — register-resident FA2 SHIPPED (+20% prefill) + CUDA 13.3 switch
Picked up the iter8b lever: rewrite the smem-materializing fp8 FMHA into a true FlashAttention-2.
- **New kernel `fmha_sm120_fa2_kernel<HD>`** (`attention_fmha_sm120.cu`): 8 warps × 16 query rows
  (Bq=128), each warp runs an INDEPENDENT online softmax — **S, P, and O stay in REGISTERS**, only
  K(fp8)+V(f16) staged in smem → **1 `__syncthreads`/KV tile** (vs the fp8 kernel's smem-materialized
  S/P/O + 4 barriers). The transpose-free trick: the m16n8 QK accumulator layout is byte-identical to
  the m16n8k16 PV A-operand layout, so two adjacent 16×8 S tiles assemble directly into the PV A-frag
  after in-register softmax. PV via hand-written `mma.sync.m16n8k16` (not `nvcuda::wmma`/HMMA). 144 reg,
  0 spill, 40 KB smem (vs ~81 KB). head_dim=128 first cut; other HD fall through to fp8 (safe).
- **Flag-gated** `attention.fmha_fa2` (default "never") / env `IMP_FMHA_FA2=1`; wired into
  `attention_dispatch.cu` before the fp8 path. Default-off = zero risk until enabled.
- **Correct:** 7/7 new `FmhaFA2Test` cases vs CPU oracle <5% (causal/non-causal/GQA/multi-tile/long-ctx/
  SWA/softcap); coherent long-ctx generation (FA2 fires at seq_kv=2582/6328, no degeneration); full
  attention + GPU suites green.
- **+20% prefill:** Qwen3-14B NVFP4 pp4096, interleaved ×3 (cuBLAS-drift-controlled, 10 reps): median
  **15746 → 18915 tok/s (+20.1%)**, all trials +18–22% (>> ±2.6% noise).
- **ncu confirms the diagnosis + fix** (pp4096, identical 13.3 conditions): fp8 = SM 16.2% / L1TEX 68.0%;
  **FA2 = SM 23.3% / L1TEX 52.9%** — the smem-round-trip bottleneck (iter8b: 14.5%/75.7%) is relieved
  exactly as predicted. NOT yet compute-bound (23%): now occupancy-limited (16.6%, 1 block/SM @144 reg)
  + residual L1TEX (53%, K/V staging + scattered V loads). Headroom remains → next levers: register
  reduction/CompileIQ (→2 blocks/SM), `ldmatrix.trans` for V, cp.async K/V pipeline, mxf4nvf4 QK (2.6×).
- **CUDA 13.3 switch (whole project):** Dockerfile builder+runtime → `cuda-toolkit-13-3` (V13.3.33, PTX
  ISA 9.3; host driver UMD 13.3 + host toolkit 13.3). Full PTX survey 13.2 vs 13.3 @compute_120a =
  **0 of 247 instructions flipped** (sm_120 ISA is silicon-fixed; no tcgen05/wgmma/TMA from a toolkit
  bump; cp.async.bulk/.ignore_oob/st.async.b128 stay ❌). Baselines: `docs/ptx-status-2026-05-29-cuda13*-sm120a.md`.
  Entire GPU suite green under 13.3, 0 regressions. FA2 reg count identical under 13.3 ptxas (no free win).
- **Roadmap (13.3 tooling):** CUDA Tile **for C++** shipped (`cuda_tile.h` confirmed in-toolkit) — gated
  on the sm_120 perf question (Yadav et al.: cuTile = 0.53× FA2 on sm_120); CompileIQ auto-tuner (~15%
  on optimized kernels) queued as last-mile pass on the FA2 kernel. Both in `docs/roadmap.md`.

### 2026-06-13 — Catch-up: the 05-30 → 06-13 arc (40+ PRs, detail in CHANGELOG/PRs)

The per-iteration log above stops at 2026-05-29. The campaign continued through ~PR #700 but was
tracked in `CHANGELOG.md`, PR descriptions, and the agent's private memory rather than here. Strategic
summary so this journal isn't misleading on resume:

- **Cross-engine quality gap was tokenization, not numerics (#657).** Faithful Qwen2 / o200k / SPM /
  cl100k pre-tokenizers made 4 families byte-identical to HF; matched-band NLL gaps collapsed to ≤1.3%.
  This reframed several "quality" suspicions as canonicalization, not kernel error.
- **NVFP4-limit campaign (06-11/06-12).** prefill_chunk default 512→2048 (MoE pp2048 +127%); FA2
  PV/QK f16-acc default-on (#673/#674); FA2 became the primary hd=128 prefill, reclaiming ~380 MiB
  (#687). pp4096 was driven to the structural wall: every bounded lever (Cross-Tile, Grouped-GEMM
  tile axis, chunk-4096, occupancy/2-CTA, fp8-QK) was empirically refuted. fp8-QK is format-intrinsic
  infeasible (e4m3 3-mantissa-bit compounding); reframed vLLM's fp8 win as fp8-KV *storage*, and the
  −35% kv-fp8 MoE tax was diagnosed (deterministic-cuBLAS forcing, not the gather) and removed (#682).
- **VRAM audit.** Net multi-GiB reclamation on NVFP4 (#678/#679/#686/#687/#689) via fallback-buffer
  gating, SF dedup, and a contiguous per-(layer,proj) micro-scale slab. Last open item: ms_ref loader slab.
- **Speculative decode (n-gram prompt-lookup) shipped opt-in** (#668-#670; CLI +6.6%, server +5.4%).
  Graph-captured verify root-caused: a conditional-loop off-by-one wrote KV one slot high (#683 → #692).
  Spec stays opt-in: tg128 −15% on short / draft-poor prompts; default-on needs an engagement heuristic.
- **gpt-oss:** SafeTensors PPL is model-intrinsic (#663 closed); GGUF MXFP4→NVFP4 MoE support landed (#690).
- **Housekeeping:** dead-path removals (sm_90 WMMA, Hopper FMHA include), arch-comment fixes, baseline
  refresh + roofline pins, and `imp.conf.example`/parser reconciliation (#693-#700).

Net competitive position: NVFP4 decode + TTFT + MoE-pp2048 are won vs llama.cpp and vLLM; the lone
open gap is pp4096 NVFP4 prefill (~1.19-1.25× vs vLLM), now bounded-lever-exhausted. See the RESUME
block's OPEN WORK list for what's actually actionable next.

### 2026-06-18 — Load-time + serving-throughput campaign; decode frontier exhausted

Three wins shipped, every "frontier" lever decisively refuted by a cheap deterministic gate **before**
sinking days into it. Primary model: Qwen3-Coder-30B-A3B-FP4 (NVFP4, FP8 KV by default).

**Shipped (merged):**
- **#734 — CUTLASS NVFP4 SF-cache slab.** The prefill SF cache built ~18.6k SfAtom buffers via 18.6k
  per-tensor `cudaMalloc`+`cudaMemsetAsync`. One slab alloc + 256B-aligned borrowed sub-regions
  (`sf_borrowed`, mirrors `fp16_bulk_data`). nsys CUDA-API trace: malloc 433→16 ms, memset 174→0.08 ms,
  free 215→21 ms ⇒ **~785 ms load+teardown saved**, byte-identical SF (1800.55 MiB / 18,625 tensors).
- **#735 — batch the MoE expert convert LAUNCHES.** After #734 the loop still launched one convert
  kernel per tensor. Native NVFP4 experts are contiguous per (layer,proj), so group from the model and
  do ONE `convert_nvfp4_moe_scales_to_sfatom` (grid.y=ne) per group (the gpt-oss path already did this):
  **18,625 → 337 launches (−98%); convert GPU 24.6 → 8.1 ms.**
- **#736 — VRAM-aware auto `max_batch_size`.** The auto heuristic sized by weight footprint (>20 GB →
  batch=1), so imp-server served concurrent requests one-at-a-time despite ~10 GB free. KV is a shared
  paged pool clamped to free VRAM downstream (no OOM from a larger cap). Made it VRAM-aware (subtract the
  about-to-upload weight footprint — weights are host/mmap at resolver time so cudaMemGetInfo reads the
  near-empty card): **MoE auto 1→15, dense 4→17; ~2.36× aggregate server throughput at conc 16.**
  Serving-only (imp-cli/--bench force batch=1 → perf-baseline gate untouched).

All three degen-verified (55 server-suite checks, 0 fail; dense byte-exact stream==non-stream).

**Refuted with measurement (do NOT re-pursue — full detail in agent memory `perf_hunt_*_2026_06_18`):**
- **Decode-attention occupancy** (short ctx): split-K 11→22→43 = 0.00 e2e (serialized ncu over-states
  critical-path share; under CUDA graphs attention overlaps and is off-path).
- **Upload-memcpy batching** (4 attempts): the 38k H2D copies are per-expert from IRREGULAR host
  addresses (`break_e=2`); neither contiguous nor strided 2D copy can gather them. 867 ms H2D is fundamental.
- **Long-context decode attention:** at 16k ctx it dominates decode (~70%) but the kernel is
  compute-latency-bound on the per-token online-softmax chain — more splits raised occ 35→49% but
  kernel time +5%, DRAM flat ~10%. (e2e long-ctx decode A/B is host-variance noise — trust ncu.)
- **TC (tensor-core) decode attention:** a WMMA path already exists (`paged_attention_decode_nvfp4_tc`,
  BitDecoding, opt-in). ncu @16k: WMMA 118.4 µs vs scalar 119.0 µs — IDENTICAL. TC doesn't help; the
  matmul isn't the bottleneck (the softmax/dequant/latency chain is). This is why it's opt-in/default-off.
- **Graph-preserving spec-verify:** conditional-graph body hardcodes +1 token/iteration (no variable
  accept without a deep rewrite); the only device-side draft (MTP) is ~25-30% acceptance (dead-end).
  Path-B (graph the verify forward) refuted by nsys gate — verify is host-sync-bound (sync 21.9%, launch
  only 4.9%), so hiding launches doesn't help. NB: spec on REAL code is ~neutral (268 vs 300 tok/s), the
  earlier "−73%" was a synthetic --bench artifact.

Methodology note: deterministic ncu/nsys gates caught two near-phantom "wins" that were host-day decode
variance, and pre-empted two multi-day kernel rewrites (TC attention, spec-verify) shown futile by a
single measurement. Refutations > wins here — they save the next session from re-treading dead ends.

### 2026-07-04 — ThriftAttention outlier promotion: FP4 attention quality gate PASSES (#846 reopened)

Spike #846 (SageAttention3 recipe) was quality-refuted on 07-04 (noise compounds with context).
Web recheck found the documented reopen path had matured: ThriftAttention (arXiv 2605.23081) is
measured on GB202/sm_120 with Apache-2.0 sm120 kernels — runtime block-mean scores Q̄·K̄^T, top-k
KV-tile promotion to exact compute, online-softmax merge. Implemented flag-gated in the #868
scaffold (`attention.mxfp4_promote_budget`, default 0): pre-pass (tile means + top-k select, sink
+ diagonal force-included), `Promote` template param, promoted tiles = FP32 scores from global
FP16 (kmean-consistent under ksmooth) + FP16 WMMA PV.

**Teacher-forced NLL, Qwen3-14B-NVFP4, natural-prose corpus (Gutenberg 1342), Δ vs FA2 baseline:**
full recipe no-promote **+9.9% @1k / +4.4% @9.3k** (failure mode reproduced); promote=1.0 sanity
+1.4% / −0.4%; **budget 0.05 → −0.6% / −0.2% — GATE PASSED** (0.10: −0.6% / +0.2%; 0.25: −1.3% /
+0.07%; blockscale-only+0.10: −1.8% / +0.4%). Two findings: (a) sink+diagonal promotion ALONE
recovers most of the error at ≤1k (budgets ≤10% reduce to the 2 forced tiles there); (b) with
promotion active, ksmooth/pv_fp4 stop mattering (bs-only arm ≈ full recipe). Docs/markdown corpus
degrades far less (+1.9% @11.6k no-promote) — corpus choice is load-bearing for this failure mode.

**Caveats:** quality spike only — promoted path is scalar FP32 from global (slow by design), knob
default-off, NO perf round run (per spike order). A perf phase needs a register-resident FA2-class
port of the FP4+promotion path; decode-side expectations remain dampened (KV already NVFP4-stored,
long-ctx decode attention is latency-bound, see 2026-06-18 refutations).

### 2026-07-04 (later II) — KV-append-quant paged-FP4-QK: quality CONFIRMED, perf REFUTED (#846 stays closed)

Follow-up on the surviving thesis: read K/V for chunked-prefill continuation straight from the
NVFP4 KV cache (quant paid once at append) instead of gather→FP16. Shipped flag-gated
(`attention.mxfp4_paged_kv`, default off) with a PagedKV FMHA variant: past tiles feed the
block-scale MMA from cache bytes (zero quant work), the CURRENT chunk stays fresh FP16
(force-promoted tiles), promotion budget applies to past tiles.

**Quality (Qwen3-14B-NVFP4, prose, Δnll vs FP16-KV): GATE PASSED** — paged + budget 0.05/0.10 =
+0.34%/+0.31% @9.3k (chunk 2048), +1.5% @1k (chunk 256, beats even the FP16-gather reference's
+2.26% there); promote=1.0 sanity +0.05%. Load-bearing finding along the way: **quantizing the
recency window is the entire quality cost** — the pre-hybrid variant that stored the current
chunk FP4 before attention lost +3.7–5.4% NLL even with EXACT FP32 compute over the stored
values, while FP4-storing the whole past costs ≈0 (gather ref −0.12%). Corollary: teacher-forced
chunked-prefill PPL is structurally blind to recency-window storage noise (current chunk is
always fresh FP16) — an nvfp4-KV DECODE quality check (generation reads recents from cache)
would need its own probe.

**Perf: REFUTED, decisively.** nsys @9.3k attention path: gather+FA2 184.5 ms (gather itself only
4 ms — it was never the cost); paged hybrid 2197 ms (12×, scalar current-band); **pure paged-MMA
floor with zero quant work (probe, force-promote off): 1557 ms = 8.5× FA2**. Removing the quant
instructions did NOT rescue the smem-materializing kernel — it is latency-bound
(long-scoreboard), not quant-bound. Combined with FA2's profile (instruction-mix-bound, DRAM
9.6% — bandwidth is not its bottleneck), a register-resident FP4-K port's upside is ~10–20%
kernel on ~8% of prefill GPU time → not fundable. #846 stays closed; the paged path ships as a
quality-validated research scaffold, default off.
