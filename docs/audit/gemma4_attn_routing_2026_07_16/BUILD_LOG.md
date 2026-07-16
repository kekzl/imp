FA2 attention coverage dispatch — build log (append-only, per commit / per role).

================================================================================
[scout] 2026-07-16 — coverage matrix + gap list → AUDIT.md
  Finding: the dispatch's ~18% legacy-cuBLAS overhead premise is stale. FA2
  already serves hd=128 (#493/#525) and hd=256 (#932) at every length; legacy
  cuBLAS prefill is 0.0% of window on hd=128 and only a narrow deliberately-kept
  tail otherwise (accuracy reference below threshold, gpt-oss sinks, non-causal
  vision encoder, debug flag, parity tests).
  Genuine remaining gap: Gemma-4 heterogeneous per-layer head_dim (256 SWA / 512
  global). The hd=256 SWA layers (5/6 in the 5:1 pattern) are FA2-servable but
  forced to cuBLAS by a coarse MODEL-level `force_cublas_attn` gate; the hd=512
  global layers had no fused kernel.
  Correction to the "hd=512 infeasible" claim: infeasible only for the
  REGISTER-RESIDENT flash kernels (O in registers). The tiled WMMA FMHA
  (fmha_sm120_prefill) keeps O_acc/Q/KV in SMEM → hd=512 fits at Bq=16/Bkv=16
  (~65 KB < 99 KB opt-in). So the gap is a routing refinement + one kernel
  instantiation, NOT a 2-CTA cooperative kernel.

[builder] 2026-07-16 — kernel: hd=512 in fmha_sm120_prefill
  attention_fmha_sm120.cu: compile-time Bkv=(HD>=512)?16:SM120_Bkv; Bq=16 launcher
  tier; `case 512` instantiation (Bq=16). ~65 KB SMEM at Bq=16/Bkv=16/HD=512 fits
  the 99 KB opt-in; occupancy 1 (same as hd=256). FP8 kernel + all HD<=256 paths
  untouched (Bkv unchanged there). Comments touching head-dim sets updated.

[builder] 2026-07-16 — routing: per-layer fused-serves for heterogeneous models
  executor_attention_prefill.cu (chunked + non-chunked): drop the MODEL-level
  force_cublas_attn; FA2 chosen per-layer (hd 128/256 incl. Gemma-4 SWA), the
  WMMA FMHA dispatch serves hd=512 (and any FA2-decline), and the cuBLAS branch
  is gated `!hetero_shapes` → unreachable for Gemma-4. Uniform + gpt-oss routing
  unchanged. chunk_fmha_ok is now per-layer fmha_hd_ok (safer abort guard).

[builder] 2026-07-16 — coverage instrumentation
  attention_cublas.{h,cu}: relaxed atomic counter + count()/reset() accessors,
  bumped at attention_cublas_prefill entry (diagnostic only, never gates).

[validator] 2026-07-16 — parity GREEN (GPU)
  test_attention_fmha_hd512.cu (6 configs): FMHA hd=512 vs fp64 max 1.2-1.87e-2 /
  mean ~3e-4 (f16 class), at-or-below cuBLAS error on every config. All PASS.
  Buffer note: cuBLAS needs FP32-S (>=3x score buffer, #677) to be accurate at
  hd=512 — initial 2x buffer exposed the FP16-S precision cliff (test-harness bug,
  fixed to 4x). Kernel was correct throughout.
[validator] 2026-07-16 — regression GREEN (GPU): 121 tests PASS incl.
  AttentionCrossPathTest.HeadDim256_Gemma3Hotspot + FmhaSm120Test.HeadDim256.
  hd=128/256 paths byte-unchanged.
[validator] 2026-07-16 — Gemma-4 coverage guard GREEN (GPU, real model
  gemma-4-26B-A4B-it-UD-Q4_K_M): Gemma4ModelTest.PrefillNeverUsesLegacyCublas
  passes — attention_cublas_prefill_call_count()==0 after a full Gemma-4 prefill
  (executed-kernel check, not just the dispatch branch). Coherence: Answers
  "Paris", NoRepetitionDegeneration + RawCompletion all PASS (hot-path coherence).
  Note: the 8 MiB cuBLAS S-matrix is still ALLOCATED for Gemma-4 (unused now) —
  harmless, a VRAM-only follow-up (skip via fa2_serves_all_prefill).
[profiler] 2026-07-16 — kernel A/B: FMHA hd=512 is 2.8-4.6x SLOWER than cuBLAS
  (pp512 0.52x, pp2048 0.22x; PERF_LOG entry 1). Root cause: 99 KB SMEM cap →
  Bq=16 (no TMEM on sm_120 to offload the 512-wide accumulator). Bkv=32 recovered
  ~40% but broke correctness + still 2.8x slower (entry 2). CUTLASS can't beat it
  (same wall; imp has zero CUTLASS attention). => fusing hd=512 for coverage is a
  REGRESSION; measurement overrides the "make legacy unreachable" plan.

[builder] 2026-07-16 — routing REVISED to hybrid (perf-driven)
  executor_attention_prefill.cu: hd=256 SWA layers -> FA2 per-layer (the win);
  hd=512 -> cuBLAS while S-matrix fits (faster), fused FMHA hd=512 only as the
  O(n) overflow fallback. Achieved by: FA2 try no longer force_cublas-gated;
  prefer_fmha excludes hd=512; cuBLAS branch no longer !hetero-gated. Kernel
  Bkv reverted 16 (32 broke rectangular parity). Coverage test rewritten:
  Gemma4ModelTest.PrefillFusesSwaLayers_Hd512StaysCublas (cuBLAS <15 calls =
  SWA majority on FA2, ~6 hd=512 globals on cuBLAS).

[validator] 2026-07-16 — FINAL validation GREEN (build #6, GPU):
  - hd=512 parity FmhaHd512Test: 6/6 PASS (Bkv=16).
  - Gemma-4 hybrid gate PrefillFusesSwaLayers_Hd512StaysCublas: PASS (cuBLAS
    calls in [1,15) = SWA majority on FA2, ~6 hd=512 globals on cuBLAS) +
    AnswersCapitalOfFrance/RawCompletion/NoRepetitionDegeneration PASS.
  - chunked + crosspath regression: 33/33 PASS.
  - uniform-model E2E: Qwen3-4B dense (PrimaryModelTest) PASS; Qwen3.5-4B GDN
    hybrid (mxfp4) 2/2 PASS — the per_layer_shapes+uniform boundary is correct.
    (GDN Q8_0 variant failed only on a MISSING model file, not code.)
  - Filesize gate: violations=0.
  NET DELIVERABLE: Gemma-4 hd=256 SWA layers -> FA2 (perf win); hd=512 -> cuBLAS
  (faster) + validated O(n) fused fallback for long-context overflow. No
  regression. hd=512 fused kernel is a correct capacity fallback, NOT a speed
  play (measurement-driven — see PERF_LOG / AUDIT entry 3).

[builder] 2026-07-16 — fallback tuning: Bkv 16 -> 32 for hd=512
  Investigation of "why is fused hd=512 slow" (user-requested). Bkv=32 engages 2
  QK warps + halves KV iterations: +40% at pp2048 (0.22x -> 0.36x vs cuBLAS),
  near-parity at short ctx. NOT a correctness break (earlier note wrong) — within
  f16 class (2.24e-2 vs fp64); only tripped an over-strict "<= cuBLAS" test gate,
  relaxed to the fp64-absolute f16 bound. The long-context gap is fundamental
  (Bq=16 O(n) KV re-read; no TMEM on sm_120; CUTLASS same wall — confirmed).
  Shipped Bkv=32 because the fallback runs at long context where cuBLAS can't.

[builder] 2026-07-16 — S-overflow routing: sliced cuBLAS replaces the FMHA fallback
  Long-ctx fallback-regime bench (NEW DISABLED_BenchLongCtxFallback, Sq=2048 vs
  Skv 8k/16k): cuBLAS in q-row slices >=64 is 3.4-3.9x FASTER than the whole-chunk
  fused hd=512 kernel; slice 16 ~ parity. Fused kernel is KV-bandwidth-bound
  (Bq=16 -> ~Sq/16 full K/V re-reads; 34.5 ms at 16k = the DRAM re-read estimate).
  Planned QK warp-split dropped unbuilt — compute is not the binding constraint
  in the kernel's only production regime.
  Shipped: attention_cublas_prefill_sliced (attention_cublas.{h,cu}; slices sized
  to the FP32-S 3x rule, floor 16 rows, returns false when workspace too small);
  routed for hd=512 at S-overflow in BOTH prefill branches (chunked + single);
  max_safe_prefill_chunk no longer clamps hetero fused-servable models — Gemma-4
  keeps full 2048-row chunks at any ctx (was ~190-row chunks at 64k, multiplying
  MoE-dequant/launch overhead model-wide). Fused hd=512 kernel = terminal
  fallback only. Parity: SlicedCublasParity (forced 32-row slices) max 1.84e-2
  vs fp64 (= whole-call error); FmhaHd512Test 7/7 PASS.

[debugger] 2026-07-16 — "decode internal error" at 16k root-caused: silent KV-pool
  reject-newest. Gemma-4-12B-NVFP4: KV fell back to FP16 (gemma4 not in the FP8-KV
  verified-safe list) -> VRAM budget clamped the pool to exactly 1024 blocks x 16 =
  16384 tokens despite --max-seq-len 17408. Prefill of 16384 fills the pool; the
  first decode block append fails -> scheduler CANCELLED the request with NO log
  (engine_scheduler reject-newest) -> imp_decode_step mapped it to a bare
  IMP_ERROR_INTERNAL. Bisection: decode OK through ctx 16324, fails at 16384;
  --kv-fp8 (pool 16656+ tokens) decodes fine at 16k (~94-99 tok/s) = diagnosis
  confirmed. NOT graph-related (--no-cuda-graphs identical). Fix (3 files):
  - engine_scheduler.cpp: loud IMP_LOG_ERROR at the KV-exhaustion cancel (block
    numbers + remedies) and at the swa_prepare cancel; admission-time WARN when a
    prompt leaves <1 KV block of decode headroom in the pool (uses
    kv_cache_raw_->total_blocks() — KVCacheStats.total_blocks counts ALLOCATED
    blocks, not capacity, and req->max_tokens at admission is a 4096 placeholder).
  - imp_api.cpp: imp_decode_step returns IMP_ERROR_CANCELLED ("cancelled") for
    engine-cancelled requests (both pre-cancelled and cancelled mid-step);
    FINISHED keeps INTERNAL (imp_generate end-of-stream contract). Zero-token
    invariant break now logs status + output count.
  - imp.h: documented the CANCELLED/INTERNAL decode_step contract.
  Validation: repro now WARNs at submit + ERRORs at cancel + CLI prints
  "cancelled"; FP8 arm no warning + decodes; short-prompt decode unchanged
  ("Paris"); test-e2e ApiGenerateParity + PrimaryModel + Gemma4Model 11/11 PASS.
  - full test-attention suite with the sliced routing: exit 0 (1 pre-existing
    FmhaFP8Test.HD64 skip). File-size gate: violations=0.
  - Gemma-4-26B e2e (UD-Q4_K_M): 4/4 PASS incl. PrefillFusesSwaLayers coverage
    gate; long-prompt (~6k tok, slicing active from chunk 3) greedy completion
    coherent ("main topic ... legacy of the Roman Empire").
  - E2E prefill A/B (imp-cli --bench, 3 reps, idle GPU): Gemma-4-12B-NVFP4
    pp16384 845.7 -> 802.9 ms (+5.3%); 26B pp8192 439.0 -> 430.5 ms (+2%, clamp
    barely binds at 8k). The kernel-level win is 3.4-3.9x on the hd=512
    attention call itself (PERF_LOG entry 4); e2e effect grows with ctx (clamp
    at 64k was ~190-row chunks) but >=16k on 26B / >=32k on 12B currently OOMs
    on this 32 GB card — pre-existing on the unpatched build too (full fp16 KV;
    SWA-KV opt-in #1022 would relieve it), NOT a regression of this change.
  - Pre-existing (both builds, orthogonal): Gemma-4-12B-NVFP4 decode step
    fails with "internal error" (prefill fine) — worth a follow-up issue.
