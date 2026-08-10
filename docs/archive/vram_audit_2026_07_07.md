# VRAM management audit — 2026-07-07

Scope: double allocations, missed optimization potential, over-engineering in the
VRAM layer (`src/memory/`, budget/planner, pre-dequant pipeline, workspaces).
Method per the codebase-audit skill: three candidate sweeps, then **every finding
verified against the live code** (caller counts across `src/ tests/ tools/`,
consumer tracing, gate/comment reading). Priors from the 06-07/06-12 audits
(SF dedup, ms_ref dedup, attn_scores retune, G1–G5, lazy workspaces #678,
vram-budget cap #838, planner reserve #875) were not re-litigated.

**Headline:** no large double allocations remain — the big wins shipped in June
(−1810 MiB SF dedup, −1728 MiB ms_ref, −380 MiB attn_scores, +827 MiB lazy
buffers). What this pass found and fixed is dead scaffolding, one real
allocation-gate gap, formula duplication with drift, and misleading budget
diagnostics.

## Fixed in this PR

| # | Finding | Fix |
|---|---------|-----|
| 1 | `GDNState` module dead: `init()` never called, `Engine::gdn_state_` never assigned (always null), every read null-guarded across 7 files; `InferenceState::gdn_state`/`gdn_seq_id` had one (dead) writer, zero readers; `gdn_layer_map_` built but never read. GDN models run their recurrent state through `SSMState`. | Deleted `src/memory/gdn_state.{h,cu}`, the member, both `InferenceState` fields, the layer map; guards collapsed to `ssm_state_`-only. |
| 2 | `DeviceAllocator` + `PinnedAllocator`: zero runtime callers (unit tests only); the async default pool is tuned directly (`engine_weight_upload.cpp`), prefill pinned staging uses raw `cudaHostAlloc`. | Deleted both modules + their tests + the façade accessors. |
| 3 | `MemoryManager` façade reduced to "holds a VRAMAllocator": lazy accessors dead (see 2), `plan_storage_for()` zero callers, `compute_budget()` one caller. | Deleted the façade; `Engine` holds `VRAMAllocator vram_alloc_` directly, the one caller uses `compute_vram_budget()`. |
| 4 | `QuantPipeline::plan_tier_of()` — stranded migration wrapper, referenced only in comments. | Deleted; comments updated to current reality. |
| 5 | Phase-4b bisect leftover: `constexpr bool actually_free = true` with dead branches and an unreachable log string. | Constant and dead branches removed. |
| 6 | Vacuous guard: the budget-side `plan_storage()` call runs UNCONSTRAINED (`PlanHints.vram_budget_bytes` defaults to 0 → the downgrade loop and `plan.failed` can never trigger there), so `!plan.failed` on the #875 GGUF branch was always true. The "planner output diagnostic only (5.1.5)" log text has been false since #875. | Guard removed; comments/log rewritten to state the two-plan design explicitly (budget-side unconstrained plan = ideal demand for the reserve; pre-dequant budget-constrained plan = tier decisions). |
| 7 | **Real gate gap:** `moe_.raw_staging_buf` (1 expert raw, MiB-range) allocated whenever packed experts exist, but its only consumers are the `!packed.on_device` branches of the legacy MoE forward — dead weight on every all-on-device load. | Allocation now gated on `any_host_packed_experts` (exact consumer predicate, deliberately NOT the LRU predicate with its gpt-oss exemption — gpt-oss' transiently host-resident experts keep the staging fallback). |
| 8 | `nvfp4_beneficial` duplicated again (canonical in `pre_dequant_internal.h` + a re-derived lambda in `vram_budget.cpp`) — the G1 regression class; in sync today, drift waiting to happen. | Hoisted to `core/qtype.h` as the single source of truth; `pre_dequant_internal.h` re-exports via `using`, the budget calls it directly. |
| 9 | Reserve floor `max(total/10, 256 MiB)` re-derived at 6 sites, one drifted to a 1 MiB floor (`pre_dequant_phase3_nvfp4_decode.cu`). | `vram_reserve_floor()` helper in `vram_query.h`; all 6 sites converted, drift normalized to 256 MiB. NOT converted (different semantics on purpose): the `clamp(kv_reserve+256MiB, 256MiB, 1GiB)` feature-aware reserves in phase3-moe/nvfp4 and the plain additive 256 MiB safeties (workspace fp32 buffer, expert upload reserve). |
| 10 | Phantom FP8 diagnostics: strategy logged as `FP8_PREFILL_NVFP4_DECODE` with a nonzero `fp8=` MiB figure although `use_fp8_prefill` auto-resolves to false on sm_120. Zero bytes wasted (sole consumer Phase 2 is gated on `use_fp8`; the value is computed post-KV-clamp from the remainder), but the log lied. | `fp8_cache_bytes` computed only when `use_fp8_prefill`; strategy logged as `NVFP4_DECODE (fp8 prefill off)` otherwise. KV sizing math deliberately untouched (strategy still selects the same kv_fraction/target_blocks path — behavior-neutral change). |
| 11 | MemAccount attribution bit-rotted: header promised per-pool notes for the big consumers, only KV + WEIGHTS were wired → "UNTRACKED" dominated every audit run. | Added build-total notes: `WEIGHT_CACHE_FP16/_FP8/_NVFP4/_CUTLASS_SF` (after `QuantPipeline::build`) and `EXEC_WORKSPACES`; header claim now matches the wired set. |

## Verified as deliberate — do NOT re-flag

- **Fused KV / gate-up FP16 duplication** of the individual cache entries is
  load-bearing: prefill n>1 reads the fused slabs (`executor_attention_qkv.cu`,
  `executor_ffn.cu`), decode n==1 reads the individual GEMV entries. G4-analog
  prefill/decode split.
- **VRAMAllocator headroom escape hatch** (`vram_allocator.cu` allocate):
  deliberate + documented (Nemotron-30B at 29+ GiB needs it).
- **MemAccount checkpoint cost when disabled** (~7 `cudaMemGetInfo` on the init
  path): documented in the header, init-only, negligible.
- **vram_query budget view**: already centralized — one function, ~20 call
  sites. The sweep's "19 decentralized sites" suspicion was wrong.
- **Two `plan_storage()` invocations per load are intentional** (unconstrained
  ideal-demand plan for the #875 reserve vs budget-constrained tier plan in
  `QuantPipeline::build`) — now documented at both sites. Do not merge them:
  planner-driven budget sizing OOMs (G1/PR #621 finding).
- **`moe_.dequant_buf`** (1-expert slot, ~few MiB): kept unconditional — its
  consumer `can_decode_fast` (`executor_forward_moe.cu`) is live for GGUF MoE
  decode; not trivially provable dead for ST-NVFP4 and too small to chase.

## Refuted this round (do not re-chase)

- **"plan.failed skips the #875 reserve → KV-eats-everything returns"**: refuted.
  `plan.failed` was unreachable at the budget site (no budget hint → no
  downgrade loop). The Ornith "plan failed" WARN comes from the second,
  budget-constrained plan in `QuantPipeline::build` and is advisory there.
- **"~9 reserve-formula sites"**: only 6 are the same formula; the rest are
  semantically distinct reserves (see fix 9).
- **Dangling NVFP4 dequant workspace pointer on teardown**: the registered
  workspace is cleared in `free_buffers`; no dangling read.

## Measurement-gated follow-ups (need GPU evidence, reopen bar stated)

- `nvfp4_dequant_ws_buf_` (up to 512 MiB, `executor_workspace_buffers.cu`):
  backs the `gemm_nvfp4` dequant fallback and is load-bearing for capture
  safety (`nvfp4_gemm.cu` refuses cudaMalloc under capture). Post-#863 the
  small-M batched GEMV may have starved the fallback — instrument fallback
  hits across a full prefill+decode run before shrinking.
- Phase-3 MoE `d_ms_copy` non-contiguous-scales fallback
  (`pre_dequant_phase3_moe.cu`): likely unreachable after the #689 loader slab.
  Log `scales_contig` across all NVFP4-MoE models; delete only if universally true.
- `chunk_capture_k_/v_` vs `chunk_eager_k_/v_`: mutually exclusive modes, both
  can be resident. Log concurrent non-null before unifying (graph pointer
  stability likely blocks it).
- Three KV per-block cost derivations (budget exact / upload-reserve rough /
  init logging) — consolidation possible but low value; the rough pre-upload
  estimate is deliberately rough (runs before the budget).
