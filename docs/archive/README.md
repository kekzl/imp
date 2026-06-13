# Archive — summary index

This directory used to hold ~70 historical documents (~25k lines): closed design
memos, brainstorm→spec→plan→findings cycles, dated benchmark snapshots, early
audits, and three retired attention kernels. They were **consolidated into this
single summary** on 2026-06-13 and the originals removed from the tree.

**The full text of every entry below is preserved in git history.** To read an
original file:

```bash
# last commit that still contained the archive (before consolidation):
git log --oneline --all -- docs/archive/<path>
git show <commit>:docs/archive/<path>
```

Everything here is **historical / superseded / refuted** — it is not the current
state of the engine. For current docs see [`../architecture.md`](../architecture.md),
[`../sm120.md`](../sm120.md), [`../../BENCHMARKS.md`](../../BENCHMARKS.md), and the
live audit memos in [`../audit/`](../audit/).

---

## Retired attention kernels (had RESURRECTION memos + source)

| Kernel | Verdict |
|---|---|
| `fmha_sm120_cluster/` (`attention_fmha_sm120_cluster.cu`, 1102 LOC) | TMA-bulk cluster FMHA. **Refuted**: `cp.async` beats `cp.async.bulk.tensor` (TMA) on sm_120; the cluster variant was slower. Superseded by `attention_fmha_sm120.cu` / `flash_attention_blackwell`. |
| `fmha_mxf4nvf4_sm120/` (`.cu` + `.h`) | FP4 `mxf4nvf4` block-scale FMHA prefill experiment. Not merged — raw e4m3/FP4 score quality cliff on real activations (#511 class). FP4 attention stayed opt-in elsewhere. |
| `attention_naive/` (`attention_naive.cu` + `.h`) | Naive FP32 reference attention (incl. SWA). Was a correctness oracle; superseded by `flash_attention_blackwell` as the in-tree FP16 reference. |

## `tile-fa2-dispatch-shelved.md` (was root `DISPATCH.md`)

Progress log for the autonomous cuTile-FA2 (CUDA 13.3 Tile / C++) mandate. The
decisive 2026-05-29 experiment pinned the autotuned cuTile ceiling at ~26.5
TFLOPS (3.2% of roofline) — far below the hand-written FA2 — so the Tile backend
was **shelved** (GOAL §4.4 negative branch; see [`../roadmap.md`](../roadmap.md)).
Kept as a file (not consolidated) because the live `tools/analysis/tile_fa2_probe.cu`
still points to its WSL-driver dev recipe for any future re-evaluation.

## `sm120-real-perf-plan_2026-05-04.md`

The original five-lever sm_120 performance plan (RTX 5090). All five levers
**shipped or were superseded by 2026-05-10** (NVFP4 fast path, FA2 family,
register-resident attention, etc.). Historical north-star scoping.

## `audit-2026-05-08/` — first structured audit run

`master_plan` / `roadmap_inventory` / `DONE` / `followups` plus two SafeTensors
audits (`safetensors_audit`, `safetensors_nvfp4_audit`). Outcome: the SafeTensors
+ NVFP4 load path was hardened; the roadmap items were largely shipped. Superseded
by the structural-debt audits in [`../audit/`](../audit/).

## `baselines-2026-05-04/` — pre-BMMA validation snapshot

Qwen3.6 validation report + SASS dump + summary CSV taken before the BMMA work.
Frozen measurement snapshot; superseded by `tests/perf_baseline*.json` +
`BENCHMARKS.md`.

## `bench-2026-05-10/` — MoE NVFP4 prefill optimization push

`baseline_pp`, `pp_optimization_report`, `iteration2_findings`, `profile_findings`,
`decode_regression_check`, `moe_fusion_targets`, `sm120_smallM_audit` (SF native
layout), `cudnn_nvfp4_moe_audit`, `vllm_kernel_audit`. Outcome: the NVFP4 grouped
GEMM fast path landed; **cuDNN NVFP4 MoE on sm_120 was refuted** (no usable path);
the "20× / 1258 tok/s" premise from `vllm_kernel_audit` was later refuted (real gap
~1.4×). All superseded by the CUTLASS-3.x grouped-GEMM rework and
`docs/audit/prefill_gap_2026_06_07.md`.

## `cross_engine_bench_2026_05_24.md`

imp vs llama.cpp vs vLLM snapshot. Superseded by `BENCHMARKS.md` (SHA-anchored) and
`docs/MISSION_JOURNAL.md`.

## `review-2026-05-16/` — 5-phase architecture review

`MASTER_REPORT` + phase1 inventory / phase2 perf / phase3 maint / phase4 ext /
phase5 synthesis (Cartographer / PERFHAWK / CODEREAPER / INTEGRATOR). A full
architecture review; its actionable findings fed the refactor program. Superseded
by `docs/audit/structural_debt_2026_06_08.md` / `..._2026_06_09.md`.

## `plans-2026-05/` and `plans-closed/` — closed design memos

Design-scoping memos, each closed (shipped, refuted, or shelved):

- **Shipped:** `q4k_imma_design` (INT8-IMMA prefill family, #612–#617),
  `nvfp4_gemv_tuning`, `kernel_fusion_rmsnorm_into_gemv`, `gdn_scan_splitk`,
  `moe_host_offload_graphs` / `_phase5_findings` (+10%), `fp4_pv_phase3`,
  `nvfp4_smoothquant_input_scale`.
- **Refuted / shelved:** `turboquant_fp8_gap` (TurboQuant retired, PR #251),
  `bitdecoding_phase2`, `gdn_chunkwise_scan`, `k5_h2o_eviction`, `k8_cpu_offload`,
  `moe_prefill_cudagraph_via_cutlass_moe_scheduler` (sm_120 lacks the MoE
  scheduler / TMA-WS grouped GEMM).

## `superpowers-2026-05/` — brainstorm→spec→plan→findings cycles

`plans/` + `specs/` for the May feature pushes. Each pairs a design spec with an
implementation plan and (often) a findings memo:

- **Shipped:** tools + JSON-schema coordination; CUDA 13.2 modernization; paged /
  chunked prefill correctness; chunked prefill for hybrid GDN+MoE and Gemma-4 SWA;
  BitDecoding NVFP4 paged-decode port + residual FP16 cache; NVFP4 small-M grouped
  GEMM kernel; MXFP4-KV (slice 3).
- **Refuted / parked:** TurboQuant phases 1–2 (microbench + NIAH → retired);
  Q4_K_M INT8-IMMA **phase 3 refuted on the real model** (phase-2B ceiling reached —
  the dense path loses to cuBLAS; the MoE-prefill win shipped separately); MTP
  phase 2.2/3.5/5.5 + wiring (parked — see `docs/audit` / memory: "MTP no bug,
  genuine ~25-30%, parked").

---

*Consolidated 2026-06-13 (housekeeping). Originals live in git history.*
