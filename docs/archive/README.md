# Archive

Two different things live here, and the distinction matters.

**Files still present in this directory** are full documents whose numbers are
cited from source comments or from living docs — a summary would not serve them.
`check-release.sh` gates those citations, so none of these can simply be deleted.

**Everything under the section headings further down** is a *summary* of a
document that was consolidated away on 2026-06-13 and removed from the tree. The
full text of each is in git history; the recipe for reading one is below.

## What is in this directory

| file | subject | cited from |
|---|---|---|
| [`PREFILL_GAP_2026_06_07.md`](PREFILL_GAP_2026_06_07.md) | the prefill gap against llama.cpp | code (3) + docs |
| [`ROOFLINE_2026_06_07.md`](ROOFLINE_2026_06_07.md) | first roofline campaign | code (1) + docs |
| [`ROOFLINE_GPTOSS_2026_07_13.md`](ROOFLINE_GPTOSS_2026_07_13.md) | roofline on gpt-oss | code (1) + docs |
| [`TILE_FA2_DISPATCH_SHELVED.md`](TILE_FA2_DISPATCH_SHELVED.md) | the shelved tiled-FA2 dispatch | code (1) + docs |
| [`STRUCTURAL_DEBT_2026_07_07.md`](STRUCTURAL_DEBT_2026_07_07.md) | structure audit | docs |
| [`PERF_AUDIT_2026_07_23.md`](PERF_AUDIT_2026_07_23.md) | performance audit | `imp.conf.example` |
| [`PPL_PARITY_2026_07_12.md`](PPL_PARITY_2026_07_12.md) | cross-engine perplexity parity | docs |
| [`VRAM_AUDIT_2026_07_07.md`](VRAM_AUDIT_2026_07_07.md) | VRAM accounting snapshot | docs |
| [`CPP23_MIGRATION_2026_07_08.md`](CPP23_MIGRATION_2026_07_08.md) | the C++23 move | docs |
| [`HOUSEKEEPING_2026_06_13.md`](HOUSEKEEPING_2026_06_13.md) | the consolidation pass described below | docs |
| [`BENCHMARKS_PRE_V0.44.md`](BENCHMARKS_PRE_V0.44.md) | `docs/BENCHMARKS.md` before the 2026-09-23 prose cleanup, all superseded sweeps | `docs/BENCHMARKS.md` |
| [`MEMORY_CENSUS_2026.md`](MEMORY_CENSUS_2026.md) | `MEMORY.md` census, design draft A2-A6, migration plan, implementation log | `docs/internals/MEMORY.md` |
| [`LIMITATIONS_DETAIL_2026.md`](LIMITATIONS_DETAIL_2026.md) | MTP speculative-decoding investigation series | `docs/LIMITATIONS.md` |
| [`QUANTIZATION_AWQ_FINDINGS.md`](QUANTIZATION_AWQ_FINDINGS.md) | AWQ findings, refuted quantization experiments | `docs/QUANTIZATION.md` |
| [`KERNELS_REFUTED_2026.md`](KERNELS_REFUTED_2026.md) | refuted FA2 occupancy follow-up | `docs/internals/KERNELS.md` |
| [`ROADMAP_LEDGER_2026_09_28.md`](ROADMAP_LEDGER_2026_09_28.md) | `docs/ROADMAP.md` closed rows, Closed, 2026 bar, lever ledger, batch=1, MoE offload, quantizer, competitive records, known limitations, shelved | `docs/ROADMAP.md` |
| [`ROADMAP_LEDGER_2026_10_15.md`](ROADMAP_LEDGER_2026_10_15.md) | Now / Next / Later rows closed, refuted or superseded in place up to 2026-10-15 (80 rows) | `docs/ROADMAP.md` |
| [`PERFORMANCE_2026_05.md`](PERFORMANCE_2026_05.md) | 2026-05 prefill and KV-dtype comparison tables, moved out of `docs/PERF.md` 2026-09-05 (#1919) | `docs/PERF.md` |

Moved here from `audit/` on 2026-08-11 so that `audit/` holds only the ledgers
you are meant to consult. **`AUDIT_ARCH_2026_07_29.md` deliberately stayed in
`audit/`**: `check-release.sh` cross-checks it against `SETTLED.md` for
finding-status coherence, and the gate fails if either leaves that directory —
which is how this move discovered it had misclassified the file.

## Plans and audit records moved 2026-11-15

Closed plans from `docs/plans/` live in [`plans/`](plans/), closed dated audit records from
`docs/audit/` in [`audit/`](audit/). Content unchanged, links rewritten.
A plan whose state turns CLOSED moves here with its index row.

| plan | state | standing record |
|---|---|---|
| [2026-05-28-q4k-mmq-kernel-design](plans/2026-05-28-q4k-mmq-kernel-design.md) | CLOSED 2026-05-28, REFUTED before implementation | roadmap "Closed competitive records" |
| [2026-06-profiling-campaign](plans/2026-06-profiling-campaign.md) | record, moved out of `docs/internals/PROFILING.md` 2026-09-23 | `docs/internals/PROFILING.md` |
| [2026-07-22-token-recycling-spec-tree](plans/2026-07-22-token-recycling-spec-tree.md) | CLOSED 2026-08-19, neutral to -7%, default off | roadmap Closed, "speculation tree" |
| [2026-07-31-qwen3-vl-vision](plans/2026-07-31-qwen3-vl-vision.md) | CLOSED 2026-08-11, shipped #1163-#1180 | roadmap Closed, "vision beyond Gemma" |
| [2026-08-24-qwen38-port](plans/2026-08-24-qwen38-port.md) | CLOSED 2026-08-27, all 8 items terminal | roadmap Closed, "Qwen3.8 port roadmap" |
| [2026-08-27-prefill-decode-overlap](plans/2026-08-27-prefill-decode-overlap.md) | CLOSED 2026-08-27, NEUTRAL both shapes, default off | roadmap lever ledger |
| [2026-08-28-sparse-decode-attention](plans/2026-08-28-sparse-decode-attention.md) | CLOSED 2026-08-30, SHIPPED opt-in (#1808, #1818, #1819) | roadmap Open 3 (closed 2026-10-01, archive) |
| [2026-08-29-qwen38-long-context-posture](plans/2026-08-29-qwen38-long-context-posture.md) | CLOSED 2026-08-31, both open items answered | roadmap lever ledger |
| [2026-08-31-fp8-ssm-prefill](plans/2026-08-31-fp8-ssm-prefill.md) | CLOSED 2026-08-31, REFUTED e2e, closed unmerged | roadmap prefill kernels |
| [2026-08-31-mtp-multicandidate-hybrid](plans/2026-08-31-mtp-multicandidate-hybrid.md) | CLOSED 2026-08-31, built and measured, gate not met | roadmap Closed, "speculation tree" |
| [2026-08-31-roadmap-ledger-detail](plans/2026-08-31-roadmap-ledger-detail.md) | record, moved out of the roadmap 2026-08-31 | - |
| [2026-09-04-lever-ledger-detail](plans/2026-09-04-lever-ledger-detail.md) | record, moved out of the roadmap 2026-09-04 | - |
| [2026-09-11-batched-mtp-verify](plans/2026-09-11-batched-mtp-verify.md) | CLOSED 2026-11-14: the bound form ships default on (`speculative.mtp_batch_verify`); the full mode is deleted (32 streams 1966 -> 1159 tok/s) | CHANGELOG `speculative.mtp_batch_verify` |
| [2026-09-12-factored-verify-spare](plans/2026-09-12-factored-verify-spare.md) | CLOSED 2026-11-14: factor rows ship in the bound verify (#3088); the factored spare is deleted | CHANGELOG `speculative.mtp_batch_verify_k` |
| [2026-09-16-open-rows-closed-detail](plans/2026-09-16-open-rows-closed-detail.md) | record, moved out of the roadmap 2026-09-16 (Open rows 1, 2, 4, 7, 13, 14, 15) | - |
| [2026-09-19-qwen4exp-port](plans/2026-09-19-qwen4exp-port.md) | record, Qwen3.8-Flash-Next port shipped | CHANGELOG Qwen3.8-Flash-Next |
| [2026-10-01-recurrent-state-paging-measured](plans/2026-10-01-recurrent-state-paging-measured.md) | CLOSED 2026-10-01, DO NOT BUILD | roadmap Open 5 (archive) |
| [2026-11-01-kv-eviction](plans/2026-11-01-kv-eviction.md) | CLOSED 2026-11-01, REFUTED (continuation recall -11.2 % at keep 0.5), harness deleted 2026-11-14 | roadmap row 48 |
| [2026-11-10-swebench-dev50-imp-vs-vllm](plans/2026-11-10-swebench-dev50-imp-vs-vllm.md) | CLOSED 2026-11-10, ANSWERED (vLLM 32/50 vs imp 31-33, imp ~3x faster wall) | roadmap row 124 |
| [2026-10-04-allocator-headroom-measured](plans/2026-10-04-allocator-headroom-measured.md) | CLOSED 2026-10-04 (#2482) | roadmap row 91 (archive) |
| [2026-10-04-awq-groups-nrep4](plans/2026-10-04-awq-groups-nrep4.md) | CLOSED 2026-10-04 (#2474) | roadmap row 109 (archive) |
| [2026-10-04-awq-moe-experts](plans/2026-10-04-awq-moe-experts.md) | CLOSED 2026-10-05 (#2476) | roadmap row 99 (archive) |
| [2026-10-17-gptq-nvfp4](plans/2026-10-17-gptq-nvfp4.md) | CLOSED 2026-11-02 | roadmap row 101 |
| [2026-10-18-kv-cold-pages-decision](plans/2026-10-18-kv-cold-pages-decision.md) | CLOSED 2026-10-18, NO-GO | roadmap row 112 |
| [2026-10-18-one-measured-vram-margin](plans/2026-10-18-one-measured-vram-margin.md) | DONE 2026-10-18 | backlog B-29f |

| audit record | kind | last commit | read it when |
|---|---|---|---|
| [AUDIT_ARCH_2026.md](audit/AUDIT_ARCH_2026.md) | record, one-day read-only audit at `ef664dd8` (v0.37.0+1), no GPU job: 12 axis reports, P2 falsification (3 S0 and 9 S1 survive, 9 downgrades, 6 merges), P3 queue of 15 dispatches; campaign CLOSED 2026-09-08, P5 log 24 rows (#1909-#1927, #1939-#1946), 0 open findings | 2026-09-08 | structural history; the per-finding verdicts are `SETTLED.md` H; the P2 kill list before re-filing a downgraded finding; the "verify the brief" table before writing an audit brief |
| [TEST_INVENTORY.md](audit/TEST_INVENTORY.md) · [ESCAPE_ANALYSIS.md](audit/ESCAPE_ANALYSIS.md) | records of the test-hardening campaign, baseline `ad067d76` (v0.23.0) | 2026-08-08/09 | asking why the suite missed a class of bug; the live counts come from `python3 tools/check_test_lanes.py --report` |
| [LIBRARY_SHARE_2026_10_29.md](audit/LIBRARY_SHARE_2026_10_29.md) | record, nsys cuBLAS/cuBLASLt/CUTLASS share per phase, 13 models x 4 workloads, v0.59.2 | 2026-10-29 | picking the next call site for ROADMAP row 123 |
| [DEBT_INVENTORY_2026_10_15.md](audit/DEBT_INVENTORY_2026_10_15.md) · [PTXAS_BASELINE_2026_10_15.md](audit/PTXAS_BASELINE_2026_10_15.md) · [DISPATCH_CLEANUP.md](audit/DISPATCH_CLEANUP.md) | record, DISPATCH_cleanup 2026-10-15/17 at v0.53.0: scout inventory, per-kernel `cuobjdump -res-usage` table, append-only log | 2026-10-17 | a backlog row cites `DISPATCH_CLEANUP.md:N` |
| [DISPATCH_LAUNCH_2026_10_27.md](audit/DISPATCH_LAUNCH_2026_10_27.md) | record, launch prep log | 2026-10-27 | the launch checklist history |
| [LIBRARY_SHARE_2026_10_30.md](audit/LIBRARY_SHARE_2026_10_30.md) | record, second library-share inventory, ROADMAP row 123 | 2026-11-01 | the step between the 10-29 and 11-01 inventories |

## Summaries of documents consolidated away on 2026-06-13

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
state of the engine. For current docs see [`../architecture.md`](../internals/ARCHITECTURE.md),
[`../sm120.md`](../internals/SM120.md), [`../BENCHMARKS.md`](../BENCHMARKS.md), and the
live audit memos in [`../audit/`](../audit/).

---

## Retired attention kernels (had RESURRECTION memos + source)

| Kernel | Verdict |
|---|---|
| `fmha_sm120_cluster/` (`attention_fmha_sm120_cluster.cu`, 1102 LOC) | TMA-bulk cluster FMHA. **Refuted**: `cp.async` beats `cp.async.bulk.tensor` (TMA) on sm_120; the cluster variant was slower. Superseded by `attention_fmha_sm120.cu` / `flash_attention_blackwell`. |
| `fmha_mxf4nvf4_sm120/` (`.cu` + `.h`) | FP4 `mxf4nvf4` block-scale FMHA prefill experiment. Not merged — raw e4m3/FP4 score quality cliff on real activations (#511 class). FP4 attention stayed opt-in elsewhere. |
| `attention_naive/` (`attention_naive.cu` + `.h`) | Naive FP32 reference attention (incl. SWA). Was a correctness oracle; superseded by `flash_attention_blackwell` as the in-tree FP16 reference. |

## `TILE_FA2_DISPATCH_SHELVED.md` (was root `DISPATCH.md`)

Progress log for the autonomous cuTile-FA2 (CUDA 13.3 Tile / C++) mandate. The
decisive 2026-05-29 experiment pinned the autotuned cuTile ceiling at ~26.5
TFLOPS (3.2% of roofline) — far below the hand-written FA2 — so the Tile backend
was **shelved** (GOAL §4.4 negative branch; see [`../ROADMAP.md`](../ROADMAP.md)).
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
`docs/archive/PREFILL_GAP_2026_06_07.md`.

## `cross_engine_bench_2026_05_24.md`

imp vs llama.cpp vs vLLM snapshot. Superseded by `BENCHMARKS.md` (SHA-anchored) and
`docs/MISSION_JOURNAL.md`.

## `review-2026-05-16/` — 5-phase architecture review

`MASTER_REPORT` + phase1 inventory / phase2 perf / phase3 maint / phase4 ext /
phase5 synthesis (Cartographer / PERFHAWK / CODEREAPER / INTEGRATOR). A full
architecture review; its actionable findings fed the refactor program. Superseded
by [`docs/audit/structural_debt_2026_06_08.md`](https://github.com/kekzl/imp/blob/57d1f8ff6d3faf3d4a604a6e8a6a6e25b52b3031/docs/audit/structural_debt_2026_06_08.md) / `..._2026_06_09.md`.

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
  phase 2.2/3.5/5.5 + wiring (parked). **Update (#804):** the "MTP no bug,
  genuine ~25-30% acceptance" verdict was superseded — there *was* a bug (the
  attn-output gate used silu instead of sigmoid); fixing it lifts acceptance to
  85%+ on Qwen3.6. Spec-decode *generation* via the head stays parked for a
  different reason (GDN-hybrid recurrent state through verify + net-negative
  economics — needs a non-recurrent MTP model).

## 2026-06-20 consolidation — superseded audits, closed specs, diagnostic memos

A second sweep of dated one-off material whose conclusions are now embedded in code,
the live `audit/` memos, or shipped PRs. Originals removed; full text in git history.

**Structural-audit chain (superseded by `archive/HOUSEKEEPING_2026_06_13.md` + the 2026-06-17
root audit ledger, itself since removed — git history; *not* today's root `AUDIT.md`):**
- `audit/structural_debt_2026_06_08.md` — D1–D4 debt audit (ModelProfile / GraphExecutor /
  god-functions / flag sprawl). D1 shipped #622–#623; rest re-verified later.
- `audit/structural_debt_2026_06_09.md` — C1–C8 follow-up; C1/C2/C4 refuted, C5–C8 tracked.
  Final verdicts live in `HOUSEKEEPING_2026_06_13.md`.
- `audit/structural_consistency_2026_06_06.md` — doc↔code/build/CI consistency pass;
  M1–M3 resolved (CUTLASS 4.5.1 bump, `imp.conf.example` gaps, stale `performance.md`).
- `audit/vram_cache_structure_2026_06_07.md` — VRAM-cache rebuild audit; shipped as the
  StoragePlanner / RAII-caches rework (PR #621).
- `audit/known_issue_context_carryover_2026_05_31.md` — sporadic cross-request context
  carryover investigation; Phi-4 path closed (#503), remainder tracked via memory/issues.

**Diagnostic / gap memos (findings shipped or rolled into the roadmap):**
- `decode-gap-analysis-2026-05-29.md` — decode-gap decomposition; levers shipped #479
  (lm_head decode cache, opt-in) / #480 (perplexity harness).
- `plans/nvfp4_pp2048_analysis_2026_05_25.md` — NVFP4 dense pp2048 lever analysis (research-only).
- `plans/q4k_prefill_analysis_2026_05_25.md` — Q4K MoE prefill bandwidth root-cause memo (research-only).

**Closed / refuted design specs:**
- `superpowers/specs/2026-05-22-q4k-imma-phase3-eval.md` — Q4K INT8-IMMA phase-3 e2e eval;
  **refuted** on the real model, dense IMMA stays default-off.
- `superpowers/specs/2026-05-22-track-e-closed.md` — Track E closure (six-variant FA2);
  **code removed** PR #358.
- `superpowers/specs/2026-05-25-legacy-switch-cleanup.md` — legacy `gemm_dispatch` switch
  cleanup; **completed** PR #404.

**Superseded toolchain snapshot:**
- `ptx-status-2026-05-29-cuda132-sm120a.md` — CUDA 13.2 PTX-acceptance survey. (Its successor
  [`docs/ptx-status-2026-05-29-cuda133-sm120a.md`](https://github.com/kekzl/imp/blob/52a895a1c63dd26cc61f29d23edac32e3b5155c6/docs/ptx-status-2026-05-29-cuda133-sm120a.md) was also archived on 2026-06-29 — see below.)

**Completed design specs + plans (`superpowers/`) — work shipped, refactor program closed (#404):**
The component rationale now lives in the headers themselves (`quant_pipeline.h`, `workspace.h`,
`engine_internal.h`, `pre_dequant_internal.h`, `memory_manager.h`, `tensor_kind_table.h`), which
point here. DeepSeek MLA shipped (#802/#803, see the 2026-06-29 entry below) and was consolidated
here; the only spec still live under `superpowers/specs/` is the Q4_K MMQ kernel design — kept not
as future work but as the **refutation record** cited by [`../ROADMAP.md`](../ROADMAP.md) (the
forge experiment built the in-SMEM Q4_K MMQ + FP16 HMMA kernel and ncu-proved it ties, not beats,
cuBLAS).
- `specs/2026-05-20-architecture-refactor-roadmap-design.md` — Phase 1–5 refactor master roadmap;
  program **closed** (#404), Phases 3/4/5-TrackC landed across exec/runtime/memory.
- `specs/2026-05-24-phase5-pr1-plan-driven-cache.md` + `specs/2026-05-24-q4k-vs-llamacpp-mmq-analysis.md`
  — plan-driven StoragePlan cache (shipped #398/#621) + the Q4_K-vs-llama.cpp MMQ gap analysis behind it.
- `specs/2026-05-28-phase2-decode-batching-design.md` — concurrent decode (batch≤4), shipped #454.
- `specs/2026-06-07-vram-cache-architecture-design.md` — StoragePlan-authoritative VRAM cache, shipped #621.
- `specs/2026-06-08-model-profile-design.md` — central `ModelProfile` arch-fact source, shipped #622.
- `specs/2026-06-08-quant-pipeline-design.md` + `plans/2026-06-08-quant-pipeline.md` — D2 QuantPipeline extraction, shipped #629.
- `specs/2026-06-08-header-decongestion-design.md` + `plans/2026-06-08-header-decongestion.md` — D2 header split, shipped #630.
- `specs/2026-06-08-workspace-component-design.md` + `plans/2026-06-08-workspace-component.md` — D2 Workspace extraction, shipped #631.

## 2026-06-29 consolidation — DeepSeek MLA (shipped)

The MLA architecture spec + implementation plan, kept live under `superpowers/` while MLA was
forward-looking, were consolidated here once the work shipped. Originals removed; full text in git
history.

- `superpowers/specs/2026-05-28-mla-deepseeek-architecture-design.md` — MLA (Multi-head Latent
  Attention) architecture design for DeepSeek-V2/V3: latent-vector KV for ~93% KV-cache reduction.
- `superpowers/plans/2026-06-28-mla-deepseek-v2.md` — two-stage implementation plan: **Stage A**
  (materialized — reconstruct full K/V from the latent at projection time, reusing every existing
  attention/paged-KV/RoPE kernel) and **Stage B / Phase 3** (absorbed — store only the 512-dim
  latent + 64-dim decoupled-RoPE key, paged-latent attention kernel).

**Outcome:** both stages shipped — Stage A materialized (#802), Phase 3 absorbed latent-KV-cache
decode (#803, opt-in). First MLA arch in imp; validated on DeepSeek-V2-Lite (same-corpus PPL within ~3% of HF bf16 after the 2026-07-07 YaRN rope-mscale fix; the original "6.06 vs 5.07" was cross-corpus).
Current state lives in [`../supported-models.md`](../MODELS.md), [`../ROADMAP.md`](../ROADMAP.md),
and the `src/compute/mla_*.{cu,h}` headers.

## 2026-06-29 consolidation — frozen/superseded snapshots

Three dated one-off snapshots whose conclusions are now embedded in code, newer audits, or shipped
PRs. Originals removed; full text in git history.

- `audit/performance_agent_readiness_2026_05_31.md` — perf-agent-readiness audit. Its own Executive
  Summary declared the central premise dead ("'1258 tok/s, 20× behind vLLM' no longer holds"), its
  only lever (A1) is annotated DONE (#525), and its "~18% materialized attention" finding was
  re-measured to 0.0% on hd=128. Superseded by `archive/PREFILL_GAP_2026_06_07.md` +
  `archive/ROOFLINE_2026_06_07.md` (both still live).
- `ptx-status-2026-05-29-cuda133-sm120a.md` — sm_120a ptxas-acceptance survey. Frozen 494-line
  machine-generated snapshot; the filename said `cuda133` but the captured toolkit header was
  `V13.2.78` (compute_120f). Re-run the survey on the real 13.3 toolchain if a fresh PTX-acceptance
  matrix is needed.
- `TEST_AUDIT.md` (was [`docs/TEST_AUDIT.md`](https://github.com/kekzl/imp/blob/52a895a1c63dd26cc61f29d23edac32e3b5155c6/docs/TEST_AUDIT.md)) — Test-Trustworthiness Phase-1 gap analysis (2026-06-04).
  Superseded by the 2026-06-06 re-audit (`tests/TEST_AUDIT.md`) and the coverage-hardening work
  (PR #717, server-test stage, CTest unit/gpu split). The newer `tests/TEST_AUDIT.md` re-audit was itself retired 2026-07-10 (see below).

---

## 2026-07-10 consolidation — root cleanup (docs moved under docs/, stale reports removed)

Root-level project docs moved into `docs/`: `GOAL.md`, `BENCHMARKS.md`, `BENCHMARKING.md`
(→ `docs/`), `AUDIT_FILESIZE.md`, `PERF_LOG.md` (→ `docs/audit/`). At the time the repo root
held only the standard set (README, CHANGELOG, CONTRIBUTING, AGENTS.md, CLAUDE.md); a root
`AUDIT.md` exists again since #1106, as the memory-subsystem findings log — a different
document from the two ledgers below that once used that name. Removed
outright (point-in-time reports, superseded; full text in git history):

- `AUDIT.md` (root) — 2026-06-17 whole-repo structural/hardening audit ledger (469 lines,
  append-only through 06-29). Its confirmed findings shipped via the June/July fix PRs;
  later passes live in `archive/STRUCTURAL_DEBT_2026_07_07.md` / `_2026_07_10.md`.
  (Not the current root `AUDIT.md`, which is the 2026-07-29 memory-subsystem audit.)
- `AUDIT/agentic_server_scout.md` — 2026-06-23 phase-0 read-only scout for the agentic
  server hardening campaign; the campaign shipped (PRs #772–#774 and follow-ups).
- `BATTERY_REPORT.md` (root) — 2026-06-30 full-architecture prompt-battery run report
  (green matrix). Point-in-time snapshot; the battery itself is re-runnable.
- `tests/TEST_AUDIT.md` — 2026-06-06 test-suite re-audit / refactor plan. Implemented:
  R1–R9 closed (#585–#589), coverage hardening (#717), server-test stage. Test-file
  docstrings citing "TEST_AUDIT.md §7/§8" refer to this document's tiering decisions.
- [`docs/plans/qwen35_27b_mxfp4_host_dequant_design_2026_05_17.md`](https://github.com/kekzl/imp/blob/8a1d0396c989d9e57db9b7dfe7b1c4d6ef33f689/docs/plans/qwen35_27b_mxfp4_host_dequant_design_2026_05_17.md) — design-only memo for
  a model that remains blocked (OOM, no GGUF); the pre-dequant machinery that shipped
  took a different shape.
- [`docs/superpowers/specs/2026-05-28-q4k-mmq-kernel-design.md`](https://github.com/kekzl/imp/blob/8a1d0396c989d9e57db9b7dfe7b1c4d6ef33f689/docs/superpowers/specs/2026-05-28-q4k-mmq-kernel-design.md) was **moved** to
  `docs/plans/` (still referenced as refutation evidence by `docs/ROADMAP.md`), and the
  empty [`docs/superpowers/`](https://github.com/kekzl/imp/tree/8a1d0396c989d9e57db9b7dfe7b1c4d6ef33f689/docs/superpowers) tree removed.
- [`tools/prompt-test.sh`](https://github.com/kekzl/imp/blob/4b70d7da7fc5a4873e64a0dd53ef998487060b14/tools/prompt-test.sh), [`tools/benchmark.sh`](https://github.com/kekzl/imp/blob/4b70d7da7fc5a4873e64a0dd53ef998487060b14/tools/benchmark.sh), [`tools/chat.sh`](https://github.com/kekzl/imp/blob/4b70d7da7fc5a4873e64a0dd53ef998487060b14/tools/chat.sh),
  [`tools/download-models.sh`](https://github.com/kekzl/imp/blob/4b70d7da7fc5a4873e64a0dd53ef998487060b14/tools/download-models.sh) — March-era one-off shell tools (curl smoke prompts,
  pre-`--bench` timing loop, interactive chat wrapper, HF download helper).
  Zero references; superseded by `imp-cli --bench`, `bench/`, `scripts/verify.sh`,
  the server batteries, and manual model staging (the `imp-pull` pipeline was
  removed in #245).

## 2026-09-13 consolidation: records only other records cited

No living doc, source file, script or CI job referenced these. Full text:
`git show 5b5b6486:docs/archive/<file>`.

- `roofline_2026_07_11.md`: roofline re-measure.
- `structural_debt_2026_07_10.md`: structure audit, second pass after `STRUCTURAL_DEBT_2026_07_07.md`.
- `DISPATCH_BASELINE_2026_07_17.md`: dispatch baseline snapshot.
- `AUDIT_REPORT.md`: soundness-hardening branch report.

---

*Consolidated 2026-06-13 (housekeeping), 2026-06-20, 2026-06-29, 2026-07-10 (root cleanup) and 2026-09-13. Originals live in git history.*
