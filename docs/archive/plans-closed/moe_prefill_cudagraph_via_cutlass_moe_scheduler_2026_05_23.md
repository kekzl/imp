# MoE Prefill via CUTLASS `MoEProblemShape` — CUDA Graph Capture Plan
*2026-05-23 · multi-week design doc*

**Status (2026-05-23): DEFERRED — original motivation mostly closed via other levers.** The release-blocker metric (Qwen3-Coder-30B-A3B-NVFP4 MoE prefill gap to vLLM 0.20.2 single-seq) narrowed from 1.14–1.32× to **1.056×** via #374 + sustained-pp ramp work — see `memory/qwen3_coder_moe_prefill_gap_closed_2026_05_23.md`. The remaining ~5–6 % gap is what this plan would target; the lever is still valid but the urgency dropped. Also still blocked upstream on CUTLASS sm_120 `MoEProblemShape` scheduler (`IsMoEScheduler = false` at `cutlass/gemm/kernel/sm100_gemm_mixed_tma_cpasync_warpspecialized.hpp:92`). Re-open when (a) CUTLASS 4.5+ ships the sm_120 MoE scheduler OR (b) a workload re-opens a meaningful (> 8 %) MoE prefill gap.

## Mission

Close ~10–15 % of the residual MoE-prefill gap to vLLM single-seq on Qwen3-Coder-30B-A3B-NVFP4 by capturing the prefill MoE pipeline into a CUDA Graph. The bottleneck is host overhead on the per-call grouped GEMM dispatch (M-per-expert is computed on device but consumed on host today). Per the 2026-05-10 landscape memo this is the only practical lever left that doesn't require a B200 or a multi-month custom kernel.

Re-trigger condition: **CUTLASS sm120 `MoEProblemShape` scheduler reaches a usable state** (currently `IsMoEScheduler = false` stub at `cutlass/gemm/kernel/sm100_gemm_mixed_tma_cpasync_warpspecialized.hpp:92`).

## Why this is the right lever

Audited 2026-05-10:

- Hand-rolled NVFP4 grouped kernel: **refuted** (-50 % at small `M_e`, par at large)
- cuDNN MoE NVFP4 SM120: **broken** (FlashInfer #2577)
- K-tile parameter changes: **within noise** (auto-scheduler already balances SMEM)
- Cluster shape / TMA descriptor fusion: **rejected** (sm_120 has no fused-descriptor primitive)
- Gather + quant fusion (Candidate B, 2026-05-10): **refuted 2026-05-23** (gathered_base L2-hits → no measurable win; see `moe_prefill_gather_quant_fusion_refuted_2026_05_23.md`)
- CUDA Graph capture for prefill: **blocked** on `MoEProblemShape` scheduler upstream — **the subject of this plan**

Measured baseline (Qwen3-Coder-30B-A3B-NVFP4, ctx=512, 100-rep stable median 2026-05-22):
```
pp512 = 14 005 tok/s     tg128 = 259 tok/s
```
Per-prefill host overhead measured via `nsys` 2026-05-10: **~1.1 ms / pp** (was estimated 8.7 ms in a pre-implementation audit; the GrpGemm `static thread_local` cache in `769effe` collapsed most of it). Remaining ~10 % of wall is the dispatch + GEMM-launch work that's currently per-call.

## The `MoEProblemShape` API (CUTLASS upstream)

Already defined at `cutlass/gemm/group_array_problem_shape.hpp:84`:

```cpp
template <class ProblemShape_>
struct MoEProblemShape {
  using UnderlyingProblemShape = ProblemShape_;
  int32_t max_m = 0;
  int32_t max_n = 0;
  int32_t max_k = 0;
  int32_t num_groups = 0;
  int32_t* tokens_per_expert = nullptr;       // device-resident — varies per call
  int32_t* tokens_per_expert_host = nullptr;  // optional host fallback
  // ... accessors that read tokens_per_expert at runtime
};
```

The host-immutable shape (`max_m, max_n, max_k, num_groups`) is baked at graph capture. The variable per-call data (`tokens_per_expert`) is a **device pointer that the kernel reads** at runtime — the M-per-expert can change between graph replays without re-capture or `cudaGraphExecUpdate`. **This is exactly the API the graph-capture lever needs.**

What's missing: the sm120 kernel template has `IsMoEScheduler = false` and silently falls back to `GroupScheduler` (which needs the host-side `GroupProblemShape`). The work below is enabling the *real* MoE scheduler path on sm_120.

## Three landing options (decision pending Phase 0 spike)

### Option A — Wait for CUTLASS upstream
- **Effort**: 0
- **Lead time**: unknown — no public CUTLASS 4.5 release timeline ships sm120 `MoEProblemShape` enablement
- **Risk**: zero
- **Decision**: poll quarterly; not a session task

### Option B — Vendor local patch on top of CUTLASS v4.4.2
- **Effort**: 2–3 weeks (CUTLASS internals work)
- **Maintenance**: re-apply on every CUTLASS bump until upstream lands
- **Files to patch** (CUTLASS internals — sketched, not authoritative):
  - `cutlass/gemm/kernel/sm100_gemm_mixed_tma_cpasync_warpspecialized.hpp` — flip `IsMoEScheduler` from constexpr stub to a dispatch policy template parameter; route `TileSchedulerTag` to a new `MoEScheduler` when set.
  - `cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_pingpong.hpp` and the four other sm100/sm120 NVFP4 kernel headers that include the same scheduler pattern.
  - `cutlass/gemm/kernel/sm90_tile_scheduler.hpp` neighbouring file — add `MoEScheduler` (or write a new header) that consumes `MoEProblemShape::tokens_per_expert` at runtime via a CUDA `__ldg` instead of the GroupScheduler's host indirection.
  - `cutlass/gemm/group_array_problem_shape.hpp` — no change needed (`MoEProblemShape` already exists at line 84).
  - `include/cutlass/gemm/dispatch_policy.hpp` and the corresponding sm120 schedule policies — ensure the new scheduler is selectable from `KernelScheduleAuto`.
- **Risk**: high — touches sm120 instruction-selection code paths that imp's other CUTLASS uses (cluster cooperative kernel, FMHA) also depend on.

### Option C — Hybrid: ship a thin imp wrapper that *looks like* `MoEProblemShape` but compiles via the existing GroupScheduler today, and swap the underlying scheduler when CUTLASS upstream lands
- **Effort**: 3–5 days
- **Lead time**: ships today; perf delta is **zero** until CUTLASS lands the real scheduler
- **What this earns now**: imp's call sites are already written against the right API. When CUTLASS upstream lands, one-line config change unlocks the +10–15 % win.
- **What it doesn't earn**: any pp512 number today.

**Recommended**: Phase 0 spike (1 day) to validate whether Option B's patch surface is tractable. If it is, pursue B. If it isn't, ship C as a one-line-swap-ready abstraction.

## Implementation phases (Option B — full patch)

### Phase 0 — spike (1 day)
- [ ] Read the existing `GroupScheduler` implementation end-to-end. Trace one work-tile assignment from `WorkTileInfo` back to the per-group M-stride lookup.
- [ ] Identify the exact line where the scheduler reads "M of this group" — confirm it's host-side today.
- [ ] Prototype a stand-alone `MoEScheduler` that reads `tokens_per_expert[group_idx]` via `__ldg` from device. No CUTLASS integration yet — just confirm the scheduler logic can produce identical work-tile assignments given the device pointer.
- **Exit**: thumbs-up or thumbs-down on Option B feasibility.

### Phase 1 — CUTLASS patch (~1 week)
- [ ] Drop the `IsMoEScheduler = false` stub. Make it a real template parameter.
- [ ] Add `MoEScheduler` header next to `GroupScheduler`.
- [ ] Wire `TileSchedulerSelector` to dispatch on `MoEScheduler` tag.
- [ ] Make the kernel template's `to_underlying_arguments` consume `MoEProblemShape::tokens_per_expert` and pass it through to the scheduler.
- [ ] Compile-test against imp's existing NVFP4 grouped path; expect identical results when `IsMoEScheduler = false` (default) — the new code paths are additive.
- **Exit**: imp's existing pp512 / decode benches all within noise.

### Phase 2 — imp dispatch swap (~3 days)
- [ ] In `src/compute/gemm_cutlass_grouped_3x.cu`, instantiate the kernel against `MoEProblemShape<Shape<int,int,int>>` instead of `GroupProblemShape<Shape<int,int,int>>`.
- [ ] In `src/exec/executor_forward_moe.cu` `try_run_moe_cutlass3x_nvfp4_prefill_`: skip the D2H `cudaMemcpyAsync` of `expert_offsets` and the `cudaStreamSynchronize` (currently at line 1549 / 1569). The MoE scheduler reads `tokens_per_expert` at kernel runtime.
- [ ] Replace `compute_M_per_from_offsets_device` + the host iteration with a single device pointer (`routing.expert_offsets.data` minus the leading zero — i.e. derive `tokens_per_expert[i] = offsets[i+1] - offsets[i]` device-side, possibly as a fused launch).
- **Exit**: pp512 within noise (graph-capture lever is in Phase 3).

### Phase 3 — graph capture (~2 days)
- [ ] Wire MoE prefill into the existing `CudaGraphCapture` infrastructure (`src/runtime/cuda_graph.cu`). The decode path is already captured; prefill capture infrastructure exists but is gated by `runtime.prefill_graph`.
- [ ] Adjust the capture predicate to allow MoE prefill once dispatch is M-per-expert-on-device.
- [ ] Bench pp512 with `[runtime] prefill_graph = true` vs baseline.
- **Exit**: ≥ +10 % pp512 vs baseline OR clean negative result with a memo.

## Bench plan

For each phase the gate is the existing 100-rep stable median methodology on `Qwen3-Coder-30B-A3B-Instruct-FP4`:
```bash
docker run --rm --gpus all -v /home/kekz/models:/models \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  imp:test imp-cli --model /models/Qwen3-Coder-30B-A3B-Instruct-FP4 \
    --bench --bench-pp 512 --bench-reps 100 \
    --max-tokens 128 --temperature 0
```

Acceptable bands per phase:
| Phase | pp512 | tg128 |
|--|--|--|
| 1 (no behaviour change) | within ±2 % of 14 005 | within ±1 % of 259 |
| 2 (D2H sync removed) | within ±3 % | within ±2 % |
| 3 (graphs on) | **≥ 15 400 tok/s (+10 %)** OR document negative | within ±2 % |

Plus end-to-end coherence on a long-prompt MoE workload (the existing `make verify` flow when models are present).

## Risks

- **CUTLASS internals are dense**: 60 K LOC of CuTe + scheduler templates. A bad patch can cascade. Mitigation: Phase 0 spike is a real gate.
- **CUTLASS upgrade conflicts**: a vendored patch will need re-application on every bump. Mitigation: prefer landing the patch upstream once Phase 1 works locally.
- **MoE scheduler may need cluster-level changes for sm_120 specifically** (different from sm_100, where the upstream API was designed). Mitigation: Phase 0 includes a "diff sm_100 vs sm_120 scheduler" subtask.
- **Graph-capture surface area expands**: prefill graphs need re-capture when `expanded` changes, which it always does in real workloads. Mitigation: cache N different graphs keyed on `expanded` rounded to a power of 2; `cudaGraphExecUpdate` is not safe here (M-dependent strides per the 2026-05-10 audit).
- **Real-world wins may be < +10 %** if the cuBLAS variance band dominates. The 2026-05-10 audit measured per-call host overhead at ~1.1 ms / 35 ms wall = **~3 %**, not the +10–15 % the memo headline suggests. Mitigation: include a Phase 3 abort criterion if the measured win is < +3 %.

## Don't repeat

- ❌ Hand-rolling a per-expert NVFP4 grouped kernel (refuted 2026-05-10, smallM branch)
- ❌ Adding cuDNN as a dep (broken on SM120, FlashInfer #2577)
- ❌ Believing TRT-LLM has SM120 magic (it falls through to FlashInfer→CUTLASS)
- ❌ Manual K-tile parameter sweeps (auto-scheduler handles it)
- ❌ Gather + quant fusion alone (Candidate B; refuted 2026-05-23 because L2 absorbs the redundant read)

## Estimate

- **Option B end-to-end**: 1 (Phase 0) + 5–7 (Phase 1) + 3 (Phase 2) + 2 (Phase 3) days = **~2 weeks of focused work**
- **Option C end-to-end**: **3–5 days** for the wrapper + abstraction; perf delta deferred

## Re-evaluation triggers

Re-open this plan when one of these fires:
- CUTLASS 4.5+ ships sm120 `MoEProblemShape` scheduler (Option A unlocks; check `git log build/_deps/cutlass-src/ -- "**/sm120*"`)
- Hardware migrates to B200 / sm_100a (tcgen05 / TMEM / wgmma open new fronts; this plan becomes obsolete)
- imp's MoE prefill pp512 falls below 12 k tok/s (likely an unrelated regression — investigate first)
- vLLM closes its gap to imp (e.g. via FlashInfer 0.7+ — `MoEProblemShape` may land there first)

---

*This plan is a planning artefact, not a commitment to ship in any specific session. It captures the technical state so the next person to push the MoE prefill lever doesn't re-discover the 2026-05-10 landscape from scratch.*
