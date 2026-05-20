# Resurrection: FMHA sm_120 cluster prefill

**Archived 2026-05-20** (Phase 2 of architecture refactor roadmap).

## What this was

A two-block-cluster variant of the FMHA sm_120 prefill kernel using
distributed shared memory across a 2-CTA cluster to absorb half the
QK^T tile traffic. Lived at `src/compute/attention_fmha_sm120_cluster.cu`
(1102 LOC). Opt-in via `attention.no_fmha_cluster=false` (default true).

## Why it was archived

Two A/B refutes:

1. **`fmha_tma_lever_refuted_2026_05_14.md`** — TMA bulk-store on sm_120
   underperforms cp.async by 0.31×-0.79×. The cluster kernel relied on
   the TMA-style distributed-shared-memory pattern to be competitive.

2. **`m5_slice2_cluster_refuted_2026_05_17.md`** — 4-model A/B sweep on
   Qwen3.6-35B, Gemma-4-26B, Qwen3-Coder-30B, Qwen3-30B-Modelopt:
   perf signal was noise-dominated (±20% same shape, opposite signs
   between runs). Cluster output bit-identical to legacy. Default
   flipped to `attention.no_fmha_cluster=true`; code retained
   as opt-in. This PR retires the opt-in.

The Phase 2 architecture refactor removed the opt-in since (a) it was
default-off, (b) bit-identity meant it added no functional capability,
and (c) the cluster test `ClusterPathNonAligned` was failing on main
without anyone noticing — confirming the code path was unexercised.

## How to resurrect

If a future sm_120 toolchain or a new GPU SKU makes cluster execution
worth re-evaluating:

1. `git mv docs/archive/fmha_sm120_cluster/attention_fmha_sm120_cluster.cu src/compute/`
2. Restore the conditional in `CMakeLists.txt`:
   `list(APPEND IMP_COMPUTE_SOURCES src/compute/attention_fmha_sm120_cluster.cu)`
3. Restore the `try_fmha_sm120_cluster_prefill` forward decl in
   `src/compute/attention_fmha_sm120.h` and the call site in `attention_fmha_sm120.cu`.
4. Restore the `attention.no_fmha_cluster` field in `runtime/config.h`.
5. Restore the `ClusterEnableGuard` + `ClusterPath*` tests in
   `tests/test_attention_fmha_sm120.cu`.
6. Re-run the 4-model A/B from the memo and document the win condition.

## Original source

Frozen as of commit (this PR's HEAD). Use `git log --follow` for
pre-archive history.
