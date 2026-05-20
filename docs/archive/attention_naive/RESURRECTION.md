# Resurrection: naive attention reference

**Archived 2026-05-20** (Phase 2 of architecture refactor roadmap).

## What this was

A pure-FP16 reference attention prefill (no FMHA, no cuBLAS, no flash):
straightforward QK^T + softmax + PV with optional sliding window. Lived
at `src/compute/attention_naive.{h,cu}` (152 LOC).

Two callers existed before archival:
1. **Runtime SWA fallback** in `executor_attention.cu`, gated by
   `attention.no_naive_swa=false`. Used to be the only safe path for
   Gemma-4 SWA layers when cuBLAS S-matrix overflowed.
2. **Parity test** in `tests/test_attention_chunked.cu` — ground-truth
   reference to validate cuBLAS-SWA output.

## Why it was archived

1. **Runtime path:** Replaced by chunked prefill (PR documented in
   `gemma4_chunked_prefill_2026_05_15.md`). The Gemma-4 SWA layers
   now use cuBLAS sliding-window mask via the chunked path, with no
   S-matrix overflow.
2. **Test parity:** The reference function was inlined into
   `test_attention_chunked.cu` as a local static
   `naive_attention_prefill_ref`, preserving cuBLAS-SWA parity
   coverage without a public symbol.

## How to resurrect (runtime fallback)

If a future model needs a non-tiled SWA fallback again:

1. `git mv docs/archive/attention_naive/attention_naive.{cu,h} src/compute/`
2. Restore `src/compute/attention_naive.cu` in `CMakeLists.txt`.
3. Restore the gate + call in `executor_attention.cu` (was at
   `:817-846` at archive time; check pre-archive history with
   `git log --follow` on this file).
4. Restore `attention.naive` and `attention.no_naive_swa` in
   `runtime/config.h`.

## Original source

Frozen at this PR's HEAD.
