# Phase 3: Residual FP16 Cache for BitDecoding NVFP4 Decode

**Date:** 2026-05-09
**Author:** Raphael Friedmann (kekz@kekz.org), via Claude Code autonomous run
**Status:** Spec — implementation pending Phase 2 merge

## Goal

Add a small per-layer FP16 ring buffer holding the **N newest** KV-cache tokens. The TC paged decode kernel reads attention from BOTH the NVFP4 paged cache (older tokens, tokens 0..ctx_len-N-1) AND the FP16 residual (newest tokens, ctx_len-N..ctx_len-1), and combines them via the standard online-softmax merge. This is BitDecoding's third lever (after TC dispatch in Phase 1+2): the most-recently-written tokens skip the FP4-quantization round-trip on every decode step. Saves both:

1. **Per-decode quantization overhead**: writing token T's K/V no longer requires immediate FP16→NVFP4 packing + UE4M3 calibration. Just append FP16 to the residual ring; quantize+evict only when the ring is full.
2. **Per-decode dequant overhead in attention**: the residual tokens are already FP16 — no `cvt.rn.f16x2.e2m1x2` + UE4M3 scale fold for them. Direct WMMA-FP16 path.

## Non-goals

- Residual size tuning beyond a single env-var-configurable N. Auto-tuning per model = future work.
- Residual cache eviction policies beyond strict FIFO (oldest residual entry evicted to NVFP4 when ring is full).
- Full-attention prefill changes — Phase 3 only affects DECODE. Prefill continues to write directly to NVFP4 paged cache.
- New CLI flag — same `IMP_USE_BITDECODING_QK=1` env-var gates the whole Phase 1+2+3 path; residual size via second env-var `IMP_BITDECODING_RESIDUAL_TOKENS=N` (default 0 = disabled, matches Phase 2 behavior).
- Multi-batch interactions: Phase 3 supports batch_size ≥ 1, with per-sequence residual ring buffers.

## Background

Per the BitDecoding paper (HPCA 2026, [arxiv:2503.18773](https://arxiv.org/abs/2503.18773)):

> "An fp16 residual KV cache, managed by a Residual Kernel that fuses quantization and a Packing Kernel efficiently processes the low-bit packed KV cache."

The residual is a small (1-32 tokens) FP16 cache per layer per sequence. Hot tokens stay there; cold tokens get packed to low-bit (NVFP4 in our case) and moved to the bulk paged cache. Decode reads BOTH:

```
attention(Q, KV) = softmax(Q · [K_paged | K_residual]^T / sqrt(d)) · [V_paged | V_residual]
```

In imp's current state (Phase 1+2 shipped), there's no residual cache. Every K/V write quantizes immediately. Every read dequants.

## File structure (Phase 3)

This is large enough that we split into sub-phases, each shippable as its own PR:

### Phase 3a (PR #1): Residual buffer infrastructure (no kernel reads yet)

Files:
- `src/memory/kv_cache.h` (modify) — add `kv_residual_*` accessor methods.
- `src/memory/kv_cache.cu` (modify) — allocate FP16 residual buffers alongside the NVFP4 paged blocks, sized as `[max_seqs, n_layers, 2, residual_N, n_kv_heads, head_dim]` half. `2` for K and V. Allocation gated on `runtime_config.bitdecoding_residual_tokens > 0`.
- `src/memory/kv_cache_manager.h/cpp` (modify) — add a per-sequence `residual_ring_t { write_idx, fill_count }` state. Tracks the current write position in each layer's ring.
- `src/runtime/config.h/cpp` (modify) — add `runtime.bitdecoding_residual_tokens : int = 0` config option, parsed from the `IMP_BITDECODING_RESIDUAL_TOKENS` env-var.
- `tests/test_kv_residual_alloc.cu` (create) — verify allocation succeeds for various sizes; verify `0` disables the buffer entirely.

### Phase 3b (PR #2): Kernel reads from residual

Files:
- `src/compute/attention_paged.h` (modify) — `paged_attention_decode_nvfp4_tc` signature gains optional `const half* K_residual, const half* V_residual, int residual_count` arguments. Default `nullptr` / `0` for backward compatibility.
- `src/compute/attention_paged_nvfp4_tc.cu` (modify) — after the WMMA QK + WMMA V over the NVFP4 paged blocks, run a SECOND pass over the FP16 residual: WMMA QK with FP16 K (no dequant, no scales), WMMA V with FP16 V. Online-softmax merges with the running m_w/l_w/o_reg state.
- `src/graph/executor_attention.cu` (modify) — at the call site, pass residual pointers + count when env-var is set.
- `tests/test_attention_paged_nvfp4_tc_residual.cu` (create) — TC kernel runs equivalent attention over the same data with vs without residual split (e.g., last 4 tokens in residual, rest in paged). Output should match within tolerance.

### Phase 3c (PR #3): Residual writes from K/V quantize path

Files:
- `src/graph/executor_kv_write.cu` (modify) — when residual is enabled, the new K/V write path goes:
  1. Write FP16 K/V to the residual ring at `write_idx`.
  2. If ring was full before this write, the entry being overwritten is "evicted" — its FP16 value is now quantized to NVFP4 and written to the NVFP4 paged cache.
  3. Increment `write_idx` (modulo residual_N).
- This is a state-machine write path; `KVCacheManager::residual_state(seq_id)` gives the per-sequence position.

After Phase 3c, the full BitDecoding-style residual+packing flow is in place.

## Detailed architecture

### Residual buffer layout

```
residual_K: [max_seqs, n_layers, residual_N, n_kv_heads, head_dim] FP16
residual_V: same shape
```

`residual_N` is the env-var-configured ring size (typical: 4-32 tokens).

Allocation cost (rough, for a 32-layer / 8-kv-head / 128-head_dim model):
- Per token per layer per K-or-V: 8 * 128 * 2 bytes = 2 KiB
- Per layer per token (K+V): 4 KiB
- Per token across all layers: 32 * 4 = 128 KiB
- N = 8 tokens × max_seqs = 16: 16 * 8 * 128 KiB = 16 MiB total

Affordable.

### Per-sequence state (in KVCacheManager)

```cpp
struct ResidualRingState {
    int write_idx;     // next slot to write (0..residual_N-1)
    int fill_count;    // how many slots populated so far (0..residual_N)
};
std::unordered_map<int, ResidualRingState> seq_residual_state_;
```

When `seq_id`'s residual fills (`fill_count == residual_N`), the next write evicts slot `write_idx` BEFORE overwriting (i.e., quantize the to-be-overwritten K/V to NVFP4 and write to the paged cache).

### Decode kernel changes

The current Phase-2 kernel processes:
```
for blk in [first_block, num_ctx_blocks):
    process 16 tokens of NVFP4 paged data
```

New flow:
```
for blk in [first_block, num_full_paged_blocks):
    process 16 tokens of NVFP4 paged data
# (existing TC path)

if residual_count > 0:
    process residual_count FP16 tokens via second WMMA pass
    (no FP4 dequant, no UE4M3 scale fold — just direct FP16 load + WMMA)
```

Both phases contribute to the same running m_w/l_w/o_reg via the FlashAttention-style merge.

### Quantization on residual eviction

When residual ring evicts, `executor_kv_write` (or a new helper in `quant/nvfp4_quant.cu`) takes the FP16 K/V being evicted and quantizes it:

```
nvfp4_quantize_kv_token(fp16_k, fp16_v, → nvfp4_k_packed, nvfp4_v_packed,
                       → ue4m3_k_scales, ue4m3_v_scales)
```

This is already a primitive imp has — same path the PREFILL kv-write uses.

## Testing strategy

### Phase 3a tests (PR #1)

- `test_kv_residual_alloc`: construct a `KVCacheManager` with various `residual_N ∈ {0, 1, 8, 32}`, verify allocation succeeds (or skips for N=0), verify `seq_residual_state(seq_id).write_idx == 0` after `allocate_blocks`.

### Phase 3b tests (PR #2)

- `test_attention_paged_nvfp4_tc_residual`: split a 64-token KV cache as 60 paged + 4 residual. Run the TC kernel both ways:
  - All-paged: `residual_count=0`, K_residual/V_residual = nullptr.
  - Split: `residual_count=4`, K_residual/V_residual point to FP16 conversion of last 4 tokens.
  Output should match within 1% rel error.

### Phase 3c tests (PR #3)

- E2E: `make verify-fast` with `IMP_BITDECODING_RESIDUAL_TOKENS=8`. Smoke prompt produces coherent output. Decode tg/s should improve (per BitDecoding's residual claim).

## Performance expectations

Phase 3a: zero perf delta (allocation only).
Phase 3b alone: small perf gain on the residual tokens (FP16 read is faster than NVFP4 dequant). Expected: 1-3% decode improvement at residual_N=8 on Qwen3-8B.
Phase 3c alone (without 3b's read-from-residual): zero (residual gets written but never read, equivalent to dropped writes — DON'T ship 3c without 3b).
Phase 3a+3b+3c: the full BitDecoding lever 3. Expected: 5-15% decode improvement on long-context Qwen3-8B.

## Risks

- **VRAM cost**: residual buffer adds ~16 MiB per supported batch (assumes N=8 tokens, max_seqs=16). Could squeeze tight VRAM configs. Mitigation: env-var off by default.
- **Eviction correctness**: when residual fills, the to-be-overwritten K/V must be quantized to NVFP4 and written to paged BEFORE the new write happens. Race condition risk if not ordered correctly. Mitigation: atomic eviction-then-write per token, on the same CUDA stream.
- **Block_size mismatch**: imp's NVFP4 paged cache uses block_size=16 (16 tokens per block). The residual is "tokens beyond the last full block." Need to align: when ctx_len % block_size != 0, the partial last block is in the residual; once it fills to 16, evict and start a new partial.
- **Batch size > 1 with different residual fill counts per sequence**: kernel needs per-seq residual_count, not a single global value. Easy to handle with the `context_lens`-style array.

## Re-eval triggers

If after Phase 3a+3b+3c we don't see a measurable decode improvement (≥5% on long-context Qwen3-8B vs Phase 2), it likely means imp's NVFP4 dequant path is already cheap enough that residual savings don't accumulate. In that case, defer Phase 4 (software-pipelined dequant+MMA) and revisit whether the BitDecoding lever stack is producing the expected upside on imp specifically.

## Cross-references

- Memory: `bitdecoding_sass_audit_2026_05_09.md` — empirical SASS confirming Phase 1's CUDA-cores baseline.
- Memory: `kv_research_grade_eval_2026_05_09.md` — full per-item evaluation.
- Memory: `bitdecoding_phase2_v_tc_bug_2026_05_09.md` — Phase 2 bug + fix recipe (normalized rescale invariant).
- Plan: `docs/superpowers/plans/2026-05-09-bitdecoding-port.md` — the original Phase 0+1 plan; Phase 2 followed similar pattern.
- Phase 1 PR: #142 (TC QK only).
- Phase 2 PR: #143 (TC QK + TC V + block-softmax).
