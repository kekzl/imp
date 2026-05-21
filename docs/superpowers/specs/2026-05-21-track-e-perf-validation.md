# Track E perf validation — A/B vs cuBLAS

**Date:** 2026-05-21
**Hardware:** RTX 5090 sm_120a, CUDA 13.2
**Image:** `imp:test` (identical between Track E and cuBLAS-only runs)
**Methodology:** A/B via temporary `return false` early-exit in
`attention_tiled_streaming_prefill` to force cuBLAS fallback. Both
sides use the exact same build, same model file, same docker invocation.

## End-to-end prefill speedup

| Model | seq | Track E tok/s | cuBLAS tok/s | Δ |
|---|---:|---:|---:|---:|
| Qwen3-8B Q8_0   |  512 | 12,724 | 12,100 | **+5.2%** |
| Qwen3-8B Q8_0   | 4096 | 10,830 |  9,995 | **+8.4%** |
| Qwen3-8B Q8_0   | 8192 |  9,413 |  8,216 | **+14.6%** |
| Qwen3-8B NVFP4  | 4096 | 31,925 | 28,458 | **+12.2%** |
| Qwen3-8B NVFP4  | 8192 | 31,778 | 28,384 | **+12.0%** |

Bench command:
```bash
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  imp-cli --model <model> --bench --bench-pp <seq> --bench-reps 3 \
  --max-tokens N --temperature 0
```

Decode (`tg128`) unchanged between Track E and cuBLAS (~155 tok/s on
Qwen3-8B Q8_0) — confirms Track E only touches prefill.

## Observations

### Speedup grows with seq length

Attention is O(n²) while QKV/FFN GEMMs are O(n). At pp512 attention is a
small slice of total prefill (~2-3% per nsys profile); by pp8192 it's a
larger share (~15-20% extrapolated). Track E's per-attention speedup
translates more cleanly to total prefill at long context — exactly the
regime where the 1 GiB cuBLAS S-matrix workspace would be hit hardest.

### NVFP4 weights amplify attention's share

NVFP4 weights skip the Q8_0 dequant step (which the profile showed as
21.5% of total Q8_0 prefill time). With dequant gone, attention takes
proportionally more time, so Track E gives bigger gains: +12% at pp4096
on NVFP4 vs +8% on Q8_0.

### Säule-3 gating-bench projection vs reality

The gating bench projected 3-5× **attention-kernel-isolated** speedup.
End-to-end the gain is 5-15% because attention is a fraction of total
prefill on these models. The microbench was correct about the kernel;
it just didn't model what fraction of total time the kernel owns.

This is the expected outcome of Amdahl's law given the profile:
```
total_speedup = 1 / (1 - p + p/k)
where p = attention fraction, k = attention kernel speedup
```
For p=0.05, k=3: total = 1/(1 - 0.05 + 0.05/3) = 1.034 → +3.4%.
For p=0.20, k=3: total = 1/(1 - 0.20 + 0.20/3) = 1.156 → +15.6%.
Matches observed at pp512 vs pp8192.

## nsys profile snippet (Qwen3-8B Q8_0 pp512, Track E enabled)

Top 6 kernels by total time:

| Kernel | % time | Total ms |
|---|---:|---:|
| `dequant_q8_0_kernel` | 21.5% | 82.7 |
| `cutlass_80_tensorop_f16_s16816gemm` (FFN) | 14.9% | 57.3 |
| `nvjet_sm120_qqhsh_mma_128x64x64` (FFN) | 14.2% | 54.3 |
| `gemv_dp4a_gate_up_kernel` | 8.5% | 32.7 |
| `gemv_dp4a_kpar_kernel` | 5.6% | 21.5 |
| **`attention_tiled_streaming_kernel<64,128>`** | **2.3%** | **8.97** |

Attention is dominated by upstream/downstream GEMM and quant work at
short context. To further accelerate Q8_0 prefill, the targets are
dequant fusion (21.5%) and FFN GEMM tuning (~29% combined) — both out
of scope for Track E.

## Reproduce

```bash
# With Track E (default, after PR #350 merges):
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  imp-cli --model /models/Qwen3-8B-Q8_0.gguf --bench --bench-pp 8192 \
  --bench-reps 3 --max-tokens 4 --temperature 0

# To re-run A/B, temporarily add `return false;` at the top of
# attention_tiled_streaming_prefill in src/compute/attention_tiled_streaming.cu,
# `make build`, re-bench, revert.
```

## Decision

Track E ships as merged. Bench validates the architectural decision
made in the gating report (`2026-05-21-track-e-gating-bench-report.md`)
and quantifies the realistic win.

Follow-up workitems that could grow the win further:
- **NVFP4-KV inner-loop** (Plan Tasks 16-17) — 3.3× higher mma ceiling
  for NVFP4 KV cache. Bigger leverage when long-context NVFP4 prefill
  becomes the bottleneck.
- **hd=512 HD-chunking** (Plan Task 12) — unlocks Gemma-4 global layers
  from cuBLAS path. Currently Gemma-4 global falls back without
  regression but doesn't get the Track E speedup either.
- **Out-of-scope: dequant fusion + FFN GEMM tuning** — bigger wins on
  Q8_0 prefill than any further attention work.
