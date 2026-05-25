# Maintenance Sprint Log

## 2026-05-25 — Session 1

### Baseline Benchmarks (post-fixes)

RTX 5090, CUDA 13.2, sm_120a, Docker, cold GPU (28°C), batch=1, graphs ON.

| Model | Quant | pp512 (tok/s) | tg128 (tok/s) |
|-------|-------|--------------|--------------|
| Qwen3-8B | Q8_0 | 26,540 | 274 |
| Qwen3-14B | Q6_K | 16,369 | 165 |
| Qwen3-8B cortecs | NVFP4 | 25,539 | 226 |
| Qwen3-Coder-30B-A3B | NVFP4 | 16,800 | 266 |
| Qwen3.6-35B-A3B | NVFP4 | 10,906 | 227 |
| Gemma-4-26B-A4B | Q4_K_M | 4,645 | 256 |

### Completed

1. **PR #405: cudaMallocAsync weight pool** (`fix/cudamalloc-async-weight-pool`)
   - Migrates all weight allocs from cudaMalloc/cudaFree to cudaMallocAsync/cudaFreeAsync
   - Fixes cuBLAS status-14 on WSL2/CUDA 13.2 after mass cudaFree
   - Removes gemm_reset() workaround and WDDM page-refill hack
   - 6 files, -134/+67 LOC

2. **PR #406: NVFP4 use-after-free + TensorKindTable**
   - Critical: Phase-4b freed source pointers borrowed by CUTLASS NVFP4 cache. ALL native NVFP4 SafeTensors models were broken.
   - TensorKindTable: added 4 missing GDN tensor kinds. Fixes 3 pre-existing test failures.
   - All 885 tests pass (was 3 failures).

3. **This branch: MoE prefill investigation + error fixes**
   - MoE prefill analysis: the "1258 vs 25513" gap was stale (pre-CUTLASS-3.x device-args). Current pp512=16,800 tok/s. Cross-engine bench shows ~20% gap vs vLLM on comparable models. vLLM cannot load Qwen3-Coder on sm_120. Gap is largely structural (different MoE dispatch strategy) — not sprint-scoped.
   - weight_dispatch.cu: split CUTLASS GEMM failure from workspace-too-small error
   - engine_workspace_warmup.cpp: cap banned token logging at 30 entries (Gemma-3 has 6251)

### Bug Sweep Results

- Only 1 TODO in entire src/ (gemm_grouped.cu — grouped GEMM optimization)
- 3 test failures — all fixed (TensorKindTable GDN entries)
- 1 critical regression — fixed (NVFP4 use-after-free in Phase-4b)
- 1 misleading error message — fixed (weight_dispatch.cu)
- 1 warmup hang — fixed (Gemma-3 banned token logging)
- Remaining cudaFree calls in infrastructure code are shutdown-only, safe

### MoE Prefill Analysis

- **CUTLASS 3.x device-args path fires correctly** for Qwen3-Coder NVFP4
- pp512 and pp2048 show flat throughput (~14.7k tok/s without graphs, ~16.8k with) — per-layer overhead, not compute-bound
- MoE weights are memory-bandwidth-bound at M=12/expert average
- Cross-engine bench (2026-05-24) showed Qwen3-Coder MoE gap closed to 1.056x vs vLLM
- Bigger open gap: Q4_K_M prefill -48-59% vs llama.cpp (needs MMQ kernel port, 2-3 weeks)

### Next

- Push this PR
- Refactor targets scan (large files in src/compute/, src/model/)
- Test hardening (numerical parity, KV cache edge cases)
- Doc cleanup
