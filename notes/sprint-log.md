# Maintenance Sprint Log

## 2026-05-25 — Session 1

### Baseline Benchmarks (main @ b21d739 + fixes below)

RTX 5090, CUDA 13.2, sm_120a, Docker, cold GPU (28°C), batch=1.

| Model | Quant | pp512 (tok/s) | tg128 (tok/s) |
|-------|-------|--------------|--------------|
| Qwen3-8B | Q8_0 | 26,540 | 274 |
| Qwen3-14B | Q6_K | 16,369 | 165 |
| Qwen3-8B cortecs | NVFP4 | 25,539 | 226 |
| Qwen3-Coder-30B-A3B | NVFP4 | 14,625 | 270 |
| Qwen3.6-35B-A3B | NVFP4 | 10,906 | 227 |
| Gemma-4-26B-A4B | Q4_K_M | 4,645 | 256 |

### Completed

1. **PR #405: cudaMallocAsync weight pool** (`fix/cudamalloc-async-weight-pool`)
   - Migrates all weight allocs from cudaMalloc/cudaFree to cudaMallocAsync/cudaFreeAsync
   - Fixes cuBLAS status-14 on WSL2/CUDA 13.2 after mass cudaFree (WDDM page release bug)
   - Removes gemm_reset() workaround and WDDM page-refill hack
   - 6 files, -134/+67 LOC

2. **fix/tensor-kind-table-gdn-entries** (this branch, to be PR'd):
   - TensorKindTable: added 4 missing GDN tensor kinds (GDN_ALPHA, GDN_BETA, GDN_ALPHA_BETA_PACKED, GDN_INPUT_PACKED). Fixes 3 pre-existing test failures.
   - **Critical: Phase-4b use-after-free for native NVFP4 SafeTensors models.** Phase-4b VRAM reclamation was freeing source weight data still borrowed by the CUTLASS NVFP4 cache. ALL native NVFP4 models (cortecs, modelopt, Qwen3.6) were broken with illegal memory access. One-line fix: skip cudaFree when pointer is in wcache_.cutlass_nvfp4.

### Bug Sweep Results

- Only 1 TODO in entire src/ tree (gemm_grouped.cu — related to MoE prefill, task #4)
- 3 test failures — all fixed (TensorKindTable GDN entries)
- 1 critical regression — fixed (NVFP4 use-after-free)
- Remaining cudaFree calls in infrastructure code are shutdown-only, safe

### Findings

- Gemma-3-12B bench mode hangs during warmup (6251 banned special tokens — printing takes too long or bench warmup loop interacts badly with the large ban list)
- weight_dispatch.cu:195 error message is misleading: says "workspace too small" even when workspace IS large enough (the real error is a prior CUDA error from the use-after-free)

### Next

- MoE prefill investigation (Qwen3-Coder pp512=14,625 vs vLLM 25,513)
- Gemma-3 bench mode warmup hang
- Refactor targets scan
