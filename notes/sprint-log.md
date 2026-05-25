# Maintenance Sprint Log

## 2026-05-25

### Baseline Benchmarks

RTX 5090, CUDA 13.2, sm_120a, Docker, cold GPU (28°C), batch=1, graphs ON.

| Model | Quant | pp512 (tok/s) | tg128 (tok/s) |
|-------|-------|--------------|--------------|
| Qwen3-8B | Q8_0 | 26,540 | 274 |
| Qwen3-14B | Q6_K | 16,369 | 165 |
| Qwen3-8B cortecs | NVFP4 | 25,539 | 226 |
| Qwen3-Coder-30B-A3B | NVFP4 | 16,800 | 266 |
| Qwen3.6-35B-A3B | NVFP4 | 10,906 | 227 |
| Gemma-4-26B-A4B | Q4_K_M | 4,645 | 256 |

No decode regressions vs prior baselines.

### Shipped

| PR | Description |
|----|-------------|
| #405 | cudaMallocAsync weight pool — fixes cuBLAS status-14 |
| #406 | NVFP4 use-after-free + TensorKindTable GDN entries |
| #408 | Error messages, banned-token logging, test hardening, doc updates |

### Bugs Fixed

1. **cuBLAS status-14** (PR #405): mass cudaFree corrupted cuBLAS on WSL2/CUDA 13.2. Migrated to cudaMallocAsync.
2. **NVFP4 use-after-free** (PR #406): Phase-4b freed source pointers borrowed by CUTLASS cache. ALL native NVFP4 SafeTensors models were broken.
3. **TensorKindTable** (PR #406): 4 missing GDN tensor kinds caused 3 test failures.
4. **Error message** (PR #408): weight_dispatch said "workspace too small" when GEMM failed for other reasons.
5. **Gemma-3 startup** (PR #408): 6251 banned tokens built ~100KB log string, causing multi-second delays.

### MoE Prefill Analysis

The "1258 vs 25513 tok/s" gap was stale (pre-CUTLASS-3.x device-args). Current pp512 = 16,800 tok/s on Qwen3-Coder-30B-A3B NVFP4. Cross-engine bench shows ~20% gap vs vLLM on comparable models. vLLM cannot load Qwen3-Coder on sm_120.

### Test Hardening

- Added `EveryKindHasName` test (catches missing tensor_kind_name entries)
- Added `GDNProjectionsAreFP16Only` test (regression gate for PR #406 fix)
- All 887 tests pass (was 885 before session, 3 were failing)

### Open Gaps (prioritized)

1. **Q4_K_M prefill -48-59% vs llama.cpp**: needs MMQ kernel port (2-3 weeks)
2. **NVFP4 dense pp2048 -33% vs vLLM**: CUTLASS tail utilisation
3. **executor_forward_moe.cu 2844 LOC**: split candidate, deferred (decode risk)
