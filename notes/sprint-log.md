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

### Shipped (5 PRs)

| PR | Description |
|----|-------------|
| #405 | cudaMallocAsync weight pool — fixes cuBLAS status-14 |
| #406 | NVFP4 use-after-free + TensorKindTable GDN entries |
| #408 | Error messages, banned-token logging, test hardening, doc updates |
| #409 | --no-fp8-prefill flag fix + Q4_K_M prefill analysis doc |
| #410 | ban_logits_kernel: grid-stride loop + device-side ban for large token lists |

### Bugs Fixed (7)

1. **cuBLAS status-14** (PR #405): mass cudaFree corrupted cuBLAS on WSL2/CUDA 13.2. Migrated to cudaMallocAsync.
2. **NVFP4 use-after-free** (PR #406): Phase-4b freed source pointers borrowed by CUTLASS cache. ALL native NVFP4 SafeTensors models were broken.
3. **TensorKindTable** (PR #406): 4 missing GDN tensor kinds caused 3 test failures.
4. **Error message** (PR #408): weight_dispatch said "workspace too small" when GEMM failed for other reasons.
5. **Gemma-3 startup delay** (PR #408): 6251 banned tokens built ~100KB log string.
6. **--no-fp8-prefill broken** (PR #409): env var IMP_NO_FP8_PREFILL was set but never read.
7. **ban_logits_kernel 256-cap** (PR #410): single-block launch silently skipped tokens beyond 256. Host path did 6251 individual cudaMemcpyAsync.

### Analysis & Docs

- **MoE prefill**: "1258 vs 25513" gap was stale. Current pp512=16,800 tok/s. Gap ~20% vs vLLM on comparable models.
- **Q4_K_M prefill**: root-caused 8.3x bandwidth overhead (dequant round-trip). Design doc at `docs/plans/q4k_prefill_analysis_2026_05_25.md`. Custom MMQ kernel needed (2-3 weeks).
- **Gemma-3 Q4_K_M**: produces incoherent output even pre-Phase-5. Known quant quality issue, not a regression. Phase 5 dispatch turned incoherence into crashes (NVFP4 GEMV path hits corrupted logits).

### Test Hardening

- 2 new invariant tests: `EveryKindHasName`, `GDNProjectionsAreFP16Only`
- All 887 tests pass (was 885, 3 were failing)
- CI, .clang-format already existed

### Open Gaps

1. **Q4_K_M prefill -48-59% vs llama.cpp**: needs MMQ kernel (2-3 weeks)
2. **NVFP4 dense pp2048 -33% vs vLLM**: CUTLASS tail utilisation
3. **Gemma-3 Q4_K_M crash**: Phase 5 NVFP4 dispatch hits corrupted logits from quant quality issue. Workaround: use Q8_0.
4. **executor_forward_moe.cu 2844 LOC**: split candidate, deferred
