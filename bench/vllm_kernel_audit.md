# vLLM kernel audit on RTX 5090, Qwen3-Coder-30B-A3B-NVFP4 — 2026-05-10

## vLLM version + backend

- vLLM version: **0.20.2**
- Image: `vllm/vllm-openai:latest` (pulled 2026-05-10, image id `70a098d90dba`)
- FlashInfer version: **0.6.8.post1**
- Quantization spec: `modelopt_fp4` (ModelOpt NVFP4 SafeTensors format)
- NVFP4 linear backend (non-MoE): `FlashInferCutlassNvFp4LinearKernel`
- MoE backend (selected by `moe_backend='auto'`): **`FLASHINFER_CUTLASS`**
  - Full priority list evaluated: `['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTEDSL_BATCHED', 'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN', 'EMULATION']`
  - `FLASHINFER_TRTLLM` was the first candidate and was skipped (unsupported tactics for gemm1/gemm2 on SM120)
  - `FLASHINFER_CUTLASS` was selected as fallback
- KV cache dtype: `fp8_e4m3`
- torch.compile: enabled (`CompilationMode.VLLM_COMPILE`, inductor backend)
- CUDA graphs: enabled (`FULL_AND_PIECEWISE`, capture sizes [1, 2])

### Autotuner output (startup log)

```
[Autotuner]: Tuning fp4_gemm:  12/12 tactics profiled
[Autotuner]: Tuning trtllm::fused_moe::gemm1: 2/2 tactics (4 unsupported skipped)
[Autotuner]: Tuning trtllm::fused_moe::gemm2: 2/2 tactics (10 unsupported skipped)
```

This confirms TRT-LLM's `fp4_gemm` autotuner fires at startup but most tactics are
**unsupported on SM120**. The FLASHINFER_TRTLLM MoE path was skipped because `gemm1`/`gemm2`
had too many skipped tactics. The FLASHINFER_CUTLASS path was chosen instead.

---

## Profile setup

- Workload: pp=512 (513 tokens after BOS), tg=4, model=Qwen3-Coder-30B-A3B-Instruct-FP4
- Reps: 2 warmup + 3 measured (prefix cache HIT on warmup rep 2, so 4 actual prefill passes in profile)
- Achieved (measured reps):
  - rep 0: 27ms wall, 513 prompt tokens, **17,918 pp tok/s**
  - rep 1: 27ms wall, 513 prompt tokens, **19,156 pp tok/s**
  - rep 2: 28ms wall, 513 prompt tokens, **18,386 pp tok/s**
  - mean: **18,473 pp tok/s** (reference number: 25,513 from prior comparison memo — see note)
- nsys profile file: `bench/results/vllm_pp512.nsys-rep` (17 MB)
- nsys SQLite: `bench/results/vllm_pp512.sqlite` (40 MB)

**Note on 25,513 vs 18,473 tok/s**: The prior vllm_comparison_2026_05_10 memo measured 25,513
tok/s using vLLM's `benchmark_throughput.py` with multiple concurrent sequences and a larger
batch. This single-sequence pp512 benchmark achieves 18–19k tok/s, which is consistent (multi-seq
batching increases prefill throughput by aggregating expert work across sequences). The important
data point here is kernel identity, not the absolute number.

---

## Top 5 GPU kernels (across full profile, all passes)

| % | Total ms | Calls | Avg µs | Symbol (demangled, abbreviated) |
|---|---:|---:|---:|---|
| 68.6% | 1337.0 | 17,868 | 75 | `at::native::vectorized_elementwise_kernel<FillFunctor<int>>` — KV/activation buffer fill (torch.compile overhead + CUDA graph warmup) |
| 9.9% | 193.2 | 188 | 1,028 | `tensorrt_llm::kernels::delayStreamKernel(long long)` — inter-stream sync barrier |
| 2.3% | 45.3 | 3,918 | 12 | `triton_` — Triton RMS-norm / misc fused ops |
| 2.3% | 44.2 | 96 | 460 | `at::native::CatArrayBatchedCopy` — KV cache tensor concat |
| 1.7% | 33.1 | 61 | 543 | `at::native::distribution_elementwise_grid_stride_kernel` — normal dist sampling (routing stochasticity) |

**Note**: The #1 kernel (FillFunctor<int>) dominates because nsys captures the torch.compile
JIT compilation phase, CUDA graph warmup, and KV cache initialization (229K token pool).
The actual inference kernels are #8 and below.

### Inference-relevant kernels only (estimated per-pass, 4 total prefill passes profiled)

| % | Per-pass ms | Calls/pass | Avg µs | Symbol |
|---|---:|---:|---:|---|
| ~0.9% | 14.3 ms MoE GEMM | ~408 | varies 13–70 | `cutlass::device_kernel<GemmUniversal<GroupProblemShape, MainloopSm120..., tile=128x128x256>>` |
| ~0.8% | 0.8 ms | 48 | 25 | `flashinfer::BatchPrefillWithPagedKVCacheKernel` — attention |
| ~0.3% | 1.5 ms | 132 | 12 | `tensorrt_llm::cutlass_kernels::expandInputRowsKernel<fp4_e2m1, fp4_e2m1, FpXBlockScaling=1>` |
| ~0.3% | 1.4 ms | 185 | 8 | `tensorrt_llm::cutlass_kernels::doActivationKernel<fp4_e2m1, bf16, GLUAdaptor<SiLu>>` |
| ~0.1% | 0.5 ms | 101 | 5 | `tensorrt_llm::cutlass_kernels::finalizeMoeRoutingKernel<bf16, bf16, ScaleMode=1>` |

---

## Identified NVFP4 MoE GEMM kernel

**Full demangled name** (first/largest variant):

```
void cutlass::device_kernel<
  cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<cute::tuple<long, long, long>>,
    cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm120ArrayTmaWarpSpecializedBlockScaled<
        (int)2, (int)3,
        cute::tuple<cute::C<1>, cute::C<1>, cute::C<1>>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<(int)3>
      >,
      cute::tuple<cute::C<128>, cute::C<128>, cute::C<256>>,   /* tile shape: 128x128x256 */
      cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue4m3_t>,   /* A: NVF4 data + UE4M3 scale */
      ...
      cute::TiledMMA<
        cute::MMA_Atom<
          cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
            cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t, (int)16>
        >, ...>
    >,
    ...
  >
>
```

**Key attributes extracted**:
| Attribute | Value |
|---|---|
| Problem shape | `GroupProblemShape` (ptr-array batched, N problems = N active experts) |
| Mainloop | `MainloopSm120ArrayTmaWarpSpecializedBlockScaled` |
| Schedule | `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<3>` (3 pipeline stages, Cooperative = 2 CTAs per SM) |
| Tile shape | `<128, 128, 256>` (M=128, N=128, K=256) |
| A dtype | `float_e2m1_t` (NVF4) + `float_ue4m3_t` (block scale) |
| MMA atom | `SM120::BLOCKSCALED::SM120_16x8x64_TN_VS` = native NVFP4 mma.sync on SM120 |
| Scale vector | 16 (per block of 16 elements = NVF4 group_size=16) |

**14 CUTLASS kernel variants** total across the profile (different tile and prologue configs for
different expert token counts M). The largest two by total time:
- **584 calls, 29 µs avg** — small-M variant (routing warmup, tiny expert batches)
- **244 calls, 70 µs avg** — medium-M variant (main prefill, ~512 tokens across 4 experts = ~128/expert)

**Origin**: FlashInfer (via `FLASHINFER_CUTLASS` MoE backend in vllm/model_executor/layers/fused_moe/oracle/nvfp4.py)

**Library**: The CUTLASS kernel template is compiled by FlashInfer's JIT (`flashinfer.jit`).
The surrounding TRT-LLM infrastructure kernels (`expandInputRowsKernel`,
`doActivationKernel`, `finalizeMoeRoutingKernel`, `computeStridesTmaWarpSpecializedKernel`,
`fusedBuildExpertMapsSortFirstTokenKernel`) come from `libtensorrt_llm.so` (bundled with
FlashInfer's wheel).

---

## TRT-LLM infrastructure kernel inventory

These kernels handle routing, data layout, and scatter/gather around the GEMM. They are from
`tensorrt_llm::kernels::cutlass_kernels::` namespace (libtensorrt_llm.so):

| Kernel | Purpose | Calls/pass | Avg µs |
|---|---|---:|---:|
| `computeStridesTmaWarpSpecializedKernel<fp4,fp4,bf16,bf16>` | Sets up TMA stride arrays per expert problem | 180 | 1.8 |
| `expandInputRowsKernel<fp4_e2m1, fp4_e2m1, FpXBlockScaling=1>` | Expands token rows and FP4 scales into per-expert layout | 132 | 12 |
| `fusedBuildExpertMapsSortFirstTokenKernel<32,8,8>` | Routing sort: builds expert→token maps | 96 | 1.6 |
| `doActivationKernel<fp4_e2m1, bf16, GLUAdaptor<SiLu>, FpXBlockScaling=1>` | SwiGLU between gate and up proj (FP4→BF16) | 185 | 7.6 |
| `finalizeMoeRoutingKernel<bf16,bf16,ScaleMode=1>` | Weighted gather: accumulate expert outputs | 101 | 5.1 |
| `blockExpertPrefixSumKernel<512>` | Routing prefix sum | 36 | 2.9 |
| `mergeExpertPrefixSumKernel` | Multi-level prefix sum merge | 52 | 1.1 |
| `delayStreamKernel(long long)` | Routing↔GEMM stream synchronization (1ms spin-wait, 1 block/thread) | 47 | 1,028 |

**`delayStreamKernel` explanation**: This is a 1ms GPU spin-wait kernel (1 block × 1 thread) that
TRT-LLM launches on the GEMM stream to wait for the routing stream to complete. It appears to
consume 193ms total and 9.9% of kernel time, but because it runs on a separate stream from the
GEMM, it overlaps with GEMM computation and does not add wall-clock latency. The kernel itself
performs no useful work.

---

## Comparison with imp

### NVFP4 MoE GEMM kernel

| Attribute | imp (after bc3bc31) | vLLM 0.20.2 |
|---|---|---|
| Mainloop | `MainloopSm120ArrayTmaWarpSpecializedBlockScaled` | `MainloopSm120ArrayTmaWarpSpecializedBlockScaled` |
| Schedule | `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<3>` | `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<3>` |
| Tile shape | `<128, 128, 128>` | **`<128, 128, 256>`** |
| MMA atom | `SM120_BLOCKSCALED::SM120_16x8x64_TN_VS` | `SM120_BLOCKSCALED::SM120_16x8x64_TN_VS` |
| A dtype | `float_e2m1_t` + `float_ue4m3_t` | `float_e2m1_t` + `float_ue4m3_t` |
| Origin | CUTLASS (direct) | CUTLASS (via FlashInfer JIT) |

**Same kernel family, same SM120 BLOCKSCALED MMA atom. The only difference is the K-tile: vLLM
uses K=256 while imp uses K=128.**

Imp's iteration 2 (`iteration2_findings.md`) already tried `<128, 128, 256>` K tile — the
benchmark showed it as one of the tested configs. However, that was tested as `N=256` (second
dimension), not `K=256` (third dimension). The correct A/B to do is `<128, 128, 256>` K tile
specifically.

### Architecture differences

imp routes MoE prefill through a single grouped GEMM call per projection (gate, up, down
separately). vLLM's TRT-LLM path uses:
1. `computeStridesTmaWarpSpecializedKernel` — sets up per-expert TMA strides
2. `expandInputRowsKernel` — scatter tokens into per-expert layout
3. Grouped GEMM for gate+up (via `gemm1`)
4. `doActivationKernel` — SwiGLU (FP4→BF16)
5. Grouped GEMM for down (via `gemm2`)
6. `finalizeMoeRoutingKernel` — weighted gather

This is a **fused MoE pipeline** with routing-stream/GEMM-stream overlap via `delayStreamKernel`.
imp's current path does equivalent work in single-stream calls without the TRT-LLM routing
infrastructure.

---

## Throughput comparison

| | pp512 tok/s | Wall time per prefill |
|---|---:|---:|
| imp (bc3bc31, main CUTLASS path) | 13,046 | ~33 ms |
| vLLM 0.20.2 (single sequence) | ~18,500 | ~27 ms |
| vLLM 0.20.2 (multi-sequence, reference memo) | 25,513 | N/A (batched) |

The single-sequence gap is **1.42×** (imp 13k vs vLLM 18.5k), not the 20× cited in the
`baseline_pp.md` file. That 20× gap was against the multi-sequence vLLM number. The real
single-sequence gap is ~1.4×.

---

## Strategic implication

**Same CUTLASS kernel family. The edge is from K-tile and TRT-LLM MoE pipeline architecture.**

Concrete findings:

1. **vLLM's NVFP4 MoE GEMM is NOT TRT-LLM `fp4_gemm`**. The TRT-LLM autotuner fires at startup
   but skips 4–10 tactics as "unsupported" on SM120. vLLM falls through to `FLASHINFER_CUTLASS`,
   which is the same CUTLASS Sm120 cooperative block-scaled kernel that imp uses.

2. **K-tile difference**: vLLM uses `<128, 128, 256>` vs imp's `<128, 128, 128>`. A K=256 tile
   doubles register-blocking depth, which can improve efficiency for the prefill-sized M
   (128 tokens/expert at pp512 with top-k=4). This is a **contained 1-line change** to test
   in imp's `gemm_cutlass_grouped_3x_nvfp4.cu`. Estimated gain: 5–20% on prefill.

3. **TRT-LLM infrastructure overhead**: `expandInputRows`, `doActivation`, `computeStrides`,
   `finalizeMoeRouting` add ~4ms per prefill pass (per-pass estimate). This is actual work
   (routing, scatter/gather, SwiGLU) that imp also does, just differently distributed.

4. **cuDNN verdict confirmed**: No cuDNN kernels appear anywhere in the profile. The
   `cudnn_nvfp4_moe_audit.md` conclusion was correct.

5. **The 25,513 multi-seq reference is not comparable** to imp's single-sequence number. vLLM
   in multi-seq mode batches more tokens per GEMM call (larger M), using all SMs more efficiently.
   imp's current 13k tok/s on a single sequence with 512 tokens is the right comparison base.

### Next action (highest-ROI, lowest risk)

**Test K=256 tile in `gemm_cutlass_grouped_3x_nvfp4.cu`:**

```cpp
// Current (K=128):
using GrpTileShape = Shape<_128, _128, _128>;
// Test (K=256, matching vLLM):
using GrpTileShape = Shape<_128, _128, _256>;
```

This is a single-line change, compile-safe (K=256 is valid for NVFP4 block_size=16 since
16 divides 256), and directly targets the measured difference. Based on the iteration-2
negative results, if K=256 regresses prefill or decode, revert and close this line of
investigation. The remaining ~1.4× gap after K-tile is explained by TRT-LLM's fused
multi-stream MoE pipeline, which would take 2–4 weeks to replicate.

---

## Profile re-run recipe

```bash
# From host, with model at /home/kekz/models/Qwen3-Coder-30B-A3B-Instruct-FP4
docker run --rm --gpus all \
  --shm-size=4g \
  -v /home/kekz/models:/models:ro \
  -v /home/kekz/github.com/kekzl/imp/bench:/bench:ro \
  -v /home/kekz/github.com/kekzl/imp/bench/results:/out:rw \
  -v /opt/nvidia/nsight-systems/2025.6.3:/nsys:ro \
  --entrypoint /nsys/bin/nsys \
  vllm/vllm-openai:latest \
  profile \
  --output=/out/vllm_pp512 \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  -- python3 /bench/vllm_bench_pp512.py

# Extract kernel summary:
/opt/nvidia/nsight-systems/2025.6.3/bin/nsys stats \
  --report cuda_gpu_kern_sum \
  --format column \
  bench/results/vllm_pp512.nsys-rep
```
