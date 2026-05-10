# MoE fusion targets — 2026-05-10

Profile: `bench/results/nsys_post_bc3bc31_20260510_104438.nsys-rep`
Workload: Qwen3-Coder-30B-A3B-NVFP4, pp=512, 7 total runs (warmup + measured)
Branch: `perf/moe-nvfp4-prefill-fast-path` after commit `bc3bc31`

## Baseline launch count

imp MoE prefill (CUTLASS 3.x NVFP4 path): **9 launches per layer** × 48 layers = **432 total per prefill**.

| Reference | Launches/layer | Source |
|---|---:|---|
| imp (bc3bc31) | 9 | this audit |
| vLLM 0.20.2 | 7–8 | Apsys blog + `vllm_kernel_audit.md` (some routing on separate stream) |
| SGLang | 5 | Apsys blog |

imp's 9 launches vs SGLang's 5 = **4 extra launches per layer = 192 extra total per prefill**.

## Per-launch time breakdown (Qwen3-Coder pp512, post-bc3bc31)

Instance counts verified: 336 per-layer-per-run kernels ÷ 7 total runs ÷ 48 MoE layers = 1 per layer per run.
CUTLASS grouped GEMM: 1008 ÷ 7 ÷ 48 = 3 per layer. Quantize: 672 ÷ 7 ÷ 48 = 2 per layer.

| # | Launch | Instances | Avg µs | Time/layer | Time/prefill | % of MoE non-GEMM | Source |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | `moe_fused_permute_kernel` | 336 | 5.9 | 5.9 µs | 0.28 ms | 12% | `src/compute/moe_routing.cu:533` |
| 2 | `moe_gather_kernel_impl` | 336 | 9.2 | 9.2 µs | 0.44 ms | 20% | `src/compute/moe_routing.cu:430` |
| 3 | `quantize_fp16_nvfp4_cutlass_moe_kernel` [gate+up] | 336 | 8.2 | 8.2 µs | 0.39 ms | 17% | `src/compute/gemm_cutlass_sm120.cu:313` |
| 4 | CUTLASS grouped GEMM [gate] | 336 | 95.1 | 95.1 µs | 4.57 ms | — | `src/compute/gemm_cutlass_grouped_3x.cu:268` |
| 5 | CUTLASS grouped GEMM [up] | 336 | 95.1 | 95.1 µs | 4.57 ms | — | `src/compute/gemm_cutlass_grouped_3x.cu:268` |
| 6 | `swiglu_fp16_kernel` | 336 | 4.7 | 4.7 µs | 0.22 ms | 10% | `src/compute/activation.cu:59` |
| 7 | `quantize_fp16_nvfp4_cutlass_moe_kernel` [post-SwiGLU] | 336 | 8.2 | 8.2 µs | 0.39 ms | 17% | `src/compute/gemm_cutlass_sm120.cu:313` |
| 8 | CUTLASS grouped GEMM [down] | 336 | 95.1 | 95.1 µs | 4.57 ms | — | `src/compute/gemm_cutlass_grouped_3x.cu:268` |
| 9 | `moe_scatter_fused_residual_kernel` | 336 | 10.5 | 10.5 µs | 0.50 ms | 22% | `src/compute/moe_routing.cu:937` |

**Non-GEMM MoE overhead**: 46.7 µs/layer × 48 = **2.24 ms/prefill** (10% of GPU time).
**CUTLASS GEMM GPU time**: 285.3 µs/layer × 48 = **13.70 ms/prefill** (63% of GPU time).
**Total GPU kernel time**: ~21.7 ms/prefill.
**Wall-clock time**: ~33 ms/prefill.

### Wall vs GPU time gap: 11.3 ms

The 11.3 ms gap between wall-clock (33 ms) and GPU kernel time (21.7 ms) is dominated by
CPU overhead from 48 × 3 = 144 `gemm_grouped_cutlass_3x_nvfp4` calls per prefill. Each call
executes `gemm.can_implement()` + `gemm.initialize()` + `gemm.run()` with a fresh local
`GrpGemm gemm` instance (`gemm_cutlass_grouped_3x.cu:268`). CUTLASS `initialize()` traverses
template machinery to set up the tile scheduler and validate alignment — estimated ~75 µs CPU
per call × 144 calls = **10.8 ms of CPU overhead** that serializes the GPU launch stream.
The remaining ~0.5 ms is attributed to the D2H sync at `executor_forward_moe.cu:1297`.

## D2H sync: pipeline serialization point

`executor_forward_moe.cu:1293–1297` does a `cudaMemcpyAsync` + `cudaStreamSynchronize` to
retrieve `expert_offsets` from device to host before building per-expert pointer arrays. This
fires once per MoE layer, blocking the CPU until all prior GPU work for that layer drains.
In practice the sync stalls the CPU for ~10–50 µs (the GPU is nearly always still executing
prior work), then the CPU spends ~5–10 µs building pointer arrays before submitting the next
batch of kernels. Net effect: the pipeline is back-to-back serialized, never pipelining layer N+1
setup during layer N execution.

## Fusion candidates (ranked)

### Candidate A: gate+up grouped GEMM fusion (prefill-only translation unit)

**What**: Combine the gate and up projections into a single `gemm_grouped_cutlass_3x_nvfp4` call
with `2*ne` problems (gate expert e at problem 2e, up expert e at problem 2e+1). Same input
activations are reused for both projections — no extra quantize pass needed.

**Evidence**: Tested in `iteration2_findings.md` (section 5):

| | pp512 median | tg256 |
|---|---:|---:|
| Baseline | 12,282 tok/s | 268.3 tok/s |
| Fused gate+up | 13,728 tok/s (**+12%**) | 249.4 tok/s (**-7%**) |

The decode regression was caused by instruction-cache pressure: the fused lambda was compiled
inside `run_moe_ffn` (2507-line function), inflating the compiled function size and pushing
the hot decode code paths out of i-cache. The decode code itself was unchanged (decode uses
`gemv_nvfp4_moe_gate_up_fused` at `executor_forward_moe.cu:546`, a completely separate path).

**Fix**: Extract the fused prefill gate+up path into a standalone function in a separate
translation unit (e.g., `src/graph/executor_forward_moe_cutlass3x.cu`). This prevents the
fused code from sharing i-cache with the decode GEMV path.

**Entry points**:
- Caller: `src/graph/executor_forward_moe.cu:1706–1710` (the two `grouped_gemm()` calls for gate and up)
- `grouped_gemm` lambda: `src/graph/executor_forward_moe.cu:1661–1700`
- Underlying dispatch: `src/compute/gemm_cutlass_grouped_3x.cu:127`

**Estimated delta**: +12% pp (measured), 0% decode change (if in separate TU).

**Risk**:
- Decode: NONE if compiled in separate .cu file. The -7% in iter2 was i-cache pollution, not
  logic change.
- Numerical: NONE — same GEMM math, different batching.
- CUTLASS: `gemm_grouped_cutlass_3x_nvfp4` already handles `n_experts` problems; passing `2*ne`
  is a supported use case (CUTLASS GroupProblemShape scales linearly).

**Effort**: 1–2 days. Extract lambda into new function, update caller, A/B confirm decode is clean.

---

### Candidate B: fused gather + FP16→NVFP4 quantize kernel

**What**: Merge `moe_gather_kernel_impl` (step 2) and `quantize_fp16_nvfp4_cutlass_moe_kernel`
(step 3) into a single kernel. Currently: gather writes `gathered[expanded × d]` (FP16), then
quantize reads it to produce `packed[expanded × d/2]` + `SF[]`. Fused: reads `norm_out[n × d]`
indexed by `sorted_token_ids[]`, writes directly to `packed` and `SF` — the `gathered` FP16
intermediate buffer is never written or read.

**Bandwidth savings per layer** (Qwen3-Coder, pp=512, d=2048, expanded=4096):
- `gathered` intermediate: 4096 × 2048 × 2 bytes = **16.8 MB**
- Eliminated: 1 write + 1 read = **33.6 MB** at 1792 GB/s = **18.7 µs/layer**
- Plus: eliminates 1 kernel launch overhead (~2–5 µs)
- **Total savings: ~24 µs/layer × 48 = ~1.15 ms/prefill (~3.5% of wall time)**

**Entry points**:
- `moe_gather_kernel_impl`: `src/compute/moe_routing.cu:430`
- `moe_gather` (caller): `src/compute/moe_routing.cu:726`
- `quantize_fp16_nvfp4_cutlass_moe_kernel`: `src/compute/gemm_cutlass_sm120.cu:313`
- `quantize_fp16_to_nvfp4_cutlass_moe` (caller): `src/compute/gemm_cutlass_sm120.cu:432`
- Call site: `src/graph/executor_forward_moe.cu:1706` (inside `quantize_once` lambda)

**New kernel shape**: Reads `norm_out[sorted_token_ids[row]]` instead of `gathered[row]`. All
other logic from `quantize_fp16_nvfp4_cutlass_moe_kernel` is preserved. The `moe_find_expert`
call and `sfatom_offset` computation remain unchanged. The `sorted_token_ids` array (already
on device as `routing.sorted_token_ids.data`) is passed as an extra argument.

**Estimated delta**: +3–5% pp (bandwidth + kernel overhead).

**Risk**:
- Decode: NONE. Decode uses `gemv_nvfp4_moe_gate_up_mr_kernel` and `gemv_nvfp4_moe_swiglu_mr_kernel`
  (`executor_forward_moe.cu:546`), not the prefill gather/quantize path.
- Numerical: NONE — same quantization math.
- Memory access: indirect reads (`norm_out[sorted_ids[row]]`) are less coalesced than sequential
  reads from `gathered[]`. For pp=512 with d=2048, `norm_out` = 2 MB → fits comfortably in L2
  (RTX 5090 has 96 MB L2). Cache-miss cost should be negligible.

**Effort**: 2–3 days. New 100–150 line kernel, replace the two calls in `quantize_once` lambda.

---

### Candidate C: swiglu + post-SwiGLU quantize fusion (prefill-only, revisit of prior attempt)

**What**: Fuse `swiglu_fp16_kernel` (step 6) with the second `quantize_fp16_nvfp4_cutlass_moe_kernel`
(step 7) to eliminate the `expert_swiglu` FP16 intermediate.

**Prior attempt**: A similar fusion was tried previously (`executor_forward_moe.cu:1717–1719`):
> "A fused silu(gate)*up+quantize kernel was tried but regressed short-prompt decode ~11%
> due to low SM occupancy at small expanded"

This prior attempt likely suffered from the same i-cache pollution mechanism as Candidate A
(large function = decode GEMV paths pushed out). Additionally, the comment notes "low SM
occupancy at small expanded" — for decode (expanded = top_k = 8), the fused kernel would have
too few elements to fill the SM.

**Bandwidth savings per layer** (expanded=4096, eff=768):
- `expert_swiglu` intermediate: 4096 × 768 × 2 = 6.3 MB
- Eliminated: 1 write + 1 read = 12.6 MB at 1792 GB/s = **7.0 µs/layer**
- Plus: eliminates `swiglu_fp16_kernel` = 4.7 µs/layer
- **Total savings: ~11.7 µs/layer × 48 = ~0.56 ms/prefill (~1.7% of wall time)**

**Entry points**:
- `swiglu_fp16_kernel`: `src/compute/activation.cu:59`
- `apply_expert_activation`: `src/graph/executor_forward_moe.cu:116` (calls `swiglu`)
- Second quantize call: `src/graph/executor_forward_moe.cu:1721` (inside `quantize_once`)

**Estimated delta**: +1–2% pp (smaller than Candidates A and B).

**Risk**:
- Decode: The comment attributes the prior regression to "low SM occupancy at small expanded".
  If isolated to a separate TU (like Candidate A), the i-cache issue is resolved. The occupancy
  issue for decode is real but moot since decode doesn't use this path at all.
- Complexity: Medium. Requires fusing two fundamentally different kernels (element-wise SwiGLU
  vs scatter-quantize with indirect SfAtom indexing).

**Effort**: 3–4 days (higher complexity than Candidate B due to the SfAtom write pattern).

**Recommendation**: Defer — lower ROI than A or B, higher complexity, and the prior attempt
left a clear warning comment. Do A and B first.

---

### Candidate D: eliminate D2H sync via device-side pointer array construction

**What**: The `cudaMemcpyAsync` + `cudaStreamSynchronize` at `executor_forward_moe.cu:1293–1297`
retrieves `expert_offsets` to build `M_per[]` on CPU. This is needed to construct per-expert
pointer arrays for CUTLASS. Alternative: add a lightweight device kernel (analogous to vLLM's
`computeStridesTmaWarpSpecializedKernel`) that builds `M_per[]` and the full pointer arrays
on device, then modify `gemm_grouped_cutlass_3x_nvfp4` to accept device pointer arrays instead
of host arrays.

**Estimated delta**: ~0.5 ms direct sync savings. The wall/GPU gap of 11.3 ms is mostly from
CPU overhead in `gemm.initialize()` (see Wall vs GPU gap section above), NOT from the D2H sync.
Removing the D2H sync alone saves ~0.5 ms.

**Risk**: HIGH. Requires refactoring `gemm_grouped_cutlass_3x_nvfp4` API, adding a new setup
kernel, verifying CUTLASS accepts device-side inputs for grouped problem shapes. The CUTLASS
3.x grouped API uses host arrays internally to do per-expert validation — exposing device-only
paths is non-trivial.

**Effort**: 4–7 days.

**Recommendation**: Defer until A and B are landed. The direct savings from D2H sync removal are
modest; the larger CPU overhead comes from `gemm.initialize()`, which is a separate problem.

---

## What explains the remaining gap vs vLLM (1.42×)?

vLLM achieves ~18,500 tok/s vs imp's ~13,046 tok/s on single-sequence pp=512:

1. **CPU overhead per GEMM call** (~10.8 ms/prefill, 33% of wall time): vLLM uses torch.compile
   with CUDA Graphs for the forward pass, which replaces all 144 `gemm.initialize()` calls with
   a single captured graph replay. imp does not capture CUDA Graphs for prefill (graphs are
   captured only for decode).
2. **vLLM multi-stream routing** (~0 ms, overlapped): vLLM's TRT-LLM path runs routing kernels
   on a parallel stream, using `delayStreamKernel` as a spin-wait synchronization barrier. This
   hides routing latency behind GEMM compute. imp's routing is sequential on the same stream.
3. **imp's non-GEMM overhead** (2.24 ms): slightly higher than vLLM's ~1.5 ms due to 9 launches
   vs vLLM's ~7–8 (some on a separate stream).

The GPU kernel time for MoE GEMM is similar: vLLM uses the same CUTLASS Sm120 cooperative
block-scaled kernel (confirmed in `vllm_kernel_audit.md`).

## Recommended implementation order

1. **Candidate A** (gate+up GEMM fusion, separate TU): highest measured delta (+12% pp),
   zero decode risk if properly isolated, 1–2 days effort. The iter2 measurement gives high
   confidence in the delta. The only work is moving the fused code to a separate .cu file.

2. **Candidate B** (gather+quantize fusion): +3–5% pp, zero decode risk, medium complexity.
   After A lands, this gets the next 3–5% without touching the GEMM path.

3. **Candidate C** (swiglu+quantize fusion): +1–2% pp, deferred. Lower ROI, higher complexity.
   Worth revisiting only if A+B still leave a visible gap and the profile shows swiglu time
   is measurable after prior fusions.

4. **Candidate D** (device-side pointers / D2H sync elimination): deferred. The D2H sync is
   not the main bottleneck (the CUTLASS `initialize()` CPU overhead is). Address only if CUDA
   Graph capture for prefill is implemented (which would solve the CPU overhead problem entirely).

## Out of scope

- **K=256 CUTLASS tile**: Already tested in `iteration2_findings.md` (section 2b). K=256
  (`Shape<_128, _128, _256>`) showed +2% median within noise band (7k–17k variance). CLOSED.
- **Decode-regressing fusions**: Any fusion that changes the `n==1` decode code path is
  explicitly out of scope (decode must not regress >2%).
- **CUDA Graph capture for prefill**: Would eliminate the 11.3 ms CPU overhead gap entirely
  but requires all ops in the forward pass to be graph-safe. The D2H sync at line 1297 is
  a graph capture blocker — it must be removed first.
- **Custom PTX kernel (mma.sync.kind::mxf4nvf4)**: ~2–4 week project with uncertain payoff per
  `dead_ends.md` and `iteration2_findings.md` conclusion. vLLM uses the same CUTLASS templates.
- **Third-party deps** (FlashInfer MoE, TRT-LLM kernels): not aligned with imp's no-new-deps rule.

## Re-run recipe

```bash
# Re-profile after landing changes:
docker run --rm --gpus all \
  -v /home/kekz/models:/models:ro \
  -v /home/kekz/github.com/kekzl/imp/bench/results:/out:rw \
  -v /opt/nvidia/nsight-systems/2025.6.3:/nsys:ro \
  --entrypoint /nsys/bin/nsys imp:test \
  profile --trace=cuda --force-overwrite=true \
  --output=/out/nsys_post_fusion_$(date +%Y%m%d_%H%M%S) \
  -- /usr/local/bin/imp-cli \
    --model /models/Qwen3-Coder-30B-A3B-Instruct-FP4 \
    --bench --bench-pp 512 --bench-reps 3 --max-tokens 1 \
    --temperature 0 --seed 42 --no-cuda-graphs

# Kernel summary:
/opt/nvidia/nsight-systems/2025.6.3/bin/nsys stats \
  --report cuda_gpu_kern_sum --format column <out_file>.nsys-rep
```
