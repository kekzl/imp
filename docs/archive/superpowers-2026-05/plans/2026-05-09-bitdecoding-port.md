# BitDecoding NVFP4 Paged Decode Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port BitDecoding's Tensor-Core dispatch for NVFP4 KV decode (HPCA 2026, [arxiv:2503.18773](https://arxiv.org/abs/2503.18773)) into imp's `paged_attention_decode_nvfp4_kernel`. Replace the 346 scalar FFMA/FADD/FMUL ops per token (currently 0 HMMA — verified via `tools/analysis/sass_nvfp4_paged_decode.sh`) with `mma.sync.aligned.m16n8k16` Tensor-Core MMA on the dequantized half-precision KV. Target: close the -3.3% decode gap vs FP16 KV at 3712-token context (measured 2026-05-09) and unlock the 8.6× kernel-level upside the paper claims at long context.

**Architecture:** Multi-stage port. **This plan covers Phases 0–1 only** (the first two shippable PRs). Phase 0 is a standalone microbench in `tools/analysis/` that validates the TC dispatch approach on synthetic NVFP4 input without touching imp's runtime. Phase 1 forks the existing `attention_paged_nvfp4.cu` kernel into a parallel TC variant, swaps the Q.K dot loop from scalar FFMA to `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`, and wires it via `IMP_USE_BITDECODING_QK=1` opt-in env var. Existing kernel stays default; numerical-equivalence test gates correctness; SASS audit gates the Tensor-Core dispatch claim. Phases 2–5 (V accumulation TC, residual FP16 cache, production opt-in flag, A/B + flip default) are deferred to a follow-up plan after Phase 1 outcome — those phases' design depends on what Phase 1 SASS + perf numbers tell us.

**Tech Stack:** CUDA 13.2.78 (host + container), nvcc `arch=compute_120a`, sm_120a (RTX 5090 GB202), `mma.sync` PTX (Tensor Core MMA), CUTLASS 4.4.x (header-only, already vendored), GTest harness.

**Spec sources** (read first):
- `kv_research_grade_eval_2026_05_09.md` — full evaluation across 4 KV research items; BitDecoding identified as highest-ROI.
- `bitdecoding_sass_audit_2026_05_09.md` — empirical SASS audit (0 HMMA, 346 scalar) + 3712-tok A/B (NVFP4 +31% prefill, -3.3% decode vs FP16) + concrete next-step sequence.
- BitDecoding paper: [arxiv:2503.18773](https://arxiv.org/abs/2503.18773) (HPCA 2026)
- BitDecoding ref impl: [OpenBitSys/BitDecoding](https://github.com/OpenBitSys/BitDecoding) — particularly `csrc/bit_decode/src/flash_fwd_kernel.h` (the packing kernel pattern)

---

## File structure (Phase 0 + Phase 1)

```
tools/analysis/
  bench_nvfp4_qk_tc_vs_scalar.cu     CREATE (Phase 0) — single-file standalone bench. No imp deps.
  bench_nvfp4_qk_tc_vs_scalar.sh     CREATE (Phase 0) — wrapper that builds + runs the bench.

src/compute/
  attention_paged_nvfp4.cu           UNCHANGED — current scalar-FFMA kernel stays default.
  attention_paged_nvfp4_tc.cu        CREATE (Phase 1) — TC-dispatch kernel. Same signature as
                                                       `paged_attention_decode_nvfp4` so the
                                                       env-var opt-in can swap entry points.
  attention_paged.h                  MODIFY (Phase 1) — declare the TC entry point.

src/runtime/
  engine.cpp                         MODIFY (Phase 1) — wire the env-var opt-in dispatch.

tests/
  test_attention_paged_nvfp4_tc.cu   CREATE (Phase 1) — numerical equivalence test
                                                        (TC kernel vs scalar kernel on synthetic input,
                                                         max_abs_err must be within FP16 ulp tolerance).
  CMakeLists.txt                     MODIFY (Phase 1) — register the new test source.

docs/
  roadmap.md                         MODIFY (Phase 1) — update BitDecoding entry with Phase 1 status.
```

Each file has one responsibility:
- `bench_nvfp4_qk_tc_vs_scalar.cu`: validate TC dispatch on synthetic NVFP4 input. **Must compile + run before Phase 1 starts.**
- `attention_paged_nvfp4_tc.cu`: TC variant of `paged_attention_decode_nvfp4_kernel`. Swaps only the Q.K dot inner loop (V accumulation stays scalar in Phase 1).
- `test_attention_paged_nvfp4_tc.cu`: GTest fixture comparing TC and scalar outputs on synthetic input, identical inputs to both kernels. Output max-abs-error tolerance: `1e-2` (FP16 1-ULP at scale ~1.0 is ~1e-3; we allow 10× headroom for accumulation order differences).

---

## Phase 0: Standalone microbench (Tasks 1–6)

**Goal**: validate that `mma.sync.aligned.m16n8k16` Tensor-Core MMA on dequantized NVFP4 input produces correct results AND is materially faster than the scalar-FFMA equivalent. Done in a single .cu file with no imp dependencies — if this fails, Phase 1 won't ship.

### Task 1: Skeleton .cu with synthetic input + scalar-FFMA reference kernel

**Files:**
- Create: `tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu`

- [ ] **Step 1: Create the file with the full Phase-0 reference scalar-FFMA Q.K dot kernel + main()**

This kernel mirrors the inner Q.K dot from `src/compute/attention_paged_nvfp4.cu:142-153` exactly. Same NVFP4 dequant via `cvt.rn.f16x2.e2m1x2`, same UE4M3 scale fold, same `__fmaf_rn` accumulation. It IS the baseline.

```cpp
// tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu
// Phase 0 microbench: compare scalar-FFMA Q.K dot vs HMMA-MMA Q.K dot on
// dequantized NVFP4 KV. No imp dependencies. Build via the wrapper script.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// NVFP4 dequant: 1 byte = 2 packed E2M1 nibbles → 2 half via PTX cvt
// (same as imp's fp4_byte_to_half2)
// ---------------------------------------------------------------------------
__device__ __forceinline__ __half2 fp4_byte_to_half2(uint8_t b) {
    uint32_t out;
    asm("cvt.rn.f16x2.e2m1x2 %0, %1;" : "=r"(out) : "h"(static_cast<uint16_t>(b)));
    return *reinterpret_cast<__half2*>(&out);
}

// UE4M3 → fp32: 4-bit unbiased exp, 3-bit mantissa-equivalent. NVFP4 group scale.
__device__ __forceinline__ float ue4m3_decode(uint8_t s) {
    int e = (s >> 3) & 0xF;     // 4-bit exponent
    int m = s & 0x7;            // 3-bit mantissa
    if (e == 0 && m == 0) return 0.0f;
    float val = (1.0f + m / 8.0f) * exp2f(static_cast<float>(e) - 7.0f);
    return val;
}

// ---------------------------------------------------------------------------
// Reference kernel: scalar FFMA Q.K dot on NVFP4 KV (mirrors imp's current path).
// Q: half [HEAD_DIM]                — single query, single head (decode shape)
// K: uint8 [seqlen_kv, HEAD_DIM/2]  — packed NVFP4 (2 elems per byte)
// K_scales: uint8 [seqlen_kv, HEAD_DIM/16]  — UE4M3 per 16-element group
// out: float [seqlen_kv]            — per-token Q.K dot (post-scale)
// ---------------------------------------------------------------------------
template <int HEAD_DIM, int WARP_SIZE = 32>
__global__ void qk_dot_scalar_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K,
    const uint8_t* __restrict__ K_scales,
    float* __restrict__ out, int seqlen_kv) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    const int tok = blockIdx.x;  // one token per block
    const int lane = threadIdx.x;
    if (tok >= seqlen_kv) return;

    // Q into registers
    float q_reg[ELEMS];
    {
        const __half2* Q2 = reinterpret_cast<const __half2*>(Q + lane * ELEMS);
        #pragma unroll
        for (int i = 0; i < ELEMS / 2; i++) {
            __half2 h2 = Q2[i];
            q_reg[2 * i]     = __half2float(h2.x);
            q_reg[2 * i + 1] = __half2float(h2.y);
        }
    }

    const int sc_groups = HEAD_DIM / 16;
    const int lane_group = (lane * ELEMS) / 16;
    float k_scale = ue4m3_decode(K_scales[tok * sc_groups + lane_group]);
    const __half2 k_scale_h2 = __float2half2_rn(k_scale);

    float dot = 0.0f;
    const uint8_t* k_bytes = K + tok * (HEAD_DIM / 2) + lane * ELEMS / 2;
    #pragma unroll
    for (int i = 0; i < ELEMS / 2; i++) {
        __half2 kh2 = fp4_byte_to_half2(k_bytes[i]);
        kh2 = __hmul2(kh2, k_scale_h2);
        float2 kf = __half22float2(kh2);
        dot = __fmaf_rn(q_reg[2 * i],     kf.x, dot);
        dot = __fmaf_rn(q_reg[2 * i + 1], kf.y, dot);
    }

    // Warp reduce
    for (int off = 16; off > 0; off >>= 1)
        dot += __shfl_xor_sync(0xffffffff, dot, off);
    if (lane == 0) out[tok] = dot;
}

// ---------------------------------------------------------------------------
// Phase 0 main: smoke that the scalar reference path compiles + runs.
// ---------------------------------------------------------------------------
int main() {
    constexpr int HEAD_DIM = 128;
    constexpr int seqlen_kv = 4096;

    std::vector<__half> Q_h(HEAD_DIM);
    std::vector<uint8_t> K_h(seqlen_kv * HEAD_DIM / 2);
    std::vector<uint8_t> Ks_h(seqlen_kv * HEAD_DIM / 16);
    for (int i = 0; i < HEAD_DIM; i++) Q_h[i] = __float2half(0.01f * (i % 17 - 8));
    for (size_t i = 0; i < K_h.size(); i++)  K_h[i]  = static_cast<uint8_t>(i & 0xff);
    for (size_t i = 0; i < Ks_h.size(); i++) Ks_h[i] = static_cast<uint8_t>(0x38);  // ~exp=7 mid-range

    half *d_Q; uint8_t *d_K, *d_Ks; float *d_out;
    cudaMalloc(&d_Q, HEAD_DIM * sizeof(half));
    cudaMalloc(&d_K, K_h.size());
    cudaMalloc(&d_Ks, Ks_h.size());
    cudaMalloc(&d_out, seqlen_kv * sizeof(float));
    cudaMemcpy(d_Q, Q_h.data(), HEAD_DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_h.data(), K_h.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ks, Ks_h.data(), Ks_h.size(), cudaMemcpyHostToDevice);

    qk_dot_scalar_kernel<HEAD_DIM><<<seqlen_kv, 32>>>(d_Q, d_K, d_Ks, d_out, seqlen_kv);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FAIL: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<float> out(seqlen_kv);
    cudaMemcpy(out.data(), d_out, seqlen_kv * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Phase 0 scalar reference: out[0]=%.4f out[100]=%.4f out[4095]=%.4f\n",
           out[0], out[100], out[4095]);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_Ks); cudaFree(d_out);
    return 0;
}
```

- [ ] **Step 2: Build + run the scalar reference**

Run:
```bash
cd $REPO
/usr/local/cuda/bin/nvcc -O2 --generate-code=arch=compute_120a,code=sm_120a \
  tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu \
  -o /tmp/bench_nvfp4_qk_tc_vs_scalar
/tmp/bench_nvfp4_qk_tc_vs_scalar
```
Expected: prints three non-zero finite floats (e.g. `out[0]=-0.0413 out[100]=0.1247 out[4095]=0.0631`). If all-zero or NaN, the dequant or scale path is broken — debug before continuing.

- [ ] **Step 3: Commit**

```bash
git add tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu
git commit -m "$(cat <<'EOF'
bench(nvfp4): add Phase-0 microbench scaffold for BitDecoding port

Standalone CUDA bench (no imp deps) reproducing the scalar-FFMA Q.K dot
from src/compute/attention_paged_nvfp4.cu:142-153 on synthetic NVFP4
KV. First step of the BitDecoding port plan
(docs/superpowers/plans/2026-05-09-bitdecoding-port.md). Phase 0 will
add a HMMA-MMA variant in a follow-up commit and bench the two against
each other; this commit ships only the reference path and confirms the
shape compiles + runs cleanly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2: Add the HMMA-MMA Q.K dot kernel

**Files:**
- Modify: `tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu`

The TC kernel uses `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` — the simplest Tensor-Core MMA variant for FP16 inputs / FP16 accumulator on Blackwell. Layout: each warp computes one 16×8 output tile (16 query rows × 8 KV tokens) per MMA issue. For decode (seqlen_q=1) we replicate Q across 16 rows and only consume row 0 of the output.

- [ ] **Step 1: Add the TC kernel above main()**

Add this kernel **above the `main()` function** in `tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu`:

```cpp
// ---------------------------------------------------------------------------
// TC kernel: HMMA-MMA Q.K dot on dequantized NVFP4 KV.
// Same input layout as scalar version. Each block processes 8 KV tokens
// (n8 in the m16n8k16 MMA). Q is replicated across 16 rows (m16); only
// row 0 of each output tile is used.
//
// Layout (per warp):
//   A operand (Q):  [16, 16] fp16, row-major. Q replicated 16x across rows.
//   B operand (K):  [8, 16] fp16, col-major. 8 KV tokens × 16 head-dim chunk.
//   C/D accum:      [16, 8] fp16. Row 0 = Q.K_token[0..7] dot products.
// We loop HEAD_DIM/16 times to cover full head_dim, accumulating into D.
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void qk_dot_tc_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K,
    const uint8_t* __restrict__ K_scales,
    float* __restrict__ out, int seqlen_kv) {
    static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be multiple of 16 for m16n8k16");
    constexpr int K_TILES = HEAD_DIM / 16;

    const int n8_block = blockIdx.x;                  // which group of 8 tokens
    const int tok_base = n8_block * 8;
    if (tok_base >= seqlen_kv) return;

    const int lane = threadIdx.x;                     // 0..31

    // Shared mem: dequantized Q [16,16] + K [8,16] per K_TILES iteration
    __shared__ __half sQ[16 * 16];
    __shared__ __half sK[8 * 16];

    // Pre-load + replicate Q into sQ[16,16] once per block (Q is single row)
    // Each lane writes 4 elems of the head_dim chunk; loop covers all chunks below.
    // Initialize accumulator (one row per lane mapping per mma.sync convention)
    half2 d0 = __half2half2(__float2half(0.0f));
    half2 d1 = __half2half2(__float2half(0.0f));

    const int sc_groups = HEAD_DIM / 16;

    for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
        const int hd_off = k_tile * 16;

        // Load Q[hd_off : hd_off+16] into sQ rows 0..15 (replicated)
        if (lane < 16) {
            __half q_lane[16];
            #pragma unroll
            for (int i = 0; i < 16; i++) q_lane[i] = Q[hd_off + i];
            #pragma unroll
            for (int i = 0; i < 16; i++) sQ[lane * 16 + i] = q_lane[i];
        }

        // Load K[tok_base..tok_base+8, hd_off..hd_off+16] dequantized
        // Each lane handles 4 of the 128 = 8*16 elems (or 0 if past end)
        for (int i = lane; i < 8 * 16; i += 32) {
            int k_tok = tok_base + i / 16;
            int k_hd  = hd_off + (i % 16);
            if (k_tok < seqlen_kv) {
                int byte_off = k_tok * (HEAD_DIM / 2) + k_hd / 2;
                uint8_t b = K[byte_off];
                __half2 hh = fp4_byte_to_half2(b);
                __half v = (k_hd & 1) ? hh.y : hh.x;
                float scale = ue4m3_decode(K_scales[k_tok * sc_groups + (k_hd / 16)]);
                sK[i] = __float2half(__half2float(v) * scale);
            } else {
                sK[i] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
        //   D[16,8] += A[16,16] * B[8,16]^T (B in col-major so it's [16,8] effectively)
        // Each warp lane holds 8 fp16 of A in 4 .b32 regs, 4 of B in 2 .b32 regs,
        // 4 of D in 2 .b32 regs (= 2 half2). See PTX ISA §9.7.13.4.4.
        uint32_t a0, a1, a2, a3;
        uint32_t b0, b1;

        // Per ptx ISA m16n8k16 row-major A: lane (i,j) maps to row (i%16), col (8*j + (lane%8 + ... ))
        // Simplest correct mapping: use ldmatrix to load both fragments.
        // sQ is row-major, sK is row-major as 8x16 (kv × hd_chunk).
        const __half* sQ_lane = &sQ[(lane % 16) * 16 + (lane / 16) * 8];
        const __half* sK_lane = &sK[(lane % 8) * 16 + (lane / 8) * 8];
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "l"(__cvta_generic_to_shared(sQ_lane)));
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
            "{%0,%1}, [%2];"
            : "=r"(b0), "=r"(b1)
            : "l"(__cvta_generic_to_shared(sK_lane)));

        uint32_t d0_u = *reinterpret_cast<uint32_t*>(&d0);
        uint32_t d1_u = *reinterpret_cast<uint32_t*>(&d1);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
            : "+r"(d0_u), "+r"(d1_u)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        d0 = *reinterpret_cast<half2*>(&d0_u);
        d1 = *reinterpret_cast<half2*>(&d1_u);

        __syncthreads();
    }

    // After all K_TILES, d0/d1 hold the 16×8 result tile.
    // Per ptx ISA, lane L holds D[2*(L/4), 2*(L%4)+(0|1)] in d0 + same +8 cols in d1.
    // Row 0 of the output corresponds to lanes where (L/4)==0 → lanes 0..3.
    // Each of those lanes holds 2 elems in d0 (cols 2*(L%4)+0/1), giving cols 0..7.
    if ((lane / 4) == 0) {
        int col_lo = 2 * (lane % 4);
        int tok0 = tok_base + col_lo;
        int tok1 = tok_base + col_lo + 1;
        if (tok0 < seqlen_kv) out[tok0] = __half2float(d0.x);
        if (tok1 < seqlen_kv) out[tok1] = __half2float(d0.y);
    }
}
```

- [ ] **Step 2: Update main() to run BOTH kernels and compare**

Replace the existing `main()` body in `tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu` with a version that runs both kernels and compares outputs. **Replace** the whole `main` function:

```cpp
int main() {
    constexpr int HEAD_DIM = 128;
    constexpr int seqlen_kv = 4096;

    std::vector<__half> Q_h(HEAD_DIM);
    std::vector<uint8_t> K_h(seqlen_kv * HEAD_DIM / 2);
    std::vector<uint8_t> Ks_h(seqlen_kv * HEAD_DIM / 16);
    for (int i = 0; i < HEAD_DIM; i++) Q_h[i] = __float2half(0.01f * (i % 17 - 8));
    for (size_t i = 0; i < K_h.size(); i++)  K_h[i]  = static_cast<uint8_t>(i & 0xff);
    for (size_t i = 0; i < Ks_h.size(); i++) Ks_h[i] = static_cast<uint8_t>(0x38);

    half *d_Q; uint8_t *d_K, *d_Ks; float *d_out_scalar, *d_out_tc;
    cudaMalloc(&d_Q, HEAD_DIM * sizeof(half));
    cudaMalloc(&d_K, K_h.size());
    cudaMalloc(&d_Ks, Ks_h.size());
    cudaMalloc(&d_out_scalar, seqlen_kv * sizeof(float));
    cudaMalloc(&d_out_tc,     seqlen_kv * sizeof(float));
    cudaMemcpy(d_Q, Q_h.data(), HEAD_DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_h.data(), K_h.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ks, Ks_h.data(), Ks_h.size(), cudaMemcpyHostToDevice);

    qk_dot_scalar_kernel<HEAD_DIM><<<seqlen_kv, 32>>>(d_Q, d_K, d_Ks, d_out_scalar, seqlen_kv);
    cudaDeviceSynchronize();
    qk_dot_tc_kernel<HEAD_DIM><<<(seqlen_kv + 7) / 8, 32>>>(d_Q, d_K, d_Ks, d_out_tc, seqlen_kv);
    cudaDeviceSynchronize();

    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        printf("FAIL kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<float> out_s(seqlen_kv), out_t(seqlen_kv);
    cudaMemcpy(out_s.data(), d_out_scalar, seqlen_kv * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_t.data(), d_out_tc,     seqlen_kv * sizeof(float), cudaMemcpyDeviceToHost);

    float max_abs_err = 0.0f, max_val = 0.0f;
    for (int i = 0; i < seqlen_kv; i++) {
        float e = fabsf(out_s[i] - out_t[i]);
        if (e > max_abs_err) max_abs_err = e;
        if (fabsf(out_s[i]) > max_val) max_val = fabsf(out_s[i]);
    }
    printf("scalar[0..2]=%.4f %.4f %.4f\n", out_s[0], out_s[1], out_s[2]);
    printf("tc    [0..2]=%.4f %.4f %.4f\n", out_t[0], out_t[1], out_t[2]);
    printf("max_abs_err=%.4e  max_val=%.4f  rel=%.4e\n",
           max_abs_err, max_val, max_abs_err / (max_val + 1e-9f));

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_Ks);
    cudaFree(d_out_scalar); cudaFree(d_out_tc);
    return (max_abs_err < 1e-2f * max_val) ? 0 : 2;
}
```

- [ ] **Step 3: Build + run the comparison**

Run:
```bash
/usr/local/cuda/bin/nvcc -O2 --generate-code=arch=compute_120a,code=sm_120a \
  tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu \
  -o /tmp/bench_nvfp4_qk_tc_vs_scalar
/tmp/bench_nvfp4_qk_tc_vs_scalar
```
Expected: `max_abs_err` and `rel` in the 1e-3 to 1e-2 range, exit code 0. If `max_abs_err` is >1e-1 of `max_val`, the TC kernel's layout/ldmatrix is wrong — debug.

If exit code 2: the relative error gate failed. Inspect the first few values to see whether it's a constant offset (sign error in dequant) or random noise (layout mismatch). The `out_s` and `out_t` printouts give the comparison directly.

- [ ] **Step 4: Commit**

```bash
git add tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu
git commit -m "$(cat <<'EOF'
bench(nvfp4): add HMMA-MMA Q.K dot kernel + numerical equivalence

Adds qk_dot_tc_kernel to the Phase-0 microbench: same NVFP4 input,
same dequant path, but the inner Q.K dot is dispatched via
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 instead of
__fmaf_rn. main() runs both kernels and asserts max_abs_err < 1%
of max_val.

Phase 0 of the BitDecoding port
(docs/superpowers/plans/2026-05-09-bitdecoding-port.md).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3: Add timing harness + perf comparison

**Files:**
- Modify: `tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu`

- [ ] **Step 1: Add cudaEvent timing wrappers + a longer-running outer loop**

Modify `main()` to wrap each kernel invocation in a `cudaEvent`-based timing harness with 100 iterations. **Replace** the kernel-launch + result-print section with:

```cpp
    // Warmup
    for (int i = 0; i < 3; i++) {
        qk_dot_scalar_kernel<HEAD_DIM><<<seqlen_kv, 32>>>(d_Q, d_K, d_Ks, d_out_scalar, seqlen_kv);
        qk_dot_tc_kernel<HEAD_DIM><<<(seqlen_kv + 7) / 8, 32>>>(d_Q, d_K, d_Ks, d_out_tc, seqlen_kv);
    }
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    constexpr int REPS = 100;

    cudaEventRecord(a);
    for (int i = 0; i < REPS; i++)
        qk_dot_scalar_kernel<HEAD_DIM><<<seqlen_kv, 32>>>(d_Q, d_K, d_Ks, d_out_scalar, seqlen_kv);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float scalar_ms = 0.0f;
    cudaEventElapsedTime(&scalar_ms, a, b);
    scalar_ms /= REPS;

    cudaEventRecord(a);
    for (int i = 0; i < REPS; i++)
        qk_dot_tc_kernel<HEAD_DIM><<<(seqlen_kv + 7) / 8, 32>>>(d_Q, d_K, d_Ks, d_out_tc, seqlen_kv);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float tc_ms = 0.0f;
    cudaEventElapsedTime(&tc_ms, a, b);
    tc_ms /= REPS;

    cudaEventDestroy(a); cudaEventDestroy(b);

    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        printf("FAIL kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<float> out_s(seqlen_kv), out_t(seqlen_kv);
    cudaMemcpy(out_s.data(), d_out_scalar, seqlen_kv * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_t.data(), d_out_tc,     seqlen_kv * sizeof(float), cudaMemcpyDeviceToHost);

    float max_abs_err = 0.0f, max_val = 0.0f;
    for (int i = 0; i < seqlen_kv; i++) {
        float e = fabsf(out_s[i] - out_t[i]);
        if (e > max_abs_err) max_abs_err = e;
        if (fabsf(out_s[i]) > max_val) max_val = fabsf(out_s[i]);
    }

    printf("Phase 0 microbench: HEAD_DIM=%d seqlen_kv=%d\n", HEAD_DIM, seqlen_kv);
    printf("  scalar (FFMA): %.4f ms / iter\n", scalar_ms);
    printf("  tc     (HMMA): %.4f ms / iter\n", tc_ms);
    printf("  speedup: %.2fx (TC vs scalar)\n", scalar_ms / tc_ms);
    printf("  max_abs_err=%.4e  rel=%.4e\n", max_abs_err, max_abs_err / (max_val + 1e-9f));
    printf("  scalar[0..2]=%.4f %.4f %.4f\n", out_s[0], out_s[1], out_s[2]);
    printf("  tc    [0..2]=%.4f %.4f %.4f\n", out_t[0], out_t[1], out_t[2]);
```

- [ ] **Step 2: Run the timed comparison**

Run:
```bash
/usr/local/cuda/bin/nvcc -O2 --generate-code=arch=compute_120a,code=sm_120a \
  tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu \
  -o /tmp/bench_nvfp4_qk_tc_vs_scalar
/tmp/bench_nvfp4_qk_tc_vs_scalar
```
Expected: `speedup: X.YYx (TC vs scalar)` printed. Target X is in the 1.5–4× range for this single Q.K dot kernel — much narrower than BitDecoding's full-attention 8.6× claim because we're isolating just the Q.K dot, not the full attention pipeline.

If `speedup < 1.0`: the TC layout is dominated by `ldmatrix` overhead at this small size — would need to batch more KV tokens per warp before the MMA cost amortizes. Note the result and proceed to commit; Phase 1 will inform whether to push for higher.

- [ ] **Step 3: SASS audit the bench binary to confirm HMMA fires**

Run:
```bash
/usr/local/cuda/bin/cuobjdump --dump-sass /tmp/bench_nvfp4_qk_tc_vs_scalar 2>/dev/null | grep -cE "HMMA"
```
Expected: ≥ 1. Zero means ptxas inlined the inline-asm to FFMA (very unlikely for inline `mma.sync` PTX, but possible). If zero, both kernels actually executed via FFMA and the speedup measurement is meaningless.

- [ ] **Step 4: Commit**

```bash
git add tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu
git commit -m "$(cat <<'EOF'
bench(nvfp4): add cudaEvent timing harness for Phase-0 microbench

100-iter timed comparison of scalar-FFMA vs HMMA-MMA Q.K dot on
synthetic NVFP4 KV at HEAD_DIM=128 seqlen_kv=4096. Reports per-iter
ms + speedup ratio + max_abs_err. SASS check confirmed HMMA presence.

Phase 0 of the BitDecoding port plan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4: Wrapper script + roadmap entry

**Files:**
- Create: `tools/analysis/bench_nvfp4_qk_tc_vs_scalar.sh`
- Modify: `docs/roadmap.md`

- [ ] **Step 1: Create the wrapper script**

Create `tools/analysis/bench_nvfp4_qk_tc_vs_scalar.sh`:

```bash
#!/usr/bin/env bash
# Phase-0 microbench for the BitDecoding NVFP4 port. Builds + runs the
# scalar-FFMA vs HMMA-MMA Q.K dot comparison on synthetic input, then
# SASS-audits the binary to confirm HMMA dispatch.
#
# Re-run after each kernel-layout change to verify numerical equivalence
# stays within tolerance and HMMA dispatch is preserved.
#
# Usage: bash tools/analysis/bench_nvfp4_qk_tc_vs_scalar.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SRC="$REPO_ROOT/tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu"
BIN=/tmp/bench_nvfp4_qk_tc_vs_scalar

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
CUOBJDUMP=${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}

echo "=== build ==="
"$NVCC" -O2 --generate-code=arch=compute_120a,code=sm_120a "$SRC" -o "$BIN"

echo
echo "=== run ==="
"$BIN"

echo
echo "=== SASS HMMA check ==="
HMMA=$("$CUOBJDUMP" --dump-sass "$BIN" 2>/dev/null | grep -cE "HMMA" || true)
echo "  HMMA instruction count: $HMMA"
if [ "$HMMA" -eq 0 ]; then
    echo "  ⚠ Zero HMMA — TC dispatch did not survive ptxas. Bench result is invalid."
    exit 1
fi
echo "  ✓ HMMA dispatch confirmed."
```

- [ ] **Step 2: Make it executable + run it**

Run:
```bash
chmod +x tools/analysis/bench_nvfp4_qk_tc_vs_scalar.sh
bash tools/analysis/bench_nvfp4_qk_tc_vs_scalar.sh
```
Expected: build succeeds, comparison runs, SASS reports `HMMA instruction count > 0`.

- [ ] **Step 3: Update roadmap.md BitDecoding entry to cite Phase-0 result**

Open `docs/roadmap.md`. Find the line:

```markdown
- **BitDecoding** ([arxiv:2503.18773](https://arxiv.org/abs/2503.18773), HPCA 2026) — **HIGHEST-ROI item** in this section.
```

After the existing sentence about `tools/analysis/sass_nvfp4_paged_decode.sh` confirming 0 HMMA, append:

```markdown
Phase-0 microbench (`tools/analysis/bench_nvfp4_qk_tc_vs_scalar.sh`) shipped 2026-05-09: scalar-FFMA vs HMMA-MMA Q.K dot on synthetic NVFP4 input, numerical equivalence within 1% relative error, TC speedup measured at <fill-in-from-bench>×.
```

Replace `<fill-in-from-bench>` with the actual speedup printed by the bench script in Step 2 (e.g. `2.3` for 2.3× speedup).

- [ ] **Step 4: Build imp + run existing tests to confirm no regression**

The bench is standalone, but the .sh script is now in `tools/analysis/` which is in the build's source tree. Verify nothing imp-side broke:
```bash
make build
make test-gpu 2>&1 | grep -E "PASSED|FAILED" | tail -5
```
Expected: 78 PASSED, 0 FAILED. If anything breaks: the bench .cu file probably picked up an unintended include — debug.

- [ ] **Step 5: Commit**

```bash
git add tools/analysis/bench_nvfp4_qk_tc_vs_scalar.sh docs/roadmap.md
git commit -m "$(cat <<'EOF'
bench(nvfp4): wrapper script + roadmap update for Phase-0 BitDecoding

Adds tools/analysis/bench_nvfp4_qk_tc_vs_scalar.sh — re-runnable
build + run + SASS-audit pipeline for the Phase-0 microbench.

Roadmap entry now cites the Phase-0 measured speedup (TC vs scalar
on isolated Q.K dot, synthetic input). This grounds the next step
(Phase 1: port into production attention_paged_nvfp4_tc.cu kernel).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5: Push Phase 0 + create PR

**Files:** none modified — just git operations.

- [ ] **Step 1: Push branch + create PR**

Run:
```bash
git push -u origin <branch-name>
gh pr create --base main --title "bench(nvfp4): Phase-0 BitDecoding microbench + scalar/TC equivalence" --body "$(cat <<'EOF'
## Summary

Phase 0 of the BitDecoding NVFP4 paged decode port (per docs/superpowers/plans/2026-05-09-bitdecoding-port.md).

Standalone CUDA microbench in `tools/analysis/`:
- `bench_nvfp4_qk_tc_vs_scalar.cu` — two kernels on the same NVFP4 input: a scalar-FFMA Q.K dot (mirrors imp's current paged decode at `attention_paged_nvfp4.cu:142-153`) and an HMMA-MMA Q.K dot via `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`. Verifies numerical equivalence (max_abs_err < 1% of max_val) and reports the speedup ratio.
- `bench_nvfp4_qk_tc_vs_scalar.sh` — re-runnable build + bench + SASS-audit wrapper.

## Phase 0 result (RTX 5090, HEAD_DIM=128, seqlen_kv=4096)

(Insert measured numbers from running the bench.)

- scalar (FFMA): X.XXX ms/iter
- tc (HMMA): X.XXX ms/iter
- speedup: X.YYx
- max_abs_err: X.XXe-X (rel)
- HMMA dispatch confirmed (SASS HMMA count > 0)

## Phase 1 prerequisites met

✅ NVFP4 dequant path (`cvt.rn.f16x2.e2m1x2`) produces correct half-precision values.
✅ UE4M3 group-scale fold matches the FFMA reference numerically.
✅ `mma.sync.aligned.m16n8k16` accepts the dequantized fragments via `ldmatrix`.
✅ Speedup is measurable on isolated Q.K dot (full-attention impact will be larger after PV is also TC).

## Test plan

- [x] Bench runs, equivalence within tolerance.
- [x] SASS check confirms HMMA dispatch (not silently downgraded by ptxas).
- [x] `make build` clean, `make test-gpu` 78/0/18 passes (no imp-side regression — bench is standalone).

## Next step

Phase 1: port the TC dispatch into a new `src/compute/attention_paged_nvfp4_tc.cu` (parallel to the existing `attention_paged_nvfp4.cu`), gated via `IMP_USE_BITDECODING_QK=1` env var, with a GTest numerical-equivalence test against the existing kernel.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Replace `<branch-name>` with the actual current branch name (e.g. `phase0-bitdecoding-bench`).

Expected: PR opens, CI runs (only `Build (CUDA 13.2)` job, since no test paths changed), auto-merge fires on green.

---

## Phase 1: Port TC Q.K dot into production kernel (Tasks 6–10)

**Goal**: take the validated TC Q.K dot pattern from Phase 0 and apply it to a real production-grade `paged_attention_decode_nvfp4_tc_kernel`. The new kernel mirrors `paged_attention_decode_nvfp4_kernel` byte-for-byte except for the inner Q.K dot loop. Wired via `IMP_USE_BITDECODING_QK=1` env-var so it's strictly opt-in. Numerical equivalence test gates correctness; SASS audit gates the dispatch claim; perf A/B grounds the production case.

**Why env-var instead of CLI flag in Phase 1**: avoids touching `tools/imp-cli/args.cpp` and config-schema validation in this PR. CLI flag (`--use-bitdecoding-nvfp4`) lands in Phase 4 (production opt-in) once Phase 1 + 2 + 3 perf data justifies the CLI surface.

### Task 6: Fork the kernel into `attention_paged_nvfp4_tc.cu`

**Files:**
- Create: `src/compute/attention_paged_nvfp4_tc.cu`
- Modify: `src/compute/attention_paged.h:60-90` (declaration block for paged decode entry points)
- Modify: `CMakeLists.txt` (add to `IMP_COMPUTE_SOURCES` list)

- [ ] **Step 1: Copy the existing kernel as starting point**

```bash
cp src/compute/attention_paged_nvfp4.cu src/compute/attention_paged_nvfp4_tc.cu
```

- [ ] **Step 2: Rename the kernel functions in the new file**

In `src/compute/attention_paged_nvfp4_tc.cu`:
- Rename `paged_attention_decode_nvfp4_kernel` → `paged_attention_decode_nvfp4_tc_kernel` (3 occurrences: declaration, splitk variant call site, dispatch).
- Rename `paged_attention_splitk_nvfp4_kernel` → `paged_attention_splitk_nvfp4_tc_kernel` (3 occurrences).
- Rename `paged_attention_decode_nvfp4` → `paged_attention_decode_nvfp4_tc` (1 occurrence: top-level entry point).
- Update the `IMP_LOG_ERROR` strings to match new names.

Use sed for safety:
```bash
sed -i 's/paged_attention_decode_nvfp4_kernel/paged_attention_decode_nvfp4_tc_kernel/g; \
        s/paged_attention_splitk_nvfp4_kernel/paged_attention_splitk_nvfp4_tc_kernel/g; \
        s/paged_attention_decode_nvfp4(/paged_attention_decode_nvfp4_tc(/g; \
        s/paged_attention_decode_nvfp4 splitk/paged_attention_decode_nvfp4_tc splitk/g' \
       src/compute/attention_paged_nvfp4_tc.cu
```

Verify the new file has no name collisions:
```bash
grep -c "_nvfp4_kernel\|_nvfp4(" src/compute/attention_paged_nvfp4_tc.cu
# Expected: 0  — all should now end in _tc or _tc_kernel
```

- [ ] **Step 3: Add the new entry-point declaration to attention_paged.h**

Open `src/compute/attention_paged.h`. Find the existing declaration block ending around line 70 (after `paged_attention_decode_nvfp4`). Add immediately after:

```cpp
// BitDecoding-style TC dispatch: same signature + semantics as
// paged_attention_decode_nvfp4 but routes the inner Q.K dot through
// mma.sync.aligned.m16n8k16 Tensor Core MMA. Phase 1 of the BitDecoding
// port (docs/superpowers/plans/2026-05-09-bitdecoding-port.md). V
// accumulation remains scalar in Phase 1; Phase 2 will add TC PV.
void paged_attention_decode_nvfp4_tc(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                     const Tensor& K_scales, const Tensor& V_scales,
                                     const Tensor& block_tables, const Tensor& context_lens,
                                     int block_size, float scale, int max_context_len,
                                     int sliding_window, float softcap, cudaStream_t stream);
```

- [ ] **Step 4: Register the new file in CMakeLists.txt**

Open `CMakeLists.txt`. Find the line:

```cmake
src/compute/attention_paged_nvfp4.cu
```

Add the TC variant directly below it:

```cmake
src/compute/attention_paged_nvfp4.cu
src/compute/attention_paged_nvfp4_tc.cu
```

- [ ] **Step 5: Build to verify the fork compiles cleanly**

Run:
```bash
make build
```
Expected: clean Docker build. The TC kernel is functionally identical to the original at this point (just renamed), so this should succeed without changes.

If it fails on duplicate symbol errors: a function was missed in the rename — use `grep -n "_nvfp4(" src/compute/attention_paged_nvfp4_tc.cu` to find leftover names.

- [ ] **Step 6: Commit**

```bash
git add src/compute/attention_paged_nvfp4_tc.cu src/compute/attention_paged.h CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat(nvfp4): fork attention_paged_nvfp4 kernel for BitDecoding TC port

Creates src/compute/attention_paged_nvfp4_tc.cu as an exact copy of
attention_paged_nvfp4.cu with all kernel and entry-point names suffixed
with _tc. Functionally identical to the original at this point — the
TC dispatch lands in the next commit.

Phase 1 of the BitDecoding port. The fork-then-modify pattern keeps
the existing kernel untouched and reviewable in isolation while the
new path lights up.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7: Replace the Q.K dot with TC MMA in the new kernel

**Files:**
- Modify: `src/compute/attention_paged_nvfp4_tc.cu`

This is the core kernel change. Replace the inner scalar-FFMA Q.K dot loop in `paged_attention_decode_nvfp4_tc_kernel` (around line 142-153 of the file, mirroring `attention_paged_nvfp4.cu:142-153`) with an HMMA-based dispatch using the validated Phase-0 pattern.

The kernel currently processes one KV token per loop iteration with a per-lane partial dot reduced via `__shfl_xor`. To use HMMA we need to batch 8 tokens (m16n8k16's n=8 dimension) per MMA issue. Strategy: keep the outer per-token loop but unroll-by-8 and dispatch via MMA, with a scalar-tail for `tok_end - tok_start` not divisible by 8.

- [ ] **Step 1: Add NVFP4-to-half-shared-memory helper at the top of the file**

In `src/compute/attention_paged_nvfp4_tc.cu`, add the following helper inside the `imp` namespace, after the existing includes and before the first `__global__` kernel. (Reuses `fp4_byte_to_half2` from `ptx92_utils.cuh` — already included via the NVFP4 paged path.)

```cpp
// Dequantize 8 KV tokens × 16-element head-dim chunk into a contiguous
// shared-memory tile [8, 16] half, ready for ldmatrix into a B operand.
// Each lane handles 4 of the 128 elems (8 tokens × 16 dim / 32 lanes).
template <int HEAD_DIM>
__device__ __forceinline__ void dequant_kv_tile_to_smem(
    __half* sK,                          // [8, 16] half output (row-major)
    const uint8_t* K_block, int n_kv_heads, int kv_head, int block_size,
    int tok_start, int tok_end, int hd_off,
    const uint8_t* K_sc_block) {
    constexpr int sc_groups = HEAD_DIM / 16;
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_slot_stride = n_kv_heads * kv_head_bytes;
    const int sc_slot_stride = n_kv_heads * sc_groups;
    const int lane = threadIdx.x % 32;

    #pragma unroll
    for (int i = lane; i < 8 * 16; i += 32) {
        int tok_idx = i / 16;             // 0..7 within this 8-token group
        int hd_idx  = i % 16;             // 0..15 within the 16-elem chunk
        int t = tok_start + tok_idx;
        if (t < tok_end) {
            int hd_global = hd_off + hd_idx;
            int byte_off = t * kv_slot_stride + kv_head * kv_head_bytes + hd_global / 2;
            uint8_t b = K_block[byte_off];
            __half2 hh = fp4_byte_to_half2(b);
            __half v = (hd_global & 1) ? hh.y : hh.x;
            float scale = ue4m3_decode(
                K_sc_block[t * sc_slot_stride + kv_head * sc_groups + (hd_global / 16)]);
            sK[i] = __float2half(__half2float(v) * scale);
        } else {
            sK[i] = __float2half(0.0f);
        }
    }
}
```

- [ ] **Step 2: Replace the inner Q.K dot loop with TC dispatch + scalar tail**

Find the existing inner per-token loop in `paged_attention_decode_nvfp4_tc_kernel` (the block currently containing `dot = __fmaf_rn(...)` calls). It looks like:

```cpp
        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            // ... single-token Q.K dot via FFMA ...
        }
```

Replace it with the TC-batched version. The replacement processes 8 tokens at a time via MMA, falling back to the scalar path for the tail. The full replacement (paste in place of the existing inner loop):

```cpp
        // TC Q.K dot: 8-token MMA tiles + scalar tail.
        // Per ptx ISA m16n8k16 row.col.f16.f16.f16.f16:
        //   D[16,8] += A[16,16] * B[8,16]^T
        // We replicate the single Q row into A[0..15, :] and consume only
        // row 0 of D for our 8 token-dot results.
        constexpr int N_TILE = 8;
        constexpr int K_TILES = HEAD_DIM / 16;
        const int n_toks = tok_end - tok_start;
        const int n_full_tiles = (n_toks - first_tok) / N_TILE;
        const int tail_start = first_tok + n_full_tiles * N_TILE;

        // Per-tile dot accumulator (lane 0..3 only — they hold row 0 of D).
        // Each tile dispatches one MMA over K_TILES iterations, but we use
        // a fresh accumulator per tile and write the result before moving on.

        __shared__ __half sQ_smem[16 * 16];
        __shared__ __half sK_smem[8 * 16];

        // Pre-replicate Q's full HEAD_DIM into 16 rows of sQ
        if (warp_id == 0) {
            for (int i = lane_id; i < 16 * HEAD_DIM; i += 32) {
                int row = i / HEAD_DIM;
                int col = i % HEAD_DIM;
                if (row < 16) sQ_smem[(row % 16) * 16 + (col % 16)] =
                                  Q_ptr[col];
            }
        }
        __syncthreads();

        for (int tile = 0; tile < n_full_tiles; tile++) {
            const int t0 = tok_start + first_tok + tile * N_TILE;
            half2 d0 = __half2half2(__float2half(0.0f));
            half2 d1 = __half2half2(__float2half(0.0f));

            #pragma unroll
            for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
                const int hd_off = k_tile * 16;
                dequant_kv_tile_to_smem<HEAD_DIM>(
                    sK_smem, K_block, n_kv_heads, kv_head, block_size,
                    t0, t0 + N_TILE, hd_off, K_sc_block);
                __syncthreads();

                // ldmatrix loads + mma.sync issue
                uint32_t a0, a1, a2, a3, b0, b1;
                const __half* sQ_ld = &sQ_smem[(lane_id % 16) * 16];
                const __half* sK_ld = &sK_smem[(lane_id % 8) * 16];
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                    "{%0,%1,%2,%3}, [%4];"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                    : "l"(__cvta_generic_to_shared(sQ_ld)));
                asm volatile(
                    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
                    "{%0,%1}, [%2];"
                    : "=r"(b0), "=r"(b1)
                    : "l"(__cvta_generic_to_shared(sK_ld)));

                uint32_t d0_u = *reinterpret_cast<uint32_t*>(&d0);
                uint32_t d1_u = *reinterpret_cast<uint32_t*>(&d1);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                    "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                    : "+r"(d0_u), "+r"(d1_u)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                d0 = *reinterpret_cast<half2*>(&d0_u);
                d1 = *reinterpret_cast<half2*>(&d1_u);
                __syncthreads();
            }

            // Row 0 of D lives in lanes 0..3 (each holds 2 cols in d0).
            // Issue 8 partial dots via shared-mem scratch + softmax-online step.
            __shared__ float s_dot[8];
            if ((lane_id / 4) == 0) {
                int col_lo = 2 * (lane_id % 4);
                s_dot[col_lo]     = __half2float(d0.x);
                s_dot[col_lo + 1] = __half2float(d0.y);
            }
            __syncthreads();

            // Apply per-token scale + softcap + online softmax for each of the 8.
            #pragma unroll
            for (int t_off = 0; t_off < N_TILE; t_off++) {
                float dot = s_dot[t_off];
                dot *= scale;
                dot = apply_softcap(dot, softcap);

                float rescale, w_new;
                online_softmax_step(dot, m_w, l_w, rescale, w_new);

                // V accumulation (scalar — Phase 2 will TC this)
                int t = first_tok + tile * N_TILE + t_off;
                const uint8_t* v_bytes = V_block + t * kv_slot_stride
                                         + kv_head * kv_head_bytes
                                         + lane_offset / 2;
                float v_scale = ue4m3_decode(
                    V_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);
                const __half2 v_scale_h2 = __float2half2_rn(v_scale);
                #pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    __half2 vh2 = fp4_byte_to_half2(v_bytes[i]);
                    vh2 = __hmul2(vh2, v_scale_h2);
                    float2 vf = __half22float2(vh2);
                    o_reg[2 * i]     = __fmaf_rn(w_new, vf.x, rescale * o_reg[2 * i]);
                    o_reg[2 * i + 1] = __fmaf_rn(w_new, vf.y, rescale * o_reg[2 * i + 1]);
                }
            }
            __syncthreads();
        }

        // Scalar tail (fewer than N_TILE tokens left)
        for (int t = tail_start; t < n_toks; t++) {
            const uint8_t* K_tok = K_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            float k_scale = ue4m3_decode(
                K_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);
            float v_scale = ue4m3_decode(
                V_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);
            const __half2 k_scale_h2 = __float2half2_rn(k_scale);
            const __half2 v_scale_h2 = __float2half2_rn(v_scale);

            float dot = 0.0f;
            const uint8_t* k_bytes = K_tok + lane_offset / 2;
            #pragma unroll
            for (int i = 0; i < ELEMS / 2; i++) {
                __half2 kh2 = fp4_byte_to_half2(k_bytes[i]);
                kh2 = __hmul2(kh2, k_scale_h2);
                float2 kf = __half22float2(kh2);
                dot = __fmaf_rn(q_reg[2 * i],     kf.x, dot);
                dot = __fmaf_rn(q_reg[2 * i + 1], kf.y, dot);
            }
            dot = warp_reduce_sum(dot);
            dot *= scale;
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

            const uint8_t* v_bytes = V_tok + lane_offset / 2;
            #pragma unroll
            for (int i = 0; i < ELEMS / 2; i++) {
                __half2 vh2 = fp4_byte_to_half2(v_bytes[i]);
                vh2 = __hmul2(vh2, v_scale_h2);
                float2 vf = __half22float2(vh2);
                o_reg[2 * i]     = __fmaf_rn(w_new, vf.x, rescale * o_reg[2 * i]);
                o_reg[2 * i + 1] = __fmaf_rn(w_new, vf.y, rescale * o_reg[2 * i + 1]);
            }
        }
```

Note: this replacement assumes the kernel already has `m_w`, `l_w`, `o_reg`, `q_reg`, `lane_offset`, `lane_group`, `ELEMS`, `Q_ptr`, `K_block`, `V_block`, `K_sc_block`, `V_sc_block`, `kv_head`, `kv_head_bytes`, `kv_slot_stride`, `sc_groups`, `sc_slot_stride`, `n_kv_heads`, `block_size`, `warp_id`, `lane_id`, `scale`, and `softcap` in scope from earlier in the kernel — which they are, mirroring the original kernel structure. The same `apply_softcap`, `online_softmax_step`, `warp_reduce_sum` helpers are also already in scope from `attention_paged_nvfp4.cu`'s shared headers.

- [ ] **Step 3: Build to verify the kernel compiles**

Run:
```bash
make build
```
Expected: clean. If you get "use of undeclared identifier" errors, scroll up in the kernel for the variable definitions and verify the helper at Step 1 is inside the right namespace.

- [ ] **Step 4: SASS audit confirms HMMA appears**

Adapt the audit script to point at the TC kernel (the existing script targets `paged_attention_decode_nvfp4_kernel<HEAD_DIM>`; we need the new `paged_attention_decode_nvfp4_tc_kernel<HEAD_DIM>`).

Run inline:
```bash
CID=$(docker create imp:test)
docker cp ${CID}:/usr/local/bin/imp-cli /tmp/imp-cli-tc-audit
docker rm ${CID} > /dev/null
ELF=$(/usr/local/cuda/bin/cuobjdump --list-elf /tmp/imp-cli-tc-audit | grep sm_120a | head -1 | awk '{print $NF}')
TMP=$(mktemp -d) && pushd "$TMP" > /dev/null
/usr/local/cuda/bin/cuobjdump --extract-elf "$ELF" /tmp/imp-cli-tc-audit > /dev/null
/usr/local/cuda/bin/nvdisasm -c -ndf "$ELF" > /tmp/imp.tc.sass
popd > /dev/null && rm -rf "$TMP"
awk '/paged_attention_decode_nvfp4_tc_kernelILi128EE/{f=1} f && /^\/\/-+ \.text\.[^ ]+ -+$/ && !/_tc_kernel/ {if(s)exit} f{if(/^[[:space:]]+\/\*[0-9a-f]+\*\//)s=1; if(s)print}' /tmp/imp.tc.sass | grep -cE "HMMA"
```
Expected: ≥ 1. Zero means the kernel is now compiled but the MMA inline-asm got optimized away, which would imply the result of the MMA isn't reaching the output — a real bug.

- [ ] **Step 5: Commit**

```bash
git add src/compute/attention_paged_nvfp4_tc.cu
git commit -m "$(cat <<'EOF'
perf(nvfp4): swap Q.K dot to mma.sync in attention_paged_nvfp4_tc

Replaces the per-token scalar-FFMA Q.K dot loop in
paged_attention_decode_nvfp4_tc_kernel with an 8-token-batched
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 dispatch + scalar
tail. Pattern validated in Phase 0 microbench (PR #<phase0-pr>).
V accumulation remains scalar in this commit; Phase 2 will swap that.

SASS audit confirms HMMA dispatch in the new kernel; existing
attention_paged_nvfp4_kernel (default path) is unchanged.

Phase 1 of the BitDecoding port. The kernel is not yet wired in —
that lands in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 8: Wire the env-var opt-in in engine.cpp + dispatch

**Files:**
- Modify: `src/compute/attention_paged_nvfp4.cu` (entry point dispatch)
- Modify: `src/runtime/engine.cpp` (or wherever paged_attention_decode_nvfp4 is called)

The cleanest opt-in surface is at the dispatch level: when `IMP_USE_BITDECODING_QK=1` is set, route to `paged_attention_decode_nvfp4_tc` instead of `paged_attention_decode_nvfp4`. Implement at the call site rather than inside the dispatch entry point so the original kernel binary path is untouched when the flag is off.

- [ ] **Step 1: Find the call site for paged_attention_decode_nvfp4**

Run:
```bash
grep -rn "paged_attention_decode_nvfp4(" src/ --include="*.cu" --include="*.cpp" | grep -v "_tc"
```
Expected: a single dispatch in `src/graph/executor_attention.cu` or similar. Note the file + line number for the next step.

- [ ] **Step 2: Add an opt-in check + dispatch around the existing call**

In the file from Step 1, find the existing call:

```cpp
paged_attention_decode_nvfp4(Q, K_cache, V_cache, O,
                             K_scales, V_scales,
                             block_tables, context_lens,
                             block_size, scale, max_context_len,
                             sliding_window, softcap, stream);
```

Replace with the dispatch:

```cpp
static const bool use_tc = []() {
    const char* env = std::getenv("IMP_USE_BITDECODING_QK");
    return env && env[0] == '1';
}();
if (use_tc) {
    paged_attention_decode_nvfp4_tc(Q, K_cache, V_cache, O,
                                    K_scales, V_scales,
                                    block_tables, context_lens,
                                    block_size, scale, max_context_len,
                                    sliding_window, softcap, stream);
} else {
    paged_attention_decode_nvfp4(Q, K_cache, V_cache, O,
                                 K_scales, V_scales,
                                 block_tables, context_lens,
                                 block_size, scale, max_context_len,
                                 sliding_window, softcap, stream);
}
```

The `static const` ensures the env-var read is one-shot per process.

If `attention_paged.h` doesn't already include `<cstdlib>` for `std::getenv`, add the include at the top of the .cu file.

- [ ] **Step 3: Build + run smoke test**

```bash
make build
docker run --rm --gpus all -v $REPO/models:/models imp:test \
    imp-cli --model /models/Qwen3-8B-Q8_0.gguf \
    --prompt "The capital of France is" --max-tokens 32 --temperature 0 \
    --chat-template none --kv-nvfp4 2>&1 | tail -5

docker run --rm --gpus all -v $REPO/models:/models -e IMP_USE_BITDECODING_QK=1 imp:test \
    imp-cli --model /models/Qwen3-8B-Q8_0.gguf \
    --prompt "The capital of France is" --max-tokens 32 --temperature 0 \
    --chat-template none --kv-nvfp4 2>&1 | tail -5
```
Expected: both runs produce coherent output ending in "Paris". If TC path produces garbage, the kernel has a bug — debug before continuing. The TC pp/tg numbers will be reported separately.

- [ ] **Step 4: Commit**

```bash
git add src/graph/executor_attention.cu  # or wherever the dispatch lives
git commit -m "$(cat <<'EOF'
feat(nvfp4): wire IMP_USE_BITDECODING_QK env-var opt-in for TC dispatch

Adds a one-shot env-var check around the paged_attention_decode_nvfp4
call site. When IMP_USE_BITDECODING_QK=1, dispatch routes to the new
paged_attention_decode_nvfp4_tc kernel; otherwise the existing
scalar-FFMA path runs unchanged.

Smoke test: both paths produce identical "The capital of France is
Paris" greedy decode on Qwen3-8B Q8_0 with --kv-nvfp4. Production
opt-in CLI flag (--use-bitdecoding-nvfp4) lands in Phase 4 once
A/B perf data justifies the surface.

Phase 1 of the BitDecoding port.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 9: Add numerical-equivalence GTest

**Files:**
- Create: `tests/test_attention_paged_nvfp4_tc.cu`
- Modify: `tests/CMakeLists.txt` (register the new test source)

- [ ] **Step 1: Create the test fixture**

Create `tests/test_attention_paged_nvfp4_tc.cu`:

```cpp
#include <gtest/gtest.h>
#include "compute/attention_paged.h"
#include "compute/attention.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace imp {
namespace {

class AttentionPagedNvfp4TCTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        cudaStreamDestroy(stream_);
    }
    cudaStream_t stream_ = nullptr;

    // Helper to allocate + fill a Tensor on device
    void* alloc_filled(size_t bytes, std::function<void(uint8_t*)> filler) {
        std::vector<uint8_t> host(bytes);
        filler(host.data());
        void* dev = nullptr;
        cudaMalloc(&dev, bytes);
        cudaMemcpy(dev, host.data(), bytes, cudaMemcpyHostToDevice);
        return dev;
    }
};

TEST_F(AttentionPagedNvfp4TCTest, ScalarVsTC_HD128_NumericalEquivalence) {
    // Synthetic decode-shape NVFP4 paged attention input.
    constexpr int batch = 1;
    constexpr int n_heads = 32;
    constexpr int n_kv_heads = 32;
    constexpr int HEAD_DIM = 128;
    constexpr int seqlen_kv = 256;
    constexpr int block_size = 16;
    constexpr int n_blocks = (seqlen_kv + block_size - 1) / block_size;

    // Q [batch, n_heads, HEAD_DIM] half
    size_t q_bytes = batch * n_heads * HEAD_DIM * sizeof(half);
    void* d_Q = alloc_filled(q_bytes, [](uint8_t* p) {
        half* q = reinterpret_cast<half*>(p);
        for (int i = 0; i < 1 * 32 * 128; i++)
            q[i] = __float2half(0.01f * ((i % 17) - 8));
    });

    // K/V cache: [n_blocks, block_size, n_kv_heads, HEAD_DIM/2] uint8
    size_t kv_bytes = static_cast<size_t>(n_blocks) * block_size * n_kv_heads * (HEAD_DIM / 2);
    void* d_K = alloc_filled(kv_bytes, [](uint8_t* p) {
        for (size_t i = 0; i < kv_bytes; i++) p[i] = static_cast<uint8_t>((i * 7 + 3) & 0xff);
    });
    void* d_V = alloc_filled(kv_bytes, [](uint8_t* p) {
        for (size_t i = 0; i < kv_bytes; i++) p[i] = static_cast<uint8_t>((i * 11 + 5) & 0xff);
    });

    // K/V scales: [n_blocks, block_size, n_kv_heads, HEAD_DIM/16] uint8 (UE4M3)
    size_t sc_bytes = static_cast<size_t>(n_blocks) * block_size * n_kv_heads * (HEAD_DIM / 16);
    void* d_Ks = alloc_filled(sc_bytes, [](uint8_t* p) {
        for (size_t i = 0; i < sc_bytes; i++) p[i] = 0x38;  // mid-range
    });
    void* d_Vs = alloc_filled(sc_bytes, [](uint8_t* p) {
        for (size_t i = 0; i < sc_bytes; i++) p[i] = 0x38;
    });

    // block_tables: identity mapping
    std::vector<int32_t> bt(n_blocks);
    for (int i = 0; i < n_blocks; i++) bt[i] = i;
    int* d_bt = nullptr;
    cudaMalloc(&d_bt, n_blocks * sizeof(int));
    cudaMemcpy(d_bt, bt.data(), n_blocks * sizeof(int), cudaMemcpyHostToDevice);

    // context_lens
    int ctx_len = seqlen_kv;
    int* d_cl = nullptr;
    cudaMalloc(&d_cl, sizeof(int));
    cudaMemcpy(d_cl, &ctx_len, sizeof(int), cudaMemcpyHostToDevice);

    // Output buffers (one per kernel)
    void* d_O_scalar = nullptr;
    void* d_O_tc = nullptr;
    cudaMalloc(&d_O_scalar, q_bytes);
    cudaMalloc(&d_O_tc, q_bytes);
    cudaMemset(d_O_scalar, 0, q_bytes);
    cudaMemset(d_O_tc, 0, q_bytes);

    int64_t Q_shape[]   = {batch, 1, n_heads, HEAD_DIM};
    int64_t KV_shape[]  = {n_blocks, block_size, n_kv_heads, HEAD_DIM / 2};
    int64_t Sc_shape[]  = {n_blocks, block_size, n_kv_heads, HEAD_DIM / 16};
    int64_t bt_shape[]  = {batch, n_blocks};
    int64_t cl_shape[]  = {batch};

    Tensor Q(d_Q, QType::F16, 4, Q_shape, true);
    Tensor K(d_K, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor V(d_V, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor Ks(d_Ks, QType::U8, 4, Sc_shape, true);
    Tensor Vs(d_Vs, QType::U8, 4, Sc_shape, true);
    Tensor BT(d_bt, QType::I32, 2, bt_shape, true);
    Tensor CL(d_cl, QType::I32, 1, cl_shape, true);
    Tensor O_scalar(d_O_scalar, QType::F16, 4, Q_shape, true);
    Tensor O_tc(d_O_tc, QType::F16, 4, Q_shape, true);

    float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    int sliding_window = 0;
    float softcap = 0.0f;

    paged_attention_decode_nvfp4(Q, K, V, O_scalar, Ks, Vs, BT, CL,
                                 block_size, scale, ctx_len, sliding_window, softcap, stream_);
    paged_attention_decode_nvfp4_tc(Q, K, V, O_tc, Ks, Vs, BT, CL,
                                    block_size, scale, ctx_len, sliding_window, softcap, stream_);
    cudaStreamSynchronize(stream_);

    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<half> h_scalar(batch * n_heads * HEAD_DIM);
    std::vector<half> h_tc(batch * n_heads * HEAD_DIM);
    cudaMemcpy(h_scalar.data(), d_O_scalar, q_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tc.data(),     d_O_tc,     q_bytes, cudaMemcpyDeviceToHost);

    float max_abs_err = 0.0f, max_val = 0.0f;
    for (size_t i = 0; i < h_scalar.size(); i++) {
        float s = __half2float(h_scalar[i]);
        float t = __half2float(h_tc[i]);
        max_abs_err = std::max(max_abs_err, std::fabs(s - t));
        max_val     = std::max(max_val, std::fabs(s));
    }
    EXPECT_LT(max_abs_err, max_val * 1e-2f)
        << "TC kernel diverges from scalar reference: max_abs_err=" << max_abs_err
        << " max_val=" << max_val
        << " (scalar[0]=" << __half2float(h_scalar[0])
        << " tc[0]=" << __half2float(h_tc[0]) << ")";

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_Ks); cudaFree(d_Vs); cudaFree(d_bt); cudaFree(d_cl);
    cudaFree(d_O_scalar); cudaFree(d_O_tc);
}

}  // namespace
}  // namespace imp
```

- [ ] **Step 2: Register the test in tests/CMakeLists.txt**

Open `tests/CMakeLists.txt`. Find the existing list of test source files (look for `test_attention_*.cu`). Add `test_attention_paged_nvfp4_tc.cu` to that list, alphabetically:

Find the line ending in `test_attention_paged_*.cu` or similar (or just append to the end of the test sources list). Add:

```cmake
test_attention_paged_nvfp4_tc.cu
```

(If there's no current entry for `attention_paged_nvfp4` testing, add it as a new line in the list.)

- [ ] **Step 3: Build + run the new test**

```bash
make build
make test-gpu 2>&1 | grep -E "AttentionPagedNvfp4TCTest|FAILED" | tail -10
```
Expected: `AttentionPagedNvfp4TCTest.ScalarVsTC_HD128_NumericalEquivalence` PASSES. If it fails: read the EXPECT_LT message — it shows the first divergent values, which usually identifies whether it's a constant offset (algorithmic bug) or position-dependent error (layout bug).

- [ ] **Step 4: Run full test suite for regression check**

```bash
make test-gpu 2>&1 | grep -E "PASSED|FAILED" | tail -5
```
Expected: 79 PASSED, 0 FAILED, 18 SKIPPED (one more than before — the new test).

- [ ] **Step 5: Commit**

```bash
git add tests/test_attention_paged_nvfp4_tc.cu tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
test(nvfp4): numerical equivalence test for TC paged decode kernel

Adds tests/test_attention_paged_nvfp4_tc.cu with a synthetic-input
GTest fixture that runs both paged_attention_decode_nvfp4 (scalar
reference) and paged_attention_decode_nvfp4_tc (the new BitDecoding
TC variant) on the same input and asserts max_abs_err < 1% of
max_val.

Synthetic input is small enough (seqlen_kv=256 with HD=128, 1 batch,
32 heads) to fit in a fast unit test (<200 ms) but large enough to
exercise both the n_full_tiles MMA loop and the scalar tail of the
new kernel.

Phase 1 of the BitDecoding port.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 10: A/B perf bench + roadmap update + push Phase 1

**Files:**
- Modify: `docs/roadmap.md`
- Modify: `tools/analysis/sass_nvfp4_paged_decode.sh` (extend to also audit the TC kernel)

- [ ] **Step 1: Run E2E A/B comparison**

```bash
PROMPT="The history of artificial intelligence began in antiquity with myths and stories of artificial beings endowed with intelligence. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols."

echo "=== scalar (default) ==="
docker run --rm --gpus all -v $REPO/models:/models imp:test \
    imp-cli --model /models/Qwen3-8B-Q8_0.gguf \
    --prompt "$PROMPT" --max-tokens 64 --temperature 0 --chat-template none \
    --kv-nvfp4 2>&1 | grep -E "^pp |^tg |^total"

echo "=== TC (IMP_USE_BITDECODING_QK=1) ==="
docker run --rm --gpus all -v $REPO/models:/models -e IMP_USE_BITDECODING_QK=1 imp:test \
    imp-cli --model /models/Qwen3-8B-Q8_0.gguf \
    --prompt "$PROMPT" --max-tokens 64 --temperature 0 --chat-template none \
    --kv-nvfp4 2>&1 | grep -E "^pp |^tg |^total"
```
Capture the resulting `pp` and `tg` numbers for the PR description and the roadmap update.

- [ ] **Step 2: Extend the SASS audit script to support both kernels**

Open `tools/analysis/sass_nvfp4_paged_decode.sh`. Find the `KERNEL_PATTERN` line:

```bash
KERNEL_PATTERN="paged_attention_decode_nvfp4_kernelILi${HEAD_DIM}EE"
```

Replace with a dispatch on a second positional argument (`tc` or `scalar`, default `scalar`):

```bash
VARIANT="${2:-scalar}"
case "$VARIANT" in
    scalar) KERNEL_PATTERN="paged_attention_decode_nvfp4_kernelILi${HEAD_DIM}EE" ;;
    tc)     KERNEL_PATTERN="paged_attention_decode_nvfp4_tc_kernelILi${HEAD_DIM}EE" ;;
    *)
        echo "ERROR: unknown variant '$VARIANT' (expected 'scalar' or 'tc')" >&2
        exit 1
        ;;
esac
```

Update the script header comment to document the new positional arg:

```
# Usage: bash tools/analysis/sass_nvfp4_paged_decode.sh [HEAD_DIM] [VARIANT]
#   HEAD_DIM (default 128): Qwen3-style attention dim. Other supported: 64, 96, 256, 512.
#   VARIANT (default scalar): 'scalar' for the FFMA path, 'tc' for the BitDecoding TC variant.
```

- [ ] **Step 3: Run the audit on both variants**

```bash
bash tools/analysis/sass_nvfp4_paged_decode.sh 128 scalar
bash tools/analysis/sass_nvfp4_paged_decode.sh 128 tc
```
Expected: `scalar` reports HMMA=0, FFMA+FADD+FMUL=346 (unchanged); `tc` reports HMMA>0, FFMA+FADD+FMUL<346 (the V accumulation still uses scalar in Phase 1, so SCALAR will not be 0). Both variants should be within ulp tolerance per the GTest from Task 9.

- [ ] **Step 4: Update roadmap.md with Phase 1 result**

Open `docs/roadmap.md`. Find the BitDecoding entry. After the Phase-0 sentence (added in Task 4), append:

```markdown
Phase 1 shipped 2026-05-09 (PR #<phase1-pr>): forked the paged decode kernel into `attention_paged_nvfp4_tc.cu`, swapped Q.K dot to `mma.sync.aligned.m16n8k16` (V accumulation still scalar in this phase), wired via `IMP_USE_BITDECODING_QK=1` env-var opt-in. SASS audit confirms HMMA dispatch (`bash tools/analysis/sass_nvfp4_paged_decode.sh 128 tc`). E2E A/B on Qwen3-8B Q8_0 + `--kv-nvfp4` at pp=N: scalar tg=A, TC tg=B (Δ=ΔY%). Numerical equivalence within 1% rel error verified by `tests/test_attention_paged_nvfp4_tc.cu`.
```

Replace `<phase1-pr>`, `N`, `A`, `B`, `ΔY` with actual numbers from Step 1.

- [ ] **Step 5: Commit**

```bash
git add tools/analysis/sass_nvfp4_paged_decode.sh docs/roadmap.md
git commit -m "$(cat <<'EOF'
docs(roadmap): Phase-1 BitDecoding TC dispatch shipped + audit dispatch

Roadmap entry now records Phase-1 completion: TC Q.K dot in
attention_paged_nvfp4_tc.cu, env-var opt-in, equivalence test, A/B
perf data. The SASS audit script learns a second positional arg to
target the TC kernel variant in addition to the default scalar one.

Phase 2 (TC V accumulation) gates on Phase 1 A/B perf data showing
the QK-only swap is at minimum perf-neutral. If Phase 1 A/B shows
regression, decide whether to abandon or escalate to a different
kernel architecture before continuing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Push branch + create PR**

```bash
git push -u origin <phase1-branch>
gh pr create --base main --title "perf(nvfp4): Phase-1 BitDecoding TC Q.K dot dispatch" --body "$(cat <<'EOF'
## Summary

Phase 1 of the BitDecoding NVFP4 paged decode port (per docs/superpowers/plans/2026-05-09-bitdecoding-port.md).

Forks `src/compute/attention_paged_nvfp4.cu` → `attention_paged_nvfp4_tc.cu`. The new kernel is identical to the original except the inner Q.K dot loop swaps from per-token scalar `__fmaf_rn` to 8-token-batched `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` Tensor Core MMA. V accumulation remains scalar in Phase 1.

Wired via `IMP_USE_BITDECODING_QK=1` env-var. The default path (existing scalar-FFMA kernel) is byte-for-byte unchanged when the env-var is absent.

## Empirical results (RTX 5090)

**E2E A/B** on Qwen3-8B Q8_0 with `--kv-nvfp4`:

| Path | pp tok/s | tg tok/s |
|---|---|---|
| scalar (default) | A | B |
| TC (`IMP_USE_BITDECODING_QK=1`) | C | D |

(Insert measured numbers from PR-prep bench.)

**SASS audit** (HEAD_DIM=128):

| Variant | HMMA | FFMA+FADD+FMUL |
|---|---|---|
| scalar | 0 | 346 |
| tc | X | Y |

(Insert from `bash tools/analysis/sass_nvfp4_paged_decode.sh 128 {scalar,tc}`.)

**Numerical equivalence**: `tests/test_attention_paged_nvfp4_tc.cu` PASSES — TC output within 1% relative error of scalar reference on synthetic input (HEAD_DIM=128, seqlen_kv=256, 32 heads).

## Test plan

- [x] `make build` clean
- [x] `make test-gpu` 79 PASSED, 18 SKIPPED, 0 FAILED (one new test)
- [x] `make verify-fast` green (default path unchanged)
- [x] Smoke: Gemma-4 + `--kv-nvfp4` answers "Paris" with both env-var on and off

## Next step

Phase 2: TC V accumulation. Same kernel file, replaces the scalar V accum loop with a second `mma.sync.aligned.m16n8k16` dispatch operating on the dequantized V tile. Gates on Phase 1's A/B showing the QK-only swap is at minimum perf-neutral.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Replace `<phase1-branch>` with the actual current branch name.

---

## Phase 1 success criteria (gate for Phase 2)

Phase 1 PR merges only if **all** of the following hold:

1. **Numerical equivalence**: `tests/test_attention_paged_nvfp4_tc.cu` passes within 1% relative error.
2. **Tensor-Core dispatch confirmed**: `bash tools/analysis/sass_nvfp4_paged_decode.sh 128 tc` reports HMMA > 0.
3. **No regression on default path**: `make verify-fast` green; `make test-gpu` 79/0/18.
4. **Default-path SASS unchanged**: `bash tools/analysis/sass_nvfp4_paged_decode.sh 128 scalar` reports the same instruction profile as before Phase 1.
5. **TC perf is at minimum perf-neutral on E2E**: TC tg ≥ scalar tg − 5% on the Qwen3-8B Q8_0 + `--kv-nvfp4` A/B at pp=128 to pp=4096. (Below -5%, TC is regressing the QK side enough that V swap might not recover it — re-evaluate before Phase 2.)

If criterion 5 fails: capture the bench numbers in a memory memo and pause Phase 2 until the regression source is understood. The TC packing kernel may need a different MMA shape or a softmax-pipelining trick that wasn't in scope for Phase 1.

---

## Phase 2–5: deferred to follow-up plan

The remaining phases require Phase 1 outcome data to design correctly. Specifically:

- **Phase 2 (TC V accumulation)** depends on Phase 1's MMA layout choice — V accum may need a different B-operand stride than QK does, and the exact pipeline (MMA-issue vs ldmatrix-issue interleave) tuning is informed by the QK perf signal.
- **Phase 3 (FP16 residual cache)** depends on whether Phase 1+2 perf wins justify the residual-cache complexity; if Phase 1 closes the -3.3% gap on its own, Phase 3 may be premature.
- **Phase 4 (production CLI flag)** depends on at least one of {Phase 1, Phase 2, Phase 3} demonstrating measurable user-visible improvement.
- **Phase 5 (flip default)** depends on Phase 4's user-facing A/B harness.

A follow-up plan document will be written after Phase 1 PR merges, covering Phases 2–5 with the same bite-sized step granularity. The follow-up plan path will be `docs/superpowers/plans/2026-MM-DD-bitdecoding-port-phase2plus.md`.

---

## Self-review

**Spec coverage** (against `bitdecoding_sass_audit_2026_05_09.md` + `kv_research_grade_eval_2026_05_09.md`):

- ✅ Phase-0 microbench validates TC dispatch on synthetic NVFP4 input → Tasks 1–4
- ✅ Phase-1 production kernel TC dispatch → Tasks 6–7
- ✅ Env-var opt-in for safe rollout → Task 8
- ✅ Numerical equivalence test → Task 9
- ✅ SASS audit of new kernel → Task 7 Step 4 + Task 10 Step 3
- ✅ E2E A/B perf measurement → Task 10 Step 1
- ✅ Roadmap update with Phase-1 results → Task 10 Step 4

**Placeholder scan**: search performed for "TBD", "TODO", "implement later", "fill in details", "similar to", "add error handling", "handle edge cases":
- "Insert measured numbers" appears in PR body templates (Tasks 5, 10 Step 6) — these are run-time data the engineer fills in from their actual run, not plan placeholders.
- `<phase0-pr>`, `<phase1-pr>`, `<phase1-branch>`, `<branch-name>` are run-time identifiers with explicit instructions to replace.
- `<fill-in-from-bench>` in Task 4 Step 3 is one such placeholder, instructed to replace with measured value.
- No "TBD", "TODO", or "fill in details" patterns found.

**Type consistency**:
- `paged_attention_decode_nvfp4_tc` signature in Task 6 Step 3 (.h declaration) matches the entry-point usage in Task 8 Step 2 dispatch.
- `paged_attention_decode_nvfp4_tc_kernel` template parameter `<HEAD_DIM>` is consistent across Tasks 6, 7, 9, 10.
- `IMP_USE_BITDECODING_QK` env-var name is consistent in Tasks 8 (definition), 10 (bench).
- Test fixture name `AttentionPagedNvfp4TCTest` is consistent across the test file and the make-test-gpu grep in Task 9 Step 3.

**Scope check**: this plan covers Phases 0 + 1 only — clearly delimited. Phases 2–5 are deferred to a follow-up plan with explicit gating criteria.

No issues found.
