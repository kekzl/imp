// =============================================================================
// fp4_pv_bench.cu — Phase 3a FP4 PV microbench
// =============================================================================
//
// See bench/fp4_pv_bench.h for the design rationale (in short: discriminates
// whether single-level FP4 quantisation of post-softmax probabilities
// preserves enough numerical precision for the +13 % MMA-level upside to
// translate into useful end-to-end attention quality — before committing
// the multi-week Phase 3b/3c production work).
// =============================================================================

#include "bench/fp4_pv_bench.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace imp {

// -----------------------------------------------------------------------------
// FP4 + UE4M3 encode/decode (host-side reference)
// -----------------------------------------------------------------------------
//
// E2M1 magnitudes (sign bit separate): {0, 0.5, 1, 1.5, 2, 3, 4, 6}
// E4M3 unsigned: exponent 4 bits (bias 7), mantissa 3 bits → max=448, min>0=2^-9.
//
// Mirrors the software fallback in `src/compute/attention_fmha_mxfp4_sm120.cu`
// (`pack_fp4_pair` and `float_to_fp8_e4m3`). The PTX HW path uses IEEE RNE;
// our software path uses midpoint cascade — tiny divergence at boundaries
// that is not material for the long-tail truncation question Phase 3a tests.

static constexpr float kFp4Mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

static uint8_t quantise_fp4(float v) {
    uint8_t sign = (v < 0.0f) ? 1u : 0u;
    float a = std::fabs(v);
    int mag = (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f) +
              (a >= 2.5f)  + (a >= 3.5f)  + (a >= 5.0f);
    return static_cast<uint8_t>((sign << 3) | mag);
}

static float dequantise_fp4(uint8_t code) {
    int sign = (code >> 3) & 1;
    int mag = code & 7;
    float m = kFp4Mag[mag];
    return sign ? -m : m;
}

static uint8_t encode_ue4m3(float v) {
    if (v <= 0.0f) return 0;
    // Bias 7, 3 mantissa bits, no inf/NaN encoding (positive only).
    int exp;
    float frac = std::frexp(v, &exp);  // frac in [0.5, 1), v = frac * 2^exp
    int biased = (exp - 1) + 7;        // shift to normalised [1, 2) representation
    if (biased < 0) return 0;
    if (biased > 15) biased = 15;
    float normalised = std::ldexp(frac, 1);  // now in [1, 2)
    int mantissa = static_cast<int>(std::round((normalised - 1.0f) * 8.0f));
    if (mantissa >= 8) { mantissa = 0; biased++; if (biased > 15) biased = 15; }
    return static_cast<uint8_t>((biased << 3) | mantissa);
}

static float decode_ue4m3(uint8_t bits) {
    int exp = (bits >> 3) & 0xF;
    int mantissa = bits & 7;
    if (exp == 0) return 0.0f;  // simplification — actual subnormals not used here
    return std::ldexp(1.0f + mantissa / 8.0f, exp - 7);
}

// -----------------------------------------------------------------------------
// Per-16-element block-scale quantise + dequantise (matches PTX semantics)
// -----------------------------------------------------------------------------

static void quantise_row_fp4(const float* row, int K, std::vector<uint8_t>& codes_out,
                             std::vector<uint8_t>& scales_out) {
    const int n_groups = K / 16;
    codes_out.assign(static_cast<size_t>(K), 0);
    scales_out.assign(static_cast<size_t>(n_groups), 0);
    for (int g = 0; g < n_groups; ++g) {
        float absmax = 0.0f;
        for (int i = 0; i < 16; ++i)
            absmax = std::fmax(absmax, std::fabs(row[g * 16 + i]));
        float raw = absmax / 6.0f;  // 6 = max FP4 magnitude
        scales_out[g] = encode_ue4m3(raw);
        float dq = decode_ue4m3(scales_out[g]);
        float inv = (dq > 0.0f) ? (1.0f / dq) : 0.0f;
        for (int i = 0; i < 16; ++i)
            codes_out[g * 16 + i] = quantise_fp4(row[g * 16 + i] * inv);
    }
}

static void dequantise_row_fp4(const std::vector<uint8_t>& codes,
                               const std::vector<uint8_t>& scales, int K,
                               std::vector<float>& out) {
    const int n_groups = K / 16;
    out.assign(static_cast<size_t>(K), 0.0f);
    for (int g = 0; g < n_groups; ++g) {
        float dq = decode_ue4m3(scales[g]);
        for (int i = 0; i < 16; ++i)
            out[g * 16 + i] = dequantise_fp4(codes[g * 16 + i]) * dq;
    }
}

// Generate one synthetic post-softmax row.
//
// Pattern: softmax(N(0, 1) + spike_at_random) where spike strength is
// drawn from U(3, 8). This produces 1-3 high-mass values (0.05..0.95) and
// a long tail in [1e-4 .. 1e-1] — representative of real attention rows
// (`fp4_pv_potential_2026_04_25.md` cites the same distribution shape).
static void generate_postsoftmax_row(float* row, int K, std::mt19937& rng) {
    std::normal_distribution<float> logit_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> spike_idx(0, K - 1);
    std::uniform_real_distribution<float> spike_str(3.0f, 8.0f);

    std::vector<float> logits(static_cast<size_t>(K));
    for (int i = 0; i < K; ++i)
        logits[i] = logit_dist(rng);
    logits[spike_idx(rng)] += spike_str(rng);

    float maxl = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        logits[i] = std::exp(logits[i] - maxl);
        sum += logits[i];
    }
    for (int i = 0; i < K; ++i)
        row[i] = static_cast<float>(logits[i] / sum);
}

// -----------------------------------------------------------------------------
// Phase 3a accuracy harness
// -----------------------------------------------------------------------------

Fp4PvAccuracyResult bench_fp4_pv_accuracy(int n_rows, int K, int head_dim,
                                          unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> v_dist(0.0f, 1.0f);

    // V matrix [K × head_dim] — column-major (per-K-group quantisation runs
    // down each output column independently, matching the PV MMA layout).
    std::vector<float> V(static_cast<size_t>(K) * head_dim);
    for (auto& x : V) x = v_dist(rng);

    // Pre-quantise V columns (V is loaded once per attention block — quant
    // cost amortises across many P rows).
    std::vector<std::vector<uint8_t>> V_fp4(head_dim);
    std::vector<std::vector<uint8_t>> V_scales(head_dim);
    std::vector<std::vector<float>> V_lossy(head_dim);
    for (int n = 0; n < head_dim; ++n) {
        std::vector<float> col(static_cast<size_t>(K));
        for (int k = 0; k < K; ++k)
            col[k] = V[static_cast<size_t>(k) * head_dim + n];
        quantise_row_fp4(col.data(), K, V_fp4[n], V_scales[n]);
        dequantise_row_fp4(V_fp4[n], V_scales[n], K, V_lossy[n]);
    }

    std::vector<float> abs_errs;
    std::vector<float> rel_errs;
    abs_errs.reserve(static_cast<size_t>(n_rows) * head_dim);
    rel_errs.reserve(static_cast<size_t>(n_rows) * head_dim);
    int catastrophic = 0;

    for (int r = 0; r < n_rows; ++r) {
        std::vector<float> P(static_cast<size_t>(K));
        generate_postsoftmax_row(P.data(), K, rng);

        std::vector<uint8_t> P_fp4, P_scales;
        quantise_row_fp4(P.data(), K, P_fp4, P_scales);
        std::vector<float> P_lossy;
        dequantise_row_fp4(P_fp4, P_scales, K, P_lossy);

        for (int n = 0; n < head_dim; ++n) {
            // FP32 reference: P (exact) @ V (exact)
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += static_cast<double>(P[k]) *
                       static_cast<double>(V[static_cast<size_t>(k) * head_dim + n]);
            // FP4 reconstruction: P_lossy @ V_lossy (same numerical result
            // as the actual MMA modulo accumulator rounding mode).
            double fp4 = 0.0;
            for (int k = 0; k < K; ++k)
                fp4 += static_cast<double>(P_lossy[k]) * static_cast<double>(V_lossy[n][k]);
            float abs_e = static_cast<float>(std::fabs(fp4 - ref));
            float rel_e = abs_e / static_cast<float>(std::fabs(ref) + 1e-9);
            abs_errs.push_back(abs_e);
            rel_errs.push_back(rel_e);
            if (rel_e > 0.5f) ++catastrophic;
        }
    }

    auto pct = [](std::vector<float>& v, double p) -> float {
        size_t idx = std::min(v.size() - 1, static_cast<size_t>(v.size() * p));
        std::nth_element(v.begin(), v.begin() + idx, v.end());
        return v[idx];
    };

    Fp4PvAccuracyResult r{};
    r.n_rows = n_rows;
    r.K = K;
    r.head_dim = head_dim;
    r.abs_err_median = pct(abs_errs, 0.5);
    r.abs_err_p99   = pct(abs_errs, 0.99);
    r.abs_err_max   = pct(abs_errs, 0.999999);
    r.rel_err_median = pct(rel_errs, 0.5);
    r.rel_err_p90   = pct(rel_errs, 0.90);
    r.rel_err_p99   = pct(rel_errs, 0.99);
    r.rel_err_max   = pct(rel_errs, 0.999999);
    r.frac_rel_err_above_50pct =
        static_cast<float>(catastrophic) / static_cast<float>(rel_errs.size());
    return r;
}

// -----------------------------------------------------------------------------
// Phase 3a throughput harness (raw MMA loop, HMMA reference vs blockscale)
// -----------------------------------------------------------------------------
//
// Same warps × iterations × tight-loop pattern as
// `bench/mxf4nvf4_mma_bench.cu`. The HMMA reference is `mma.sync.m16n8k16`
// FP16-in/FP32-out — the same instruction the WMMA m16n16k16 wrapper
// decomposes into. Two MMA tiles for n=16 are skipped (single n=8 issue
// is sufficient for relative throughput).

__global__ void bench_hmma_m16n8k16_pv_kernel(int iterations, float* sink) {
    uint32_t a0 = threadIdx.x * 37u + 1u;
    uint32_t a1 = threadIdx.x * 41u + 2u;
    uint32_t a2 = threadIdx.x * 43u + 3u;
    uint32_t a3 = threadIdx.x * 47u + 4u;
    uint32_t b0 = threadIdx.x * 53u + 5u;
    uint32_t b1 = threadIdx.x * 59u + 6u;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#if __CUDA_ARCH__ >= 800
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3));
    }
#endif
    if (threadIdx.x == 0 && blockIdx.x == 0) sink[0] = d0 + d1 + d2 + d3;
}

__global__ void bench_mxf4nvf4_m16n8k64_pv_kernel(int iterations, float* sink) {
    uint32_t a0 = threadIdx.x * 37u + 1u;
    uint32_t a1 = threadIdx.x * 41u + 2u;
    uint32_t a2 = threadIdx.x * 43u + 3u;
    uint32_t a3 = threadIdx.x * 47u + 4u;
    uint32_t b0 = threadIdx.x * 53u + 5u;
    uint32_t b1 = threadIdx.x * 59u + 6u;
    uint32_t sfa = 0x38383838u;  // UE4M3 ~ 1.0
    uint32_t sfb = 0x38383838u;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    constexpr uint16_t tidA = 0, bidA = 0, bidB = 0, tidB0 = 0;
#if __CUDA_ARCH__ >= 1200
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32."
            "ue4m3 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13},"
            "{%14},"
            "{%15, %16},"
            "{%17},"
            "{%18, %19};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa), "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB0));
    }
#endif
    if (threadIdx.x == 0 && blockIdx.x == 0) sink[0] = d0 + d1 + d2 + d3;
}

static float run_kernel_bench(void (*kernel)(int, float*), int warps, int iterations,
                              cudaStream_t stream) {
    float* d_sink = nullptr;
    if (cudaMalloc(&d_sink, sizeof(float)) != cudaSuccess) return -1.0f;
    kernel<<<warps, 32, 0, stream>>>(iterations / 10, d_sink);
    cudaStreamSynchronize(stream);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    constexpr int NUM_REPS = 5;
    float total = 0.0f;
    for (int r = 0; r < NUM_REPS; ++r) {
        cudaEventRecord(a, stream);
        kernel<<<warps, 32, 0, stream>>>(iterations, d_sink);
        cudaEventRecord(b, stream);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        total += ms;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaFree(d_sink);
    return total / NUM_REPS;
}

Fp4PvThroughputResult bench_fp4_pv_throughput(int warps, int iterations,
                                              cudaStream_t stream) {
    Fp4PvThroughputResult r{};
    r.hmma_ms = run_kernel_bench(bench_hmma_m16n8k16_pv_kernel, warps, iterations, stream);
    r.blockscale_ms =
        run_kernel_bench(bench_mxf4nvf4_m16n8k64_pv_kernel, warps, iterations, stream);
    // Ops per MMA (FMA = 2 ops):
    //   HMMA m16n8k16:        16*8*16*2 =  4096
    //   blockscale m16n8k64:  16*8*64*2 = 16384
    constexpr double kHmmaOps = 16.0 * 8.0 * 16.0 * 2.0;
    constexpr double kBlockOps = 16.0 * 8.0 * 64.0 * 2.0;
    const double total = static_cast<double>(warps) * iterations;
    r.hmma_tops = (kHmmaOps * total) / (r.hmma_ms * 1e-3) / 1e12;
    r.blockscale_tops = (kBlockOps * total) / (r.blockscale_ms * 1e-3) / 1e12;
    r.speedup = r.blockscale_tops / r.hmma_tops;
    return r;
}

}  // namespace imp
