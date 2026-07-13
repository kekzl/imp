// test(P2.7): gpt-oss attention-sink correctness — kernel vs fp64 reference.
//
// gpt-oss (#547) adds a per-head LEARNED attention sink: a virtual extra
// softmax column whose logit `sink[h]` joins the row max and the denominator
// but contributes NO value to the output. The probabilities over the real
// keys therefore sum to < 1 (mass exp(sink-max)/denom is "parked" on the
// sink and discarded). This is distinct from StreamingLLM *sink tokens*
// (the `n_sinks` KV-slot range), which is exercised separately below.
//
// Coverage here:
//   1. attention_cublas_prefill(..., sinks) end-to-end vs an fp64 reference
//      that implements the exact sink semantics (sink in normalizer, not in
//      the V-weighted sum). Sink=-inf (nullptr-equivalent) must reproduce
//      plain softmax; large sink must visibly shrink the output norm.
//   2. StreamingLLM sink-slot eviction geometry (compute_context_range /
//      block_token_range) via a device probe — the two-range split and the
//      gap-skip the paged decode loop relies on.
//
// Tolerance: f16-score-chain attention class per tests/refs/README.md —
// cuBLAS materializes QK^T in FP32 (use_fp32_s) then softmax+downcast to f16
// P, PV in f16. Bound: 1e-2 vs fp64 (inputs are f16-rounded), measured as
// |got-ref| normalized by the RMS magnitude of the reference output (NOT a
// per-element floor: diffuse-softmax outputs cluster near zero, where a
// per-element ratio turns ordinary f16 P/V rounding into spurious large
// relative errors). Measured ~4e-3..1e-2 across configs. ASSERTED.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "core/tensor.h"
#include "compute/attention_cublas.h"
#include "compute/attention_fmha_sm120.h"
#include "compute/attention_paged_common.cuh"

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace imp {
namespace {

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err); \
    } while (0)

// LCG fill identical to tests/refs/gen_attention_crosspath_golden.py: f32
// multiply-only transforms then round to f16, so the reference is computed
// from the exact bits the GPU sees. Heavy-tailed (cubed) to mimic QK-normed
// activations.
std::vector<half> lcg_fill_f16(uint32_t seed, size_t n, float amp) {
    std::vector<half> out(n);
    uint32_t x = seed;
    const float inv = 1.0f / 8192.0f;
    for (size_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        int32_t v = static_cast<int32_t>((x >> 8) & 0x3FFFu) - 8192;
        float f = static_cast<float>(v) * inv;
        float val = f * f * f * amp;
        if ((x & 0xFFu) == 0u)
            val *= 2.0f;
        out[i] = __float2half(val);
    }
    return out;
}

// fp64 attention reference with gpt-oss sink semantics, computed from the
// f16-rounded inputs. sink_logit per head; pass -INFINITY for "no sink".
// Returns O as fp64 [q_len, n_heads*head_dim].
std::vector<double> ref_attention_sink(const std::vector<half>& Qh, const std::vector<half>& Kh,
                                       const std::vector<half>& Vh, const std::vector<double>& sink_logit,
                                       int q_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                                       bool causal, int sliding_window) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    const int gqa = n_heads / n_kv_heads;
    std::vector<double> O(static_cast<size_t>(q_len) * n_heads * head_dim, 0.0);

    for (int h = 0; h < n_heads; h++) {
        int kvh = h / gqa;
        for (int i = 0; i < q_len; i++) {
            // scores
            std::vector<double> s(kv_len);
            double mx = sink_logit[h];  // sink joins the max
            for (int j = 0; j < kv_len; j++) {
                bool masked = (causal && j > i) ||
                              (sliding_window > 0 && (i - j) >= sliding_window);
                if (masked) {
                    s[j] = -INFINITY;
                    continue;
                }
                double dot = 0.0;
                for (int d = 0; d < head_dim; d++) {
                    double q = __half2float(Qh[(static_cast<size_t>(i) * n_heads + h) * head_dim + d]);
                    double k = __half2float(Kh[(static_cast<size_t>(j) * n_kv_heads + kvh) * head_dim + d]);
                    dot += q * k;
                }
                s[j] = dot * scale;
                mx = std::max(mx, s[j]);
            }
            // denominator includes the sink term
            double denom = std::exp(sink_logit[h] - mx);
            for (int j = 0; j < kv_len; j++)
                if (std::isfinite(s[j]))
                    denom += std::exp(s[j] - mx);
            // V-weighted sum — sink contributes NO value
            for (int j = 0; j < kv_len; j++) {
                if (!std::isfinite(s[j]))
                    continue;
                double p = std::exp(s[j] - mx) / denom;
                for (int d = 0; d < head_dim; d++) {
                    double vv = __half2float(Vh[(static_cast<size_t>(j) * n_kv_heads + kvh) * head_dim + d]);
                    O[(static_cast<size_t>(i) * n_heads + h) * head_dim + d] += p * vv;
                }
            }
        }
    }
    return O;
}

struct AttnDevBufs {
    half *Q = nullptr, *K = nullptr, *V = nullptr, *O = nullptr, *S = nullptr, *sinks = nullptr;
    void free() {
        for (void* p : {(void*)Q, (void*)K, (void*)V, (void*)O, (void*)S, (void*)sinks})
            if (p) cudaFree(p);
    }
};

// Run the real cuBLAS prefill with optional per-head sink logits.
// sink_logit empty -> nullptr (plain softmax). Returns O as fp64.
std::vector<double> run_kernel(const std::vector<half>& Qh, const std::vector<half>& Kh,
                               const std::vector<half>& Vh, const std::vector<double>& sink_logit,
                               int q_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                               bool causal, int sliding_window) {
    AttnDevBufs b;
    const size_t qn = static_cast<size_t>(q_len) * n_heads * head_dim;
    const size_t kn = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t sn = static_cast<size_t>(n_heads) * q_len * kv_len;

    EXPECT_EQ(cudaMalloc(&b.Q, qn * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&b.K, kn * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&b.V, kn * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&b.O, qn * sizeof(half)), cudaSuccess);
    // S workspace sized for FP32 (so use_fp32_s picks the fused FP32 softmax,
    // matching the production gpt-oss path).
    EXPECT_EQ(cudaMalloc(&b.S, sn * 2 * sizeof(half)), cudaSuccess);
    cudaMemcpy(b.Q, Qh.data(), qn * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b.K, Kh.data(), kn * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b.V, Vh.data(), kn * sizeof(half), cudaMemcpyHostToDevice);

    void* sinks_ptr = nullptr;
    if (!sink_logit.empty()) {
        std::vector<half> sh(n_heads);
        for (int h = 0; h < n_heads; h++) sh[h] = __float2half(static_cast<float>(sink_logit[h]));
        EXPECT_EQ(cudaMalloc(&b.sinks, n_heads * sizeof(half)), cudaSuccess);
        cudaMemcpy(b.sinks, sh.data(), n_heads * sizeof(half), cudaMemcpyHostToDevice);
        sinks_ptr = b.sinks;
    }

    int64_t qshape[2] = {q_len, static_cast<int64_t>(n_heads) * head_dim};
    int64_t kshape[2] = {kv_len, static_cast<int64_t>(n_kv_heads) * head_dim};
    int64_t sshape[3] = {n_heads, q_len, kv_len};
    Tensor Q(b.Q, QType::F16, 2, qshape, true);
    Tensor K(b.K, QType::F16, 2, kshape, true);
    Tensor V(b.V, QType::F16, 2, kshape, true);
    Tensor O(b.O, QType::F16, 2, qshape, true);
    Tensor S(b.S, QType::F16, 3, sshape, true);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    attention_cublas_prefill(Q, K, V, O, S, n_heads, n_kv_heads, head_dim, scale, causal,
                             /*softcap=*/0.0f, /*q_offset=*/0, /*stream=*/nullptr, sliding_window,
                             sinks_ptr);
    cudaDeviceSynchronize();

    std::vector<half> Oh(qn);
    cudaMemcpy(Oh.data(), b.O, qn * sizeof(half), cudaMemcpyDeviceToHost);
    b.free();

    std::vector<double> Od(qn);
    for (size_t i = 0; i < qn; i++) Od[i] = __half2float(Oh[i]);
    return Od;
}

// Run the WMMA FMHA prefill (#992 sink support) with optional per-head sink
// logits. Same memory layout as run_kernel ([seq, nh, hd] row-major); the
// FMHA entry point takes 4D [batch, seq, nh, hd] views of the same buffers.
std::vector<double> run_kernel_fmha(const std::vector<half>& Qh, const std::vector<half>& Kh,
                                    const std::vector<half>& Vh, const std::vector<double>& sink_logit,
                                    int q_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                                    bool causal, int sliding_window) {
    AttnDevBufs b;
    const size_t qn = static_cast<size_t>(q_len) * n_heads * head_dim;
    const size_t kn = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

    EXPECT_EQ(cudaMalloc(&b.Q, qn * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&b.K, kn * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&b.V, kn * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&b.O, qn * sizeof(half)), cudaSuccess);
    cudaMemcpy(b.Q, Qh.data(), qn * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b.K, Kh.data(), kn * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b.V, Vh.data(), kn * sizeof(half), cudaMemcpyHostToDevice);

    const half* sinks_ptr = nullptr;
    if (!sink_logit.empty()) {
        std::vector<half> sh(n_heads);
        for (int h = 0; h < n_heads; h++) sh[h] = __float2half(static_cast<float>(sink_logit[h]));
        EXPECT_EQ(cudaMalloc(&b.sinks, n_heads * sizeof(half)), cudaSuccess);
        cudaMemcpy(b.sinks, sh.data(), n_heads * sizeof(half), cudaMemcpyHostToDevice);
        sinks_ptr = b.sinks;
    }

    int64_t qshape[4] = {1, q_len, n_heads, head_dim};
    int64_t kshape[4] = {1, kv_len, n_kv_heads, head_dim};
    Tensor Q(b.Q, QType::F16, 4, qshape, true);
    Tensor K(b.K, QType::F16, 4, kshape, true);
    Tensor V(b.V, QType::F16, 4, kshape, true);
    Tensor O(b.O, QType::F16, 4, qshape, true);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    bool ok = fmha_sm120_prefill(Q, K, V, O, scale, causal, /*sliding_window=*/sliding_window,
                                 /*softcap=*/0.0f, /*stream=*/nullptr, /*q_offset=*/0, sinks_ptr);
    EXPECT_TRUE(ok) << "fmha_sm120_prefill declined the config";
    cudaDeviceSynchronize();

    std::vector<half> Oh(qn);
    cudaMemcpy(Oh.data(), b.O, qn * sizeof(half), cudaMemcpyDeviceToHost);
    b.free();

    std::vector<double> Od(qn);
    for (size_t i = 0; i < qn; i++) Od[i] = __half2float(Oh[i]);
    return Od;
}

double max_rel_err(const std::vector<double>& got, const std::vector<double>& ref) {
    // Normalize by the RMS magnitude of the reference output, not per-element
    // |ref[i]|. Diffuse-softmax attention outputs are V-averages clustered
    // near zero; a per-element floor turns ordinary f16 P/V rounding on a
    // ~0 element into a spurious "large" relative error. RMS-normalized abs
    // error is the f16-score-chain class metric (matches the crosspath suite
    // intent: error relative to the signal scale, not to a near-zero element).
    double sumsq = 0.0;
    for (double v : ref) sumsq += v * v;
    double rms = std::sqrt(sumsq / std::max<size_t>(1, ref.size()));
    double scale = std::max(1e-3, rms);
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); i++)
        worst = std::max(worst, std::abs(got[i] - ref[i]) / scale);
    return worst;
}

// (name, q_len, kv_len, n_heads, n_kv_heads, head_dim, causal, sliding_window)
// gpt-oss-20b: head_dim=64, n_heads=64, n_kv_heads=8, sliding_window=128 on
// half the layers. We use small GQA shapes (sink semantics are head_dim- and
// count-agnostic) so the test runs everywhere and stays bounded.
struct SinkCfg {
    const char* name;
    int q_len, kv_len, n_heads, n_kv_heads, head_dim, sliding_window;
    bool causal;
};

const SinkCfg kCfgs[] = {
    {"gptoss_gqa_full", 48, 48, 16, 4, 64, 0, true},
    {"gptoss_gqa_swa", 160, 160, 8, 2, 64, 128, true},
};

constexpr double kRelTol = 1e-2;  // f16-score-chain class (tests/refs/README.md)

TEST(GptOssSinkRef, NoSinkMatchesPlainSoftmax) {
    // sinks=nullptr must reproduce ordinary softmax attention exactly (the
    // shift code is guarded on `sinks != nullptr`). Guards against a sink
    // term leaking into the no-sink path.
    for (const auto& c : kCfgs) {
        auto Q = lcg_fill_f16(0x1001u, static_cast<size_t>(c.q_len) * c.n_heads * c.head_dim, 2.0f);
        auto K = lcg_fill_f16(0x2002u, static_cast<size_t>(c.kv_len) * c.n_kv_heads * c.head_dim, 2.0f);
        auto V = lcg_fill_f16(0x3003u, static_cast<size_t>(c.kv_len) * c.n_kv_heads * c.head_dim, 2.0f);

        std::vector<double> no_sink(c.n_heads, -INFINITY);  // reference w/o sink
        auto ref = ref_attention_sink(Q, K, V, no_sink, c.q_len, c.kv_len, c.n_heads, c.n_kv_heads,
                                       c.head_dim, c.causal, c.sliding_window);
        auto got = run_kernel(Q, K, V, /*sink_logit=*/{}, c.q_len, c.kv_len, c.n_heads, c.n_kv_heads,
                              c.head_dim, c.causal, c.sliding_window);
        double e = max_rel_err(got, ref);
        EXPECT_LT(e, kRelTol) << c.name << " no-sink rel err " << e;
        printf("[sink] %-18s no-sink   rel=%.2e\n", c.name, e);
    }
}

TEST(GptOssSinkRef, SinkLogitShiftMatchesReference) {
    // Per-head learned sink logits (heavy: spans the regime where the sink
    // takes meaningful softmax mass). The kernel's exp(sink-max)/denom term
    // must match the fp64 reference; the V-weighted sum must NOT include the
    // sink. A wrong-sign or missing-denominator-term bug fails here.
    for (const auto& c : kCfgs) {
        auto Q = lcg_fill_f16(0x4001u, static_cast<size_t>(c.q_len) * c.n_heads * c.head_dim, 2.0f);
        auto K = lcg_fill_f16(0x5002u, static_cast<size_t>(c.kv_len) * c.n_kv_heads * c.head_dim, 2.0f);
        auto V = lcg_fill_f16(0x6003u, static_cast<size_t>(c.kv_len) * c.n_kv_heads * c.head_dim, 2.0f);

        std::vector<double> sink(c.n_heads);
        for (int h = 0; h < c.n_heads; h++)
            sink[h] = -1.5 + 0.5 * (h % 7);  // span [-1.5, +1.5]: sink mass 4%..70%

        auto ref = ref_attention_sink(Q, K, V, sink, c.q_len, c.kv_len, c.n_heads, c.n_kv_heads,
                                       c.head_dim, c.causal, c.sliding_window);
        auto got = run_kernel(Q, K, V, sink, c.q_len, c.kv_len, c.n_heads, c.n_kv_heads, c.head_dim,
                              c.causal, c.sliding_window);
        double e = max_rel_err(got, ref);
        EXPECT_LT(e, kRelTol) << c.name << " sink rel err " << e;
        printf("[sink] %-18s with-sink rel=%.2e\n", c.name, e);
    }
}

TEST(GptOssSinkRef, FmhaSinkMatchesReference) {
    // #992: the WMMA FMHA folds the sink into its online-softmax init
    // (m = sink, l = 1). Same fp64 reference and tolerance class as the
    // cuBLAS test above — full-attention AND sliding-window configs (gpt-oss
    // alternates SWA=128 layers, so the SWA×sink interaction is load-bearing).
    for (const auto& c : kCfgs) {
        auto Q = lcg_fill_f16(0x4001u, static_cast<size_t>(c.q_len) * c.n_heads * c.head_dim, 2.0f);
        auto K = lcg_fill_f16(0x5002u, static_cast<size_t>(c.kv_len) * c.n_kv_heads * c.head_dim, 2.0f);
        auto V = lcg_fill_f16(0x6003u, static_cast<size_t>(c.kv_len) * c.n_kv_heads * c.head_dim, 2.0f);

        std::vector<double> sink(c.n_heads);
        for (int h = 0; h < c.n_heads; h++)
            sink[h] = -1.5 + 0.5 * (h % 7);

        auto ref = ref_attention_sink(Q, K, V, sink, c.q_len, c.kv_len, c.n_heads, c.n_kv_heads,
                                       c.head_dim, c.causal, c.sliding_window);
        auto got = run_kernel_fmha(Q, K, V, sink, c.q_len, c.kv_len, c.n_heads, c.n_kv_heads,
                                   c.head_dim, c.causal, c.sliding_window);
        double e = max_rel_err(got, ref);
        EXPECT_LT(e, kRelTol) << c.name << " FMHA sink rel err " << e;
        printf("[sink] %-18s FMHA with-sink rel=%.2e\n", c.name, e);
    }
}

TEST(GptOssSinkRef, FmhaNoSinkUnchanged) {
    // sinks=nullptr must leave the FMHA's plain-softmax path bit-compatible
    // with its pre-#992 behavior (init still -FLT_MAX/0): compare against the
    // fp64 no-sink reference.
    for (const auto& c : kCfgs) {
        auto Q = lcg_fill_f16(0x4001u, static_cast<size_t>(c.q_len) * c.n_heads * c.head_dim, 2.0f);
        auto K = lcg_fill_f16(0x5002u, static_cast<size_t>(c.kv_len) * c.n_kv_heads * c.head_dim, 2.0f);
        auto V = lcg_fill_f16(0x6003u, static_cast<size_t>(c.kv_len) * c.n_kv_heads * c.head_dim, 2.0f);
        std::vector<double> no_sink(c.n_heads, -INFINITY);
        auto ref = ref_attention_sink(Q, K, V, no_sink, c.q_len, c.kv_len, c.n_heads, c.n_kv_heads,
                                       c.head_dim, c.causal, c.sliding_window);
        auto got = run_kernel_fmha(Q, K, V, {}, c.q_len, c.kv_len, c.n_heads, c.n_kv_heads, c.head_dim,
                                   c.causal, c.sliding_window);
        double e = max_rel_err(got, ref);
        EXPECT_LT(e, kRelTol) << c.name << " FMHA no-sink rel err " << e;
        printf("[sink] %-18s FMHA no-sink rel=%.2e\n", c.name, e);
    }
}

TEST(GptOssSinkRef, LargeSinkShrinksOutputNorm) {
    // Sanity that the sink actually parks mass: a huge sink logit drives the
    // discarded fraction to ~1, so ||O|| must collapse vs the no-sink output.
    // This catches a sink that joins the max but is forgotten in the denom.
    const auto& c = kCfgs[0];
    auto Q = lcg_fill_f16(0x7001u, static_cast<size_t>(c.q_len) * c.n_heads * c.head_dim, 2.0f);
    auto K = lcg_fill_f16(0x8002u, static_cast<size_t>(c.kv_len) * c.n_kv_heads * c.head_dim, 2.0f);
    auto V = lcg_fill_f16(0x9003u, static_cast<size_t>(c.kv_len) * c.n_kv_heads * c.head_dim, 2.0f);

    auto base = run_kernel(Q, K, V, {}, c.q_len, c.kv_len, c.n_heads, c.n_kv_heads, c.head_dim, c.causal,
                           c.sliding_window);
    std::vector<double> big(c.n_heads, 30.0);  // exp dominates -> ~all mass parked
    auto shrunk = run_kernel(Q, K, V, big, c.q_len, c.kv_len, c.n_heads, c.n_kv_heads, c.head_dim,
                             c.causal, c.sliding_window);
    double nb = 0.0, ns = 0.0;
    for (size_t i = 0; i < base.size(); i++) { nb += base[i] * base[i]; ns += shrunk[i] * shrunk[i]; }
    printf("[sink] norm base=%.4f huge-sink=%.6f ratio=%.4f\n", std::sqrt(nb), std::sqrt(ns),
           std::sqrt(ns / std::max(1e-30, nb)));
    EXPECT_LT(std::sqrt(ns), 0.05 * std::sqrt(nb)) << "huge sink should collapse output norm";
}

// ---------------------------------------------------------------------------
// StreamingLLM sink-SLOT eviction geometry (the paged-path `n_sinks` range).
// compute_context_range/block_token_range are __device__ only; probe them via
// a tiny kernel that records the attended token set, then check eviction
// semantics on the host.
// ---------------------------------------------------------------------------
__global__ void probe_streaming_mask(int ctx_len, int block_size, int sliding_window, int n_sinks,
                                      uint8_t* attended /*[ctx_len]*/, int* out_streaming) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    ContextRange r = compute_context_range(ctx_len, block_size, sliding_window, n_sinks);
    *out_streaming = streaming_active(r) ? 1 : 0;
    for (int t = 0; t < ctx_len; t++)
        attended[t] = 0;
    int blk = r.first_block;
    while (blk < r.num_ctx_blocks) {
        int first_tok, last_tok;
        if (block_token_range(r, blk, block_size, ctx_len, first_tok, last_tok)) {
            for (int t = first_tok; t < last_tok; t++) {
                int abs_t = blk * block_size + t;
                if (abs_t >= 0 && abs_t < ctx_len)
                    attended[abs_t] = 1;
            }
        }
        blk = next_valid_block(r, blk);
    }
}

TEST(GptOssSinkRef, StreamingSinkSlotEviction) {
    // StreamingLLM: tokens [0, n_sinks) (the sink slots) plus the trailing
    // sliding window are attended; the middle is EVICTED (never loaded). This
    // is the geometry the paged decode loop relies on.
    const int ctx_len = 100, block_size = 16, sliding_window = 32, n_sinks = 4;

    uint8_t* d_att = nullptr;
    int* d_stream = nullptr;
    CUDA_CHECK(cudaMalloc(&d_att, ctx_len));
    CUDA_CHECK(cudaMalloc(&d_stream, sizeof(int)));
    probe_streaming_mask<<<1, 32>>>(ctx_len, block_size, sliding_window, n_sinks, d_att, d_stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> att(ctx_len);
    int streaming = 0;
    CUDA_CHECK(cudaMemcpy(att.data(), d_att, ctx_len, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&streaming, d_stream, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_att);
    cudaFree(d_stream);

    EXPECT_EQ(streaming, 1) << "expected StreamingLLM two-range mode active";

    const int window_start = ctx_len - sliding_window;  // 68
    for (int t = 0; t < ctx_len; t++) {
        bool expect = (t < n_sinks) || (t >= window_start);
        EXPECT_EQ(att[t] != 0, expect) << "token " << t << " attended=" << (int)att[t];
    }
    // Middle region [n_sinks, window_start) must be fully evicted.
    int middle = 0;
    for (int t = n_sinks; t < window_start; t++) middle += att[t];
    EXPECT_EQ(middle, 0) << "evicted middle region attended " << middle << " tokens";
    printf("[sink] streaming eviction: sinks[0,%d) + window[%d,%d), %d middle tokens evicted\n", n_sinks,
           window_start, ctx_len, window_start - n_sinks);
}

}  // namespace
}  // namespace imp
