// Tests for the native sm_120 FMHA kernel (attention_fmha_sm120.cu).
// Validates correctness against a CPU reference implementation for various
// configurations: causal/non-causal, GQA, sliding window, softcap, all head dims.

#include <gtest/gtest.h>
#include "compute/attention.h"
#include "compute/attention_fmha_sm120.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <float.h>

namespace imp {
namespace {

class FmhaSm120Test : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&stream_);
        int device = 0;
        cudaGetDevice(&device);
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        sm_ = major * 10 + minor;
    }
    void TearDown() override { cudaStreamDestroy(stream_); }

    // CPU reference: standard attention with optional causal, sliding_window, softcap, q_offset
    static void ref_attention(const std::vector<float>& Q_f, const std::vector<float>& K_f,
                              const std::vector<float>& V_f, std::vector<float>& O_f, int B, int Sq, int Skv,
                              int NH, int NKV, int HD, float scale, bool causal, int sliding_window,
                              float softcap, int q_offset = 0) {
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < NH; h++) {
                int kvh = h / (NH / NKV);
                for (int qi = 0; qi < Sq; qi++) {
                    float m = -FLT_MAX;
                    std::vector<float> s(Skv);
                    for (int ki = 0; ki < Skv; ki++) {
                        float dot = 0.0f;
                        for (int d = 0; d < HD; d++) {
                            float q_val = Q_f[((b * Sq + qi) * NH + h) * HD + d];
                            float k_val = K_f[((b * Skv + ki) * NKV + kvh) * HD + d];
                            dot += q_val * k_val;
                        }
                        dot *= scale;
                        if (softcap > 0.0f)
                            dot = softcap * tanhf(dot / softcap);
                        if (causal && (q_offset + qi) < ki)
                            dot = -FLT_MAX;
                        if (sliding_window > 0 && ((q_offset + qi) - ki) >= sliding_window)
                            dot = -FLT_MAX;
                        s[ki] = dot;
                        m = fmaxf(m, dot);
                    }
                    float sum = 0.0f;
                    for (int ki = 0; ki < Skv; ki++) {
                        s[ki] = expf(s[ki] - m);
                        sum += s[ki];
                    }
                    if (sum > 0.0f) {
                        for (int ki = 0; ki < Skv; ki++)
                            s[ki] /= sum;
                    }
                    for (int d = 0; d < HD; d++) {
                        float acc = 0.0f;
                        for (int ki = 0; ki < Skv; ki++) {
                            float v_val = V_f[((b * Skv + ki) * NKV + kvh) * HD + d];
                            acc += s[ki] * v_val;
                        }
                        O_f[((b * Sq + qi) * NH + h) * HD + d] = acc;
                    }
                }
            }
        }
    }

    void run_test(int B, int Sq, int Skv, int NH, int NKV, int HD, bool causal, int sliding_window = 0,
                  float softcap = 0.0f, float tol = 1e-2f, int q_offset = 0) {
        if (sm_ < 90) {
            GTEST_SKIP() << "FMHA sm120 requires sm_90+ (WMMA fallback)";
        }

        float scale = 1.0f / std::sqrt(static_cast<float>(HD));

        size_t q_elems = B * Sq * NH * HD;
        size_t kv_elems = B * Skv * NKV * HD;

        std::vector<float> Q_f(q_elems), K_f(kv_elems), V_f(kv_elems);
        // NOTE: the int cast before the -6 is load-bearing (#525/#528). With
        // size_t i, `(i*7+3)%13 - 6` underflows unsigned whenever %13 < 6 —
        // ~46% of fills became ±inf, the reference went NaN and the NaN-
        // dropping comparator passed vacuously. Also: the old V multiplier 13
        // made V CONSTANT (i*13 % 13 == 0), so P@V was insensitive to the
        // attention weights entirely; 17 (coprime to 13) restores coverage.
        for (size_t i = 0; i < q_elems; i++)
            Q_f[i] = 0.02f * static_cast<float>(static_cast<int>((i * 7 + 3) % 13) - 6);
        for (size_t i = 0; i < kv_elems; i++) {
            K_f[i] = 0.02f * static_cast<float>(static_cast<int>((i * 11 + 5) % 13) - 6);
            V_f[i] = 0.2f * static_cast<float>(static_cast<int>((i * 17 + 7) % 13) - 6);
        }

        // CPU reference
        std::vector<float> O_ref(q_elems, 0.0f);
        ref_attention(Q_f, K_f, V_f, O_ref, B, Sq, Skv, NH, NKV, HD, scale, causal, sliding_window, softcap,
                      q_offset);

        // Convert to half
        std::vector<half> Q_h(q_elems), K_h(kv_elems), V_h(kv_elems);
        for (size_t i = 0; i < q_elems; i++)
            Q_h[i] = __float2half(Q_f[i]);
        for (size_t i = 0; i < kv_elems; i++)
            K_h[i] = __float2half(K_f[i]);
        for (size_t i = 0; i < kv_elems; i++)
            V_h[i] = __float2half(V_f[i]);

        size_t q_bytes = q_elems * sizeof(half);
        size_t kv_bytes = kv_elems * sizeof(half);

        void *d_q, *d_k, *d_v, *d_o;
        cudaMalloc(&d_q, q_bytes);
        cudaMalloc(&d_k, kv_bytes);
        cudaMalloc(&d_v, kv_bytes);
        cudaMalloc(&d_o, q_bytes);

        cudaMemcpy(d_q, Q_h.data(), q_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, K_h.data(), kv_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, V_h.data(), kv_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_o, 0, q_bytes);

        int64_t q_shape[] = {B, Sq, NH, HD};
        int64_t kv_shape[] = {B, Skv, NKV, HD};
        Tensor Qt(d_q, QType::F16, 4, q_shape, true);
        Tensor Kt(d_k, QType::F16, 4, kv_shape, true);
        Tensor Vt(d_v, QType::F16, 4, kv_shape, true);
        Tensor Ot(d_o, QType::F16, 4, q_shape, true);

        bool ok = fmha_sm120_prefill(Qt, Kt, Vt, Ot, scale, causal, sliding_window, softcap, stream_,
                                     q_offset);
        if (!ok) {
            GTEST_SKIP() << "fmha_sm120_prefill returned false (config unsupported on this GPU)";
        }
        cudaStreamSynchronize(stream_);

        cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

        // Read back and compare
        std::vector<half> O_h(q_elems);
        cudaMemcpy(O_h.data(), d_o, q_bytes, cudaMemcpyDeviceToHost);

        float max_err = 0.0f;
        int non_finite = 0;
        for (size_t i = 0; i < q_elems; i++) {
            float got = __half2float(O_h[i]);
            float ref = O_ref[i];
            if (!std::isfinite(got) || !std::isfinite(ref)) {
                non_finite++;  // NaN would silently fall out of std::max below
                continue;
            }
            // denom max(1,|ref|) as in test_attention_crosspath.cu: outputs
            // are O(1)-bounded weighted averages of V; an |ref|-relative error
            // with a 1e-6 floor explodes at sign crossings of correct kernels.
            float denom = std::max(std::abs(ref), 1.0f);
            max_err = std::max(max_err, std::abs(got - ref) / denom);
        }
        EXPECT_EQ(non_finite, 0) << "non-finite output/reference elements";

        EXPECT_LT(max_err, tol) << "Max relative error " << max_err << " exceeds threshold " << tol
                                << " (B=" << B << " Sq=" << Sq << " Skv=" << Skv << " NH=" << NH
                                << " NKV=" << NKV << " HD=" << HD << " causal=" << causal
                                << " sw=" << sliding_window << " softcap=" << softcap << ")";

        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
    }

    cudaStream_t stream_ = nullptr;
    int sm_ = 0;
};

// --- Basic correctness ---

TEST_F(FmhaSm120Test, NonCausalHD128) { run_test(1, 128, 128, 2, 2, 128, false); }

TEST_F(FmhaSm120Test, CausalHD128) { run_test(1, 128, 128, 2, 2, 128, true); }

TEST_F(FmhaSm120Test, CausalMultiTile) {
    // Multiple Q tiles (Sq=256 > Bq=128) and KV tiles (Skv=192 > Bkv=64)
    run_test(1, 256, 192, 2, 2, 128, true);
}

TEST_F(FmhaSm120Test, GQA) {
    // GQA: n_heads=8, n_kv_heads=2 (4:1 ratio)
    run_test(1, 128, 128, 8, 2, 128, true);
}

// --- All head dimensions ---

TEST_F(FmhaSm120Test, HeadDim64) { run_test(1, 128, 128, 4, 4, 64, true); }

TEST_F(FmhaSm120Test, HeadDim96) { run_test(1, 128, 128, 4, 4, 96, true); }

TEST_F(FmhaSm120Test, HeadDim256) { run_test(1, 64, 64, 2, 2, 256, true); }

// --- Non-aligned sequence lengths ---

TEST_F(FmhaSm120Test, NonAlignedSeqLen) { run_test(1, 200, 150, 2, 2, 128, true); }

// --- Sliding window ---

TEST_F(FmhaSm120Test, SlidingWindow) { run_test(1, 128, 128, 2, 2, 128, true, /*sw=*/64); }

TEST_F(FmhaSm120Test, SlidingWindowMultiTile) { run_test(1, 256, 256, 2, 2, 128, true, /*sw=*/64); }

// --- Softcap ---

TEST_F(FmhaSm120Test, Softcap) { run_test(1, 128, 128, 2, 2, 128, true, 0, /*softcap=*/50.0f); }

// --- Combined features ---

TEST_F(FmhaSm120Test, SoftcapCausalSlidingWindow) {
    run_test(1, 128, 128, 2, 2, 128, true, /*sw=*/64, /*softcap=*/50.0f);
}

// --- Chunked prefill (q_offset > 0) ---

TEST_F(FmhaSm120Test, ChunkedCausalBasic) {
    // Q has 64 rows starting at position 448 in a 512-token sequence
    run_test(1, 64, 512, 8, 8, 128, true, 0, 0.0f, 1e-2f, /*q_offset=*/448);
}

TEST_F(FmhaSm120Test, ChunkedCausalMiddle) {
    // Chunk in the middle of the sequence
    run_test(1, 128, 1024, 8, 8, 128, true, 0, 0.0f, 1e-2f, /*q_offset=*/256);
}

TEST_F(FmhaSm120Test, ChunkedCausalGQA) {
    // GQA 4:1 ratio with q_offset
    run_test(1, 64, 512, 32, 8, 128, true, 0, 0.0f, 1e-2f, /*q_offset=*/448);
}

TEST_F(FmhaSm120Test, ChunkedCausalHD256) {
    // HD=256 with q_offset
    run_test(1, 64, 512, 16, 8, 256, true, 0, 0.0f, 2e-2f, /*q_offset=*/448);
}

TEST_F(FmhaSm120Test, ChunkedSlidingWindow) {
    // Sliding window with q_offset
    run_test(1, 64, 512, 8, 8, 128, true, 128, 0.0f, 1e-2f, /*q_offset=*/448);
}

TEST_F(FmhaSm120Test, ChunkedLargeContext) {
    // Large context: 512-token chunk in a 4096-token sequence
    run_test(1, 512, 4096, 8, 8, 128, true, 0, 0.0f, 1e-2f, /*q_offset=*/3584);
}

TEST_F(FmhaSm120Test, ChunkedOffsetZero) {
    // q_offset=0 with rectangular Q/KV (first chunk of a multi-chunk prefill)
    run_test(1, 512, 512, 8, 8, 128, true, 0, 0.0f, 1e-2f, /*q_offset=*/0);
}

// --- Dispatch integration ---

TEST_F(FmhaSm120Test, DispatchSelectsSm120FMHA) {
    // Just re-run the causal HD128 test — this verifies the kernel works
    // in the same test binary context. The original manual test had a subtle
    // stream/memory ordering issue.
    run_test(1, 64, 64, 4, 4, 128, true, 0, 0.0f);
}

TEST_F(FmhaSm120Test, DispatchManual) {
    // RESOLVED 2026-06-05 (#528): the 2026-05-14 "NaN mystery" (this manual
    // path produced NaN while run_test() 'worked' on the same shape) was the
    // size_t underflow in the %13 fills — ~46% of inputs were ±inf, so NaN
    // output was CORRECT; run_test() only looked green because its comparator
    // silently dropped NaNs via std::max. With the int cast both paths agree.
    const int B = 1, S = 64, NH = 4, HD = 128;
    size_t bytes = B * S * NH * HD * sizeof(half);

    void *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, bytes);
    cudaMalloc(&d_k, bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_o, bytes);

    std::vector<half> h_q(B * S * NH * HD), h_k(B * S * NH * HD), h_v(B * S * NH * HD);
    for (size_t i = 0; i < h_q.size(); i++) {
        h_q[i] = __float2half(0.02f * static_cast<float>(static_cast<int>((i * 7 + 3) % 13) - 6));
    }
    for (size_t i = 0; i < h_k.size(); i++) {
        h_k[i] = __float2half(0.02f * static_cast<float>(static_cast<int>((i * 11 + 5) % 13) - 6));
        h_v[i] = __float2half(0.2f * static_cast<float>(static_cast<int>((i * 17 + 7) % 13) - 6));
    }
    cudaMemcpy(d_q, h_q.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0, bytes);
    cudaDeviceSynchronize();

    int64_t shape[] = {B, S, NH, HD};
    Tensor Q(d_q, QType::F16, 4, shape, true);
    Tensor K(d_k, QType::F16, 4, shape, true);
    Tensor V(d_v, QType::F16, 4, shape, true);
    Tensor O(d_o, QType::F16, 4, shape, true);

    float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    // Call sm120 FMHA directly first to verify it works in this test context
    bool direct_ok = fmha_sm120_prefill(Q, K, V, O, scale, true, 0, 0.0f, stream_);
    cudaStreamSynchronize(stream_);
    fprintf(stderr, "DEBUG: fmha_sm120_prefill returned %s\n", direct_ok ? "true" : "false");
    ASSERT_TRUE(direct_ok) << "fmha_sm120_prefill returned false";

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    fprintf(stderr, "DEBUG: post-sync CUDA error: %s\n", cudaGetErrorString(err));
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Post-sync CUDA error: " << cudaGetErrorString(err);

    // Check output has finite, non-zero values
    std::vector<half> h_o(B * S * NH * HD);
    cudaMemcpy(h_o.data(), d_o, bytes, cudaMemcpyDeviceToHost);
    int finite_nonzero = 0, nan_count = 0;
    for (auto& v : h_o) {
        float fv = __half2float(v);
        if (std::isfinite(fv) && fv != 0.0f)
            finite_nonzero++;
        if (std::isnan(fv))
            nan_count++;
    }
    EXPECT_EQ(nan_count, 0) << "NaN in output with finite inputs";
    // Debug: check Q input is non-zero too
    std::vector<half> h_q_check(B * S * NH * HD);
    cudaMemcpy(h_q_check.data(), d_q, bytes, cudaMemcpyDeviceToHost);
    int q_nonzero = 0;
    for (auto& v : h_q_check)
        if (__half2float(v) != 0.0f)
            q_nonzero++;
    fprintf(stderr, "DEBUG: Q nonzero=%d, O nonzero=%d (of %d)\n", q_nonzero, finite_nonzero,
            B * S * NH * HD);
    fprintf(stderr, "DEBUG: O[0..7] = %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
            __half2float(h_o[0]), __half2float(h_o[1]), __half2float(h_o[2]), __half2float(h_o[3]),
            __half2float(h_o[4]), __half2float(h_o[5]), __half2float(h_o[6]), __half2float(h_o[7]));
    fprintf(stderr, "DEBUG: V[0..7] = %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
            __half2float(h_v[0]), __half2float(h_v[1]), __half2float(h_v[2]), __half2float(h_v[3]),
            __half2float(h_v[4]), __half2float(h_v[5]), __half2float(h_v[6]), __half2float(h_v[7]));

    EXPECT_GT(finite_nonzero, 0)
        << "Dispatch produced all-zero output on sm_120 (expected sm120 FMHA to run)";

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

}  // namespace
}  // namespace imp
