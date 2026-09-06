// =============================================================================
// TEST_AUDIT (retired) Phase 2.6 — Risk #8
// Every GGUF dequant kernel, and the dp4a/MMVQ GEMVs on top of them, against
// the format-derived fp64 reference in `gguf_format_ref.h`.
//
// WHY THIS EXISTS (audit §3 risk #8, §2 class-B critique):
//   The existing tests/test_mmvq.cu compares imp's MMVQ kernel against imp's
//   dp4a kernel (imp-vs-imp == class B tautology). All of GGUF serving
//   consumes these dequant + GEMV kernels, yet only INT4/INT8 toy refs
//   existed. This file is the class-A anchor: `gguf_format_ref.h` re-derives
//   each block format's byte layout and dequant ARITHMETIC from the ggml
//   definition, and this file launches imp's kernels against it. Six formats
//   had no reference at all until AUDIT_arch_2026 D-5, and Q3_K turned out to
//   be reading the wrong high-bit plane in both of its kernels.
//
// INDEPENDENCE (audit §4 — no tautologies): stated in `gguf_format_ref.h`,
//   which owns both halves. Edge cases (d = 0, all-63 / all-0 6-bit scales,
//   max-magnitude scale halfs, a NaN d-half) come from its ScaleMode set and
//   are exercised here with a hard no-NaN/Inf guard.
//
// TOLERANCES (audit §4 tolerance policy — derived per path):
//   * Dequant kernel (pure decode, half-rounding only): the kernel computes
//     d*sc*q in fp32 then rounds to f16 once. The fp64 reference rounds the
//     same product to f16. Both see identical input bits, so the only spread
//     is fp32-vs-fp64 accumulation of a 2-3 factor product => <= 1e-3 rel
//     (1 ulp of f16 ~= 2^-11 ~= 4.9e-4; we measure and assert 1e-3).
//   * fp16-dequant GEMV (gemv_q8_0 / gemv_q6k): dequant in fp32, dot in fp32,
//     output rounded to f16. Reference: dequant in fp64, dot the ORIGINAL f16
//     x in fp64. Error = fp32-vs-fp64 dot over K terms + one f16 output round.
//     fp16-class => <= 1e-2 rel (justified per audit §4, measured).
//   * dp4a / MMVQ GEMV (gemv_*_q8_1, ggml_mmvq_*): these ALSO quantize the
//     activations x to Q8_1 (amax/127 per 32-block, ggml-standard). That adds
//     ~0.4% RMS per-element activation noise on top of the f16 dot. Over a
//     K-term dot this does NOT fully average out (correlated within a block),
//     so we derive ~1-2% and MEASURE the real envelope, asserting 2.5e-2 rel
//     with the per-test printed stats as the characterization record.
// =============================================================================

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstring>
#include <vector>

#include "compute/gemm.h"
#include "compute/ggml_mmvq.h"
#include "quant/dequant_gpu.h"
#include "gguf_format_ref.h"

namespace imp {
namespace {

using namespace gguf_ref;  // Lcg, ScaleMode, ref_dequant_*, build_*, kFormats, format_spec

// -----------------------------------------------------------------------------
// GPU helpers
// -----------------------------------------------------------------------------
void* to_device(const std::vector<uint8_t>& h) {
    void* d = nullptr;
    cudaMalloc(&d, h.size());
    cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice);
    return d;
}

half* random_x(int K, Lcg& g, std::vector<half>& host) {
    host.resize(K);
    for (int i = 0; i < K; ++i)
        host[i] = __float2half(g.unit() * 2.0f);  // ~[-2,2]
    half* d = nullptr;
    cudaMalloc(&d, K * sizeof(half));
    cudaMemcpy(d, host.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    return d;
}

bool any_nan_inf(const std::vector<half>& v) {
    for (half h : v) {
        float f = __half2float(h);
        if (std::isnan(f) || std::isinf(f))
            return true;
    }
    return false;
}


// -----------------------------------------------------------------------------
// (1) DEQUANT KERNEL vs fp64 reference. Tolerance: f16-rounding only (1e-3 rel).
// -----------------------------------------------------------------------------
void check_dequant(const char* name, QType qt, int N, int K, ScaleMode mode, double rel_tol) {
    Lcg g(0xC0FFEEu + static_cast<uint32_t>(qt) * 131 + mode * 977);
    std::vector<uint8_t> buf;
    format_spec(qt).build(buf, N, K, g, mode);

    void* dW = to_device(buf);
    half* dOut = nullptr;
    cudaMalloc(&dOut, (size_t)N * K * sizeof(half));
    dequant_gpu(dW, dOut, qt, N, K, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> hOut((size_t)N * K);
    cudaMemcpy(hOut.data(), dOut, (size_t)N * K * sizeof(half), cudaMemcpyDeviceToHost);

    std::vector<double> ref;
    ref_dequant_all(buf, N, K, qt, ref);

    // No-NaN/Inf guard (the real Gemma-class assert). NORMAL/ZERO_D weights can
    // never overflow f16, so the GPU must be all-finite. MAXMAG (d=65504 * full
    // quant range) DELIBERATELY overflows f16 — there a finite output would be
    // WRONG; the independent check is "GPU is non-finite exactly where the
    // f16-rounded fp64 reference is non-finite" (verified in the loop below).
    // NAN_D injects a NaN scale half: only assert no crash / no UB (no compare).
    if (mode == NORMAL || mode == ZERO_D) {
        ASSERT_FALSE(any_nan_inf(hOut)) << name << ": dequant produced NaN/Inf on finite weights";
    }

    double max_rel = 0.0, max_abs = 0.0;
    int worst = 0;
    int classify_mismatch = 0;
    for (size_t i = 0; i < hOut.size(); ++i) {
        double r = ref[i];
        double g16 = static_cast<double>(__half2float(hOut[i]));
        // Round the fp64 reference to f16 the same way the kernel rounds, so we
        // only measure arithmetic divergence, not the unavoidable f16 step.
        float r16f = __float2half_then_float(static_cast<float>(r));
        double r16 = static_cast<double>(r16f);
        bool g_fin = std::isfinite(g16);
        bool r_fin = std::isfinite(r16);
        if (mode == NAN_D)
            continue;  // reference NaN — characterize only, no metric/classify
        if (g_fin != r_fin) {
            ++classify_mismatch;  // GPU and ref disagree on finite/overflow
            continue;
        }
        if (!g_fin)
            continue;  // both overflowed identically (MAXMAG): correct
        double a = std::fabs(g16 - r16);
        double rel = std::fabs(r16) > 1e-4 ? a / std::fabs(r16) : a;
        if (rel > max_rel) {
            max_rel = rel;
            max_abs = a;
            worst = static_cast<int>(i);
        }
    }
    printf("[dequant %-6s mode=%d] N=%d K=%d max_rel=%.3e max_abs=%.3e classify_mismatch=%d (idx=%d "
           "gpu=%.5f ref=%.5f)\n",
           name, mode, N, K, max_rel, max_abs, classify_mismatch, worst, __half2float(hOut[worst]),
           ref[worst]);

    if (mode != NAN_D) {
        EXPECT_EQ(classify_mismatch, 0)
            << name << " mode=" << mode << ": GPU/ref disagree on f16 overflow classification";
        EXPECT_LT(max_rel, rel_tol) << name << " mode=" << mode << ": dequant rel error too large";
    }

    cudaFree(dW);
    cudaFree(dOut);
}

// -----------------------------------------------------------------------------
// fp64 reference GEMV: y[r] = sum_k dequant(W)[r,k] * (f16)x[k].
// -----------------------------------------------------------------------------
void ref_gemv(const std::vector<double>& wref, const std::vector<half>& x, int N, int K,
              std::vector<double>& y) {
    y.assign(N, 0.0);
    for (int r = 0; r < N; ++r) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k)
            acc += wref[(size_t)r * K + k] * static_cast<double>(__half2float(x[k]));
        y[r] = acc;
    }
}

struct GemvStats {
    double max_rel_scaled;  // max |gpu-ref| / rms(ref)  — cancellation-robust
    double rms_rel;         // sqrt(mean(err^2)) / rms(ref)
    double max_abs;
    double ref_rms;
    int worst;
};

// Compare GEMV output to the fp64 reference. Per-element relative error is the
// WRONG metric for a dot product: genuine sign cancellation drives some ref
// outputs to ~0, where any absolute noise explodes the per-element ratio
// (measured 5x-30x on the q8_1 path). The honest column-vector metric is the
// error normalized by the TYPICAL output magnitude rms(ref): it answers "how
// large is the noise relative to a representative logit", which is exactly what
// matters for argmax/softmax downstream.
GemvStats gemv_eval(const std::vector<half>& gpu, const std::vector<double>& ref) {
    GemvStats s{};
    double sum_sq = 0.0, sum_err_sq = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
        sum_sq += ref[i] * ref[i];
    double ref_rms = std::sqrt(sum_sq / ref.size());
    double inv = ref_rms > 1e-9 ? 1.0 / ref_rms : 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double gf = static_cast<double>(__half2float(gpu[i]));
        double a = std::fabs(gf - ref[i]);
        sum_err_sq += a * a;
        double rel = a * inv;
        if (rel > s.max_rel_scaled) {
            s.max_rel_scaled = rel;
            s.max_abs = a;
            s.worst = static_cast<int>(i);
        }
    }
    s.rms_rel = std::sqrt(sum_err_sq / ref.size()) * inv;
    s.ref_rms = ref_rms;
    return s;
}

}  // namespace

// =============================================================================
// DEQUANT-KERNEL TESTS (path: src/quant/dequant_gpu.cu)
//
// TYPED_TEST over the GGUF block formats (R8 / audit §Phase-2 R8: "parametrize
// where it's cheap and real"). The dequant test body is one shared skeleton —
// build synthetic blocks, run dequant_gpu, compare to the fp64 reference over
// the 4 scale modes — that already dispatches on QType (build_* + ref_dequant_all
// switch). The heterogeneous part is the per-format fp64 reference, which stays
// in its own function; only the launch+compare driver is parametrized. Each
// format is a compile-time tag carrying its QType + name; the body is identical.
// =============================================================================
template <QType QT>
struct QTypeTag {
    static constexpr QType value = QT;
};

template <typename T>
class GgufDequant : public ::testing::Test {};

// Every QType `dequant_gpu_supported()` accepts. Six of them (Q4_1, Q5_0,
// Q5_1, Q2_K, Q3_K, Q8_K) shipped a decode path with no numerical check at any
// level until AUDIT_arch_2026 D-5; `DequantSupportedFormatsAllHaveAReference`
// below fails if the loader ever accepts a seventh without one.
using GgufDequantFormats = ::testing::Types<
    QTypeTag<QType::Q8_0>, QTypeTag<QType::Q4_0>, QTypeTag<QType::Q4_1>, QTypeTag<QType::Q5_0>,
    QTypeTag<QType::Q5_1>, QTypeTag<QType::Q2_K>, QTypeTag<QType::Q3_K>, QTypeTag<QType::Q4_K>,
    QTypeTag<QType::Q5_K>, QTypeTag<QType::Q6_K>, QTypeTag<QType::Q8_K>, QTypeTag<QType::IQ4_NL>,
    QTypeTag<QType::IQ4_XS>>;

inline const char* qtype_name(QType qt) { return format_spec(qt).name; }

TYPED_TEST_SUITE(GgufDequant, GgufDequantFormats);

TYPED_TEST(GgufDequant, AllScaleModes) {
    constexpr QType qt = TypeParam::value;
    const char* name = qtype_name(qt);
    // K=512 covers both layouts: 16 blocks/row for the 32-block formats and
    // 2 super-blocks/row for the 256-block K-quants. N chosen non-round so a
    // row-stride bug surfaces.
    check_dequant(name, qt, 37, 512, NORMAL, 1e-3);
    check_dequant(name, qt, 16, 256, ZERO_D, 1e-3);
    check_dequant(name, qt, 16, 256, MAXMAG, 1e-3);
    check_dequant(name, qt, 8, 256, NAN_D, 1e-3);  // no-crash / UB guard
}

// Two-way: the typed suite above covers exactly the formats `dequant_gpu`
// serves. Accepting a format with no reference is how six of them shipped
// unchecked (D-5); carrying a reference for a format the loader no longer
// accepts is dead test weight. Both directions fail here.
TEST(GgufDequantCoverage, EverySupportedFormatHasAReference) {
    // Every wire-stable QType, whether or not imp dequantizes it.
    const QType kAllWireTypes[] = {QType::F32,    QType::F16,  QType::Q4_0, QType::Q4_1, QType::Q5_0,
                                   QType::Q5_1,   QType::Q8_0, QType::Q8_1, QType::Q2_K, QType::Q3_K,
                                   QType::Q4_K,   QType::Q5_K, QType::Q6_K, QType::Q8_K, QType::IQ4_NL,
                                   QType::IQ4_XS, QType::BF16, QType::MXFP4};
    for (QType qt : kAllWireTypes) {
        bool has_ref = false;
        for (const FormatSpec& f : kFormats)
            if (f.qt == qt)
                has_ref = true;
        EXPECT_EQ(dequant_gpu_supported(qt), has_ref)
            << "QType " << static_cast<unsigned>(qt)
            << ": dequant_gpu_supported=" << dequant_gpu_supported(qt) << " but reference=" << has_ref
            << ". A dequant path without an independent reference is untested "
               "numerics (AUDIT_arch_2026 D-5); a reference without a path is dead weight.";
    }
}

// =============================================================================
// fp16-DEQUANT GEMV TESTS (gemv_q8_0 / gemv_q6k — fp32 dot, no q8_1 act quant)
// Tolerance: fp16-class 1e-2 rel.
// =============================================================================
TEST(GgufRef, Q8_0_GemvFp16) {
    const int N = 256, K = 1024;
    Lcg g(0x5151u);
    std::vector<uint8_t> buf;
    build_q8_0(buf, N, K, g, NORMAL);
    void* dW = to_device(buf);
    std::vector<half> hx;
    half* dx = random_x(K, g, hx);
    half* dy = nullptr;
    cudaMalloc(&dy, N * sizeof(half));
    gemv_q8_0(dW, dx, dy, N, K, nullptr);
    cudaDeviceSynchronize();
    std::vector<half> hy(N);
    cudaMemcpy(hy.data(), dy, N * sizeof(half), cudaMemcpyDeviceToHost);
    ASSERT_FALSE(any_nan_inf(hy));

    std::vector<double> wref, yref;
    ref_dequant_all(buf, N, K, QType::Q8_0, wref);
    ref_gemv(wref, hx, N, K, yref);
    GemvStats s = gemv_eval(hy, yref);
    printf("[gemv Q8_0  fp16] N=%d K=%d max_rel=%.3e rms_rel=%.3e max_abs=%.3e ref_rms=%.3e (idx=%d "
           "gpu=%.4f ref=%.4f)\n",
           N, K, s.max_rel_scaled, s.rms_rel, s.max_abs, s.ref_rms, s.worst, __half2float(hy[s.worst]),
           yref[s.worst]);
    EXPECT_LT(s.max_rel_scaled, 1e-2) << "gemv_q8_0 fp16 error too large";
    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
}

TEST(GgufRef, Q6_K_GemvFp16) {
    const int N = 256, K = 1024;
    Lcg g(0x6161u);
    std::vector<uint8_t> buf;
    build_q6_k(buf, N, K, g, NORMAL);
    void* dW = to_device(buf);
    std::vector<half> hx;
    half* dx = random_x(K, g, hx);
    half* dy = nullptr;
    cudaMalloc(&dy, N * sizeof(half));
    gemv_q6k(dW, dx, dy, N, K, nullptr);
    cudaDeviceSynchronize();
    std::vector<half> hy(N);
    cudaMemcpy(hy.data(), dy, N * sizeof(half), cudaMemcpyDeviceToHost);
    ASSERT_FALSE(any_nan_inf(hy));

    std::vector<double> wref, yref;
    ref_dequant_all(buf, N, K, QType::Q6_K, wref);
    ref_gemv(wref, hx, N, K, yref);
    GemvStats s = gemv_eval(hy, yref);
    printf("[gemv Q6_K  fp16] N=%d K=%d max_rel=%.3e rms_rel=%.3e max_abs=%.3e ref_rms=%.3e (idx=%d "
           "gpu=%.4f ref=%.4f)\n",
           N, K, s.max_rel_scaled, s.rms_rel, s.max_abs, s.ref_rms, s.worst, __half2float(hy[s.worst]),
           yref[s.worst]);
    EXPECT_LT(s.max_rel_scaled, 1e-2) << "gemv_q6k fp16 error too large";
    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
}

// =============================================================================
// dp4a / MMVQ GEMV TESTS — these quantize activations to Q8_1 (amax/127).
// Tolerance: q8_1-activation band ~1-2%, asserted at 2.5e-2, MEASURED below.
// =============================================================================
namespace {

using namespace gguf_ref;  // Lcg, ScaleMode, ref_dequant_*, build_*, kFormats, format_spec
void run_dp4a_gemv(const char* name, QType qt, int N, int K,
                   void (*dp4a_fn)(const void*, const block_q8_1*, const float*, half*, int, int,
                                   cudaStream_t)) {
    Lcg g(0xD4A0u + static_cast<uint32_t>(qt));
    std::vector<uint8_t> buf;
    format_spec(qt).build(buf, N, K, g, NORMAL);
    void* dW = to_device(buf);
    std::vector<half> hx;
    half* dx = random_x(K, g, hx);

    int padded_blocks = ((K + 255) / 256) * 8;
    block_q8_1* q8 = nullptr;
    float* d8 = nullptr;
    cudaMalloc(&q8, padded_blocks * sizeof(block_q8_1));
    cudaMalloc(&d8, padded_blocks * sizeof(float));
    cudaMemset(q8, 0, padded_blocks * sizeof(block_q8_1));
    cudaMemset(d8, 0, padded_blocks * sizeof(float));
    quantize_fp16_to_q8_1(dx, q8, d8, K, nullptr);

    half* dy = nullptr;
    cudaMalloc(&dy, N * sizeof(half));
    dp4a_fn(dW, q8, d8, dy, N, K, nullptr);
    cudaDeviceSynchronize();
    std::vector<half> hy(N);
    cudaMemcpy(hy.data(), dy, N * sizeof(half), cudaMemcpyDeviceToHost);
    ASSERT_FALSE(any_nan_inf(hy)) << name;

    std::vector<double> wref, yref;
    ref_dequant_all(buf, N, K, qt, wref);
    ref_gemv(wref, hx, N, K, yref);
    GemvStats s = gemv_eval(hy, yref);
    printf("[gemv %-5s dp4a] N=%d K=%d max_rel=%.3e rms_rel=%.3e max_abs=%.3e ref_rms=%.3e (idx=%d "
           "gpu=%.4f ref=%.4f)\n",
           name, N, K, s.max_rel_scaled, s.rms_rel, s.max_abs, s.ref_rms, s.worst,
           __half2float(hy[s.worst]), yref[s.worst]);
    // q8_1 activation quant (amax/127 per 32-block) adds ~0.4% RMS per element.
    // Over a K-dot that does not fully cancel (correlated within a block), so we
    // bound the typical-magnitude-normalized RMS at 1.5% and the single worst
    // element at 5% (cancellation tail). Both MEASURED — see printed stats.
    EXPECT_LT(s.rms_rel, 1.5e-2) << name << ": dp4a gemv RMS outside q8_1-activation band";
    EXPECT_LT(s.max_rel_scaled, 5e-2) << name << ": dp4a gemv worst element outside band";

    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(q8);
    cudaFree(d8);
}

void run_mmvq_gemv(const char* name, QType qt, int N, int K,
                   void (*mmvq_fn)(const void*, const half*, half*, int, int, int, void*, size_t,
                                   cudaStream_t)) {
    Lcg g(0x33CCu + static_cast<uint32_t>(qt));
    std::vector<uint8_t> buf;
    if (qt == QType::Q8_0)
        build_q8_0(buf, N, K, g, NORMAL);
    else
        build_q4_k(buf, N, K, g, NORMAL);
    void* dW = to_device(buf);
    std::vector<half> hx;
    half* dx = random_x(K, g, hx);

    int q8_blocks = (K + 31) / 32;
    size_t scratch_size = (size_t)q8_blocks * 36 * 2;
    void* scratch = nullptr;
    cudaMalloc(&scratch, scratch_size);
    half* dy = nullptr;
    cudaMalloc(&dy, N * sizeof(half));
    mmvq_fn(dW, dx, dy, 1, N, K, scratch, scratch_size, nullptr);
    cudaDeviceSynchronize();
    std::vector<half> hy(N);
    cudaMemcpy(hy.data(), dy, N * sizeof(half), cudaMemcpyDeviceToHost);
    ASSERT_FALSE(any_nan_inf(hy)) << name;

    std::vector<double> wref, yref;
    ref_dequant_all(buf, N, K, qt, wref);
    ref_gemv(wref, hx, N, K, yref);
    GemvStats s = gemv_eval(hy, yref);
    printf("[mmvq %-5s    ] N=%d K=%d max_rel=%.3e rms_rel=%.3e max_abs=%.3e ref_rms=%.3e (idx=%d "
           "gpu=%.4f ref=%.4f)\n",
           name, N, K, s.max_rel_scaled, s.rms_rel, s.max_abs, s.ref_rms, s.worst,
           __half2float(hy[s.worst]), yref[s.worst]);
    // Same q8_1-activation band as the dp4a path (MMVQ quantizes x to Q8_1 too).
    EXPECT_LT(s.rms_rel, 1.5e-2) << name << ": mmvq gemv RMS outside q8_1-activation band";
    EXPECT_LT(s.max_rel_scaled, 5e-2) << name << ": mmvq gemv worst element outside band";

    cudaFree(dW);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(scratch);
}
}  // namespace

TEST(GgufRef, Q8_0_GemvDp4a) { run_dp4a_gemv("Q8_0", QType::Q8_0, 256, 1024, gemv_q8_0_q8_1); }
// Q4_0 dp4a-GEMV vs the fp64 reference. This oracle surfaced AUDIT.md F1:
// Q4_0_Traits::dp4a_block read nibbles INTERLEAVED (2k = qs[k] low, 2k+1 = qs[k]
// high) while standard ggml Q4_0 is SPLIT (element e = qs[e] low, e+16 = qs[e]
// high), mispairing weights with the natural-order Q8_1 activations (~6x off).
// Fixed to the split extraction the Q4_K path already uses; now asserted.
TEST(GgufRef, Q4_0_GemvDp4a) { run_dp4a_gemv("Q4_0", QType::Q4_0, 256, 1024, gemv_q4_0_q8_1); }
TEST(GgufRef, Q6_K_GemvDp4a) { run_dp4a_gemv("Q6_K", QType::Q6_K, 256, 1024, gemv_q6k_q8_1); }
TEST(GgufRef, Q4_K_GemvDp4a) { run_dp4a_gemv("Q4_K", QType::Q4_K, 256, 1024, gemv_q4_k_q8_1); }
TEST(GgufRef, Q5_K_GemvDp4a) { run_dp4a_gemv("Q5_K", QType::Q5_K, 256, 1024, gemv_q5_k_q8_1); }
// The two decode kernels D-5 found with no reference on either side: their
// dequant sibling and their dp4a trait share the same layout reading, so
// "dispatch == direct" would have agreed while both were wrong (Q3_K was).
TEST(GgufRef, Q2_K_GemvDp4a) { run_dp4a_gemv("Q2_K", QType::Q2_K, 256, 1024, gemv_q2_k_q8_1); }
TEST(GgufRef, Q3_K_GemvDp4a) { run_dp4a_gemv("Q3_K", QType::Q3_K, 256, 1024, gemv_q3_k_q8_1); }

TEST(GgufRef, Q8_0_GemvMmvq) { run_mmvq_gemv("Q8_0", QType::Q8_0, 256, 1024, ggml_mmvq_q8_0); }
TEST(GgufRef, Q4_K_GemvMmvq) { run_mmvq_gemv("Q4_K", QType::Q4_K, 256, 1024, ggml_mmvq_q4k); }

}  // namespace imp
