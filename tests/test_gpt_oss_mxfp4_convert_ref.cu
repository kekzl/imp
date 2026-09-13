// TEST_AUDIT(retired) P1.1/R1.1 (#576): gpt-oss MXFP4->NVFP4 expert converter
// (src/quant/gpt_oss_mxfp4_convert.cu, previously 0 tests) vs a format-spec-derived
// independent fp64 reference. MXFP4 nibble order was a REAL bug in the #560 sweep - the
// class this test must catch.
// Independence: the reference decodes ORIGINAL MXFP4 values in fp64 straight from the format
// spec (v = e2m1_lut[nibble&7] * sign * 2^(u-127)), never from the converter; the converter's
// output is dequantized by imp's GPU kernel and compared back to that same original-MXFP4
// reference. Synthetic blocks are built byte-level via LCG, never via a quantizer round-trip.
// Key fact: ue8m0 scales are exact powers of two, so the converter's relative scale
// 2^(u-127-ts_exp) is also a power of two, and e4m3 represents every power of two in
// [2^-9,2^8] EXACTLY - so in-range blocks are BIT-EXACT modulo the final f16 store. Only loss
// is e4m3 clamping outside that range (characterized separately) and the f16 store.
// Tolerances: in-range <= 1e-3 rel; clamped blocks vs the SAME-clamped reference <= 1e-3;
// extra_scale folded into the reference too, same 1e-3; hard no-NaN/Inf guard on every
// expert/block.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "quant/gpt_oss_mxfp4_convert.h"
#include "quant/nvfp4_quant.h"

namespace imp {
namespace {

// Deterministic byte-level LCG (Numerical Recipes constants), independent of imp; fills raw
// nibble bytes and picks ue8m0 exponents.
struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 1664525u + 1013904223u;
        return s;
    }
    uint8_t byte() { return static_cast<uint8_t>(next() >> 24); }
};

// E2M1 magnitude LUT (4-bit FP4: 1 sign + 2 exp + 1 mantissa), the published
// format table — identical to imp's kFP4E2M1Dequant but written here from the
// E2M1 definition (codes 0..7 -> {0,.5,1,1.5,2,3,4,6}).
constexpr double kE2M1[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

// fp64 decode of one MXFP4 nibble at block exponent u: value = sign*e2m1_mag*2^(u-127). Low
// nibble of byte i -> element 2i, high nibble -> element 2i+1 (the format's nibble layout,
// same one the converter copies and imp's dequant reads).
inline double mxfp4_decode(uint8_t nibble, int u) {
    double mag = kE2M1[nibble & 0x7];
    double v = mag * std::ldexp(1.0, u - 127);
    return (nibble & 0x8) ? -v : v;
}

// e4m3 round-trip of a positive scale, to mirror the converter's
// __nv_fp8_e4m3 micro-scale store when we need the CLAMPED reference.
inline double e4m3_roundtrip(double rel) {
    __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(static_cast<float>(rel));
    return static_cast<double>(static_cast<float>(f8));
}

inline float f2h2f(double f) { return __half2float(__float2half(static_cast<float>(f))); }

// Packed MXFP4 projection: blocks [ne,n_rows_total,K/2] nibbles + scales
// [ne,n_rows_total,K/32] ue8m0 exponents. exp_lo/exp_hi bound the random exponent range PER
// ROW (some rows hot, some cold, stressing the per-expert tensor-scale max); a forced
// outlier exponent can be planted to drive clamping.
struct MxfpBuf {
    std::vector<uint8_t> blocks;  // [ne * n_rows_total * K/2]
    std::vector<uint8_t> scales;  // [ne * n_rows_total * K/32]
};

MxfpBuf build_mxfp4(int ne, int64_t n_rows_total, int64_t K, Lcg& g, int exp_lo, int exp_hi,
                    int outlier_u = -1) {
    const int64_t row_bytes = K / 2;
    const int64_t kb32 = K / 32;
    MxfpBuf b;
    b.blocks.resize(static_cast<size_t>(ne) * n_rows_total * row_bytes);
    b.scales.resize(static_cast<size_t>(ne) * n_rows_total * kb32);
    int span = exp_hi - exp_lo + 1;
    for (int e = 0; e < ne; e++) {
        for (int64_t r = 0; r < n_rows_total; r++) {
            // nibble bytes: full random range exercises every e2m1 code + sign.
            uint8_t* bp =
                b.blocks.data() + (static_cast<size_t>(e) * n_rows_total + r) * row_bytes;
            for (int64_t i = 0; i < row_bytes; i++)
                bp[i] = g.byte();
            // ue8m0 exponents: random in [exp_lo, exp_hi], plus a small per-row
            // bias so different rows have different magnitudes.
            uint8_t* sp =
                b.scales.data() + (static_cast<size_t>(e) * n_rows_total + r) * kb32;
            int bias = static_cast<int>(g.byte()) % 5;  // 0..4 octave row bias
            for (int64_t bk = 0; bk < kb32; bk++) {
                int u = exp_lo + (g.byte() % span) + bias;
                if (u > 254)
                    u = 254;
                sp[bk] = static_cast<uint8_t>(u);
            }
            // plant a single hot block per expert to make the tensor scale wide
            // and drive low blocks toward the e4m3 floor (clamp path).
            if (outlier_u >= 0 && r == 0)
                sp[0] = static_cast<uint8_t>(outlier_u);
        }
    }
    return b;
}

bool any_nan_inf(const std::vector<half>& v) {
    for (half h : v) {
        float f = __half2float(h);
        if (std::isnan(f) || std::isinf(f))
            return true;
    }
    return false;
}

// fp64 reference of the converted-and-dequantized cache: replays the converter's ONLY
// non-trivial decision (ts_exp = max_u-127-8) from the format, applying the SAME
// deterministic e4m3 clamp the converter is contractually specified to apply - never
// transcribing the kernel itself.
void ref_convert_dequant(const MxfpBuf& b, int ne, int64_t n_rows_total, int64_t K, int row_offset,
                         int row_stride, float extra_scale, std::vector<double>& out,
                         int& clamped_blocks) {
    const int64_t N = n_rows_total / row_stride;
    const int64_t kb32 = K / 32;
    const int64_t row_bytes = K / 2;
    out.assign(static_cast<size_t>(ne) * N * K, 0.0);
    clamped_blocks = 0;

    for (int e = 0; e < ne; e++) {
        // Pass 1: per-expert max ue8m0 exponent over the SELECTED rows.
        int max_u = 0;
        const uint8_t* es_base = b.scales.data() + static_cast<size_t>(e) * n_rows_total * kb32;
        for (int64_t r = 0; r < N; r++) {
            const uint8_t* sr = es_base + static_cast<size_t>(row_offset + r * row_stride) * kb32;
            for (int64_t bk = 0; bk < kb32; bk++)
                max_u = std::max(max_u, static_cast<int>(sr[bk]));
        }
        const int ts_exp = max_u - 127 - 8;
        // tensor scale = 2^ts_exp * extra_scale (FP32, as the converter stores).
        const double tensor_scale = std::ldexp(1.0, ts_exp) * static_cast<double>(extra_scale);

        const uint8_t* eb_base = b.blocks.data() + static_cast<size_t>(e) * n_rows_total * row_bytes;
        for (int64_t r = 0; r < N; r++) {
            const int64_t src_row = row_offset + r * row_stride;
            const uint8_t* bp = eb_base + static_cast<size_t>(src_row) * row_bytes;
            const uint8_t* sr = es_base + static_cast<size_t>(src_row) * kb32;
            for (int64_t bk = 0; bk < kb32; bk++) {
                int u = sr[bk];
                // relative scale (power of two) that the converter re-bases into
                // e4m3, then clamps to [2^-9, 448].
                double rel = std::ldexp(1.0, u - 127 - ts_exp);
                bool clamped = false;
                if (rel > 448.0) {
                    rel = 448.0;
                    clamped = true;
                } else if (rel > 0.0 && rel < 0.001953125 /* 2^-9 */) {
                    rel = 0.001953125;
                    clamped = true;
                }
                // The converter then stores rel as e4m3 (exact for powers of two
                // in range; the clamp endpoints 448 and 2^-9 are e4m3-exact too).
                double rel_e4m3 = e4m3_roundtrip(rel);
                if (clamped)
                    clamped_blocks++;
                double combined = tensor_scale * rel_e4m3;
                // 32 nibbles in this MXFP4 block -> 32 output elements.
                for (int j = 0; j < 32; j++) {
                    int byte_idx = static_cast<int>(bk) * 16 + j / 2;
                    uint8_t byte = bp[byte_idx];
                    uint8_t nib = (j & 1) ? (byte >> 4) & 0xF : byte & 0xF;
                    double mag = kE2M1[nib & 0x7] * combined;
                    double v = (nib & 0x8) ? -mag : mag;
                    out[(static_cast<size_t>(e) * N + r) * K + bk * 32 + j] = v;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Run the converter, dequantize on GPU, compare to the fp64 reference.
// -----------------------------------------------------------------------------
struct Stats {
    double max_rel = 0.0;
    double max_abs = 0.0;
    int worst = 0;
    int nan_count = 0;
};

void run_case(const char* name, int ne, int64_t n_rows_total, int64_t K, int row_offset, int row_stride,
              float extra_scale, Lcg& g, int exp_lo, int exp_hi, int outlier_u, double rel_tol,
              bool expect_clamp) {
    SCOPED_TRACE(name);
    MxfpBuf b = build_mxfp4(ne, n_rows_total, K, g, exp_lo, exp_hi, outlier_u);

    NvFP4MoEQuantResult res{};
    std::vector<float> h_tscales;
    ASSERT_TRUE(gpt_oss_convert_experts_to_nvfp4(b.blocks.data(), b.scales.data(), ne, n_rows_total, K,
                                                 row_offset, row_stride, res, extra_scale, &h_tscales))
        << name << ": converter returned false";

    const int64_t N = n_rows_total / row_stride;
    ASSERT_EQ(res.N, N);
    ASSERT_EQ(res.K, K);
    ASSERT_EQ(res.n_experts, ne);

    // GPU dequant of the produced NVFP4 cache -> FP16 [ne, N, K].
    const size_t elems = static_cast<size_t>(ne) * N * K;
    half* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, elems * sizeof(half)), cudaSuccess);
    dequantize_nvfp4_moe_to_fp16(res, d_out, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<half> hOut(elems);
    cudaMemcpy(hOut.data(), d_out, elems * sizeof(half), cudaMemcpyDeviceToHost);

    // fp64 reference (original MXFP4 values, with the contractual e4m3 clamp).
    std::vector<double> ref;
    int clamped_blocks = 0;
    ref_convert_dequant(b, ne, n_rows_total, K, row_offset, row_stride, extra_scale, ref, clamped_blocks);

    // Cross-checks that the converter chose the tensor scale replayed here: host tensor-scale
    // copy must equal 2^(max_u-127-8)*extra_scale per expert - independent confirmation the
    // per-expert max scan agrees (a wrong max_u would shift every block's relative scale).
    const int64_t kb32 = K / 32;
    for (int e = 0; e < ne; e++) {
        int max_u = 0;
        const uint8_t* es_base = b.scales.data() + static_cast<size_t>(e) * n_rows_total * kb32;
        for (int64_t r = 0; r < N; r++) {
            const uint8_t* sr = es_base + static_cast<size_t>(row_offset + r * row_stride) * kb32;
            for (int64_t bk = 0; bk < kb32; bk++)
                max_u = std::max(max_u, static_cast<int>(sr[bk]));
        }
        double want = std::ldexp(1.0, max_u - 127 - 8) * static_cast<double>(extra_scale);
        EXPECT_NEAR(h_tscales[e], want, want * 1e-6) << name << ": expert " << e << " tensor scale";
    }

    ASSERT_FALSE(any_nan_inf(hOut)) << name << ": dequant produced NaN/Inf";

    Stats s;
    for (size_t i = 0; i < elems; i++) {
        double gpu = static_cast<double>(__half2float(hOut[i]));
        // round the fp64 reference through f16 the way the kernel stores it, so
        // we isolate arithmetic divergence from the unavoidable f16 step.
        double r16 = static_cast<double>(f2h2f(ref[i]));
        double a = std::fabs(gpu - r16);
        double rel = std::fabs(r16) > 1e-4 ? a / std::fabs(r16) : a;
        if (rel > s.max_rel) {
            s.max_rel = rel;
            s.max_abs = a;
            s.worst = static_cast<int>(i);
        }
    }
    printf("[mxfp4->nvfp4 %-14s] ne=%d N=%ld K=%ld off=%d stride=%d xs=%.4g clamped_blk=%d max_rel=%.3e "
           "max_abs=%.3e (idx=%d gpu=%.6f ref=%.6f)\n",
           name, ne, (long)N, (long)K, row_offset, row_stride, extra_scale, clamped_blocks, s.max_rel,
           s.max_abs, s.worst, __half2float(hOut[s.worst]), ref[s.worst]);

    if (expect_clamp)
        EXPECT_GT(clamped_blocks, 0) << name << ": expected at least one clamped block";
    else
        EXPECT_EQ(clamped_blocks, 0) << name << ": unexpected scale clamp (range too wide for in-range case)";

    EXPECT_LT(s.max_rel, rel_tol) << name << ": MXFP4->NVFP4 dequant rel error too large";

    free_nvfp4_moe_result(res);
    cudaFree(d_out);
}

}  // namespace

// In-range conversion: ue8m0 exponents within ~16 octaves of the per-expert max => every
// relative scale is e4m3-exact => bit-exact modulo the f16 store. Covers down-proj (offset 0,
// stride 1) and interleaved gate/up slices.
TEST(GptOssMxfp4ConvertRef, DownProj_InRange) {
    Lcg g(0xA11CEu);
    // down: full rows, no interleave. K=256 (8 mxfp4 blocks), 6 experts, 24 rows.
    // exponents in [120,130] => spread 10 octaves, comfortably inside e4m3 range.
    run_case("down/inrange", 6, 24, 256, /*off*/ 0, /*stride*/ 1, /*xs*/ 1.0f, g, 120, 130, -1, 1e-3,
             /*expect_clamp*/ false);
}

TEST(GptOssMxfp4ConvertRef, GateSlice_InRange) {
    Lcg g(0xB22DFu);
    // gate_up interleaved: 12 physical rows -> 6 gate rows (offset 0, stride 2). Window must
    // keep every block's relative exponent in [-9,+8]: with the per-row bias (0..4), effective
    // max_u = exp_hi+4, so (exp_hi+4)-exp_lo <= 17 guarantees no clamp; [120,128] gives span 12 < 17.
    run_case("gate/inrange", 4, 12, 256, /*off*/ 0, /*stride*/ 2, /*xs*/ 1.0f, g, 120, 128, -1, 1e-3,
             /*expect_clamp*/ false);
}

TEST(GptOssMxfp4ConvertRef, UpSlice_InRange) {
    Lcg g(0xC33A0u);
    // up slice: offset 1, stride 2 (the odd interleaved rows). Same in-range
    // window as gate (span 12 < 17 => bit-exact, no clamp).
    run_case("up/inrange", 4, 12, 256, /*off*/ 1, /*stride*/ 2, /*xs*/ 1.0f, g, 120, 128, -1, 1e-3,
             /*expect_clamp*/ false);
}

// extra_scale: the down-proj residual 2^-4 rescale folds into the tensor scale; the
// reference applies the same factor, so the result stays bit-exact modulo the f16 store.
TEST(GptOssMxfp4ConvertRef, DownProj_ExtraScale) {
    Lcg g(0xD44B1u);
    run_case("down/xs=2^-4", 5, 20, 512, /*off*/ 0, /*stride*/ 1, /*xs*/ 0.0625f, g, 122, 128, -1, 1e-3,
             /*expect_clamp*/ false);
}

// CLAMP path: plants one very-hot block (u=254) so the per-expert tensor scale sits ~127
// octaves above the bulk, underflowing every normal block's relative scale past the e4m3
// 2^-9 floor. The converter's contract is clamp+log; the reference applies the SAME clamp,
// so the result must still match tightly - the Gemma mode-2 scale-collapse analogue, here it
// must stay finite and on-oracle.
TEST(GptOssMxfp4ConvertRef, Clamp_HotOutlierBlock) {
    Lcg g(0xE55C2u);
    // Bulk exponents low (100..104), one block forced hot at u_hot=135: wide enough that bulk
    // blocks underflow the e4m3 2^-9 floor (rel-exp 100-135+8=-27, clamped) but not wide enough
    // for the hot block to overflow f16 (6*2^8=1536, finite). The reference applies the same
    // deterministic clamp, so the result still matches tightly.
    run_case("clamp/floor", 4, 16, 256, /*off*/ 0, /*stride*/ 1, /*xs*/ 1.0f, g, 100, 104, /*outlier_u*/ 135,
             /*rel_tol*/ 1e-3, /*expect_clamp*/ true);
}

// All-zero MXFP4 (every nibble 0): output must be exactly zero, finite, with the tensor
// scale still well-defined (max_u from the all-equal scales).
TEST(GptOssMxfp4ConvertRef, AllZeroNibbles) {
    const int ne = 3;
    const int64_t n_rows = 12, K = 256;
    MxfpBuf b;
    const int64_t kb32 = K / 32, row_bytes = K / 2;
    b.blocks.assign(static_cast<size_t>(ne) * n_rows * row_bytes, 0);  // all nibbles 0 -> e2m1 code 0
    b.scales.assign(static_cast<size_t>(ne) * n_rows * kb32, 128);     // uniform exponent
    NvFP4MoEQuantResult res{};
    ASSERT_TRUE(gpt_oss_convert_experts_to_nvfp4(b.blocks.data(), b.scales.data(), ne, n_rows, K, 0, 1, res));
    const size_t elems = static_cast<size_t>(ne) * n_rows * K;
    half* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, elems * sizeof(half)), cudaSuccess);
    dequantize_nvfp4_moe_to_fp16(res, d_out, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<half> hOut(elems);
    cudaMemcpy(hOut.data(), d_out, elems * sizeof(half), cudaMemcpyDeviceToHost);
    for (half h : hOut)
        EXPECT_EQ(__half2float(h), 0.0f) << "all-zero MXFP4 must dequantize to exact zero";
    free_nvfp4_moe_result(res);
    cudaFree(d_out);
}

}  // namespace imp
