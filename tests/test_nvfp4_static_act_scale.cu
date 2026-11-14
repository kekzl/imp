// Static NVFP4 activation tensor scale: the quantizers divide by the
// checkpoint act scale, the GEMM alpha multiplies it back. Double reference on small-amax activations
// (the subnormal-UE4M3 regime of Gemma-4 GeGLU / Qwen3-30B o_proj inputs), small-M v2 and the prompt tile;
// mutants that drop the scale on either side are red; every producer matches the reference quantizer.

#include "compute/activation.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/layernorm.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_gemm_tile.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace imp;

namespace {

constexpr float kWTs = 0.0625f;

uint32_t lcg(uint32_t& h) {
    h = (h * 1664525u) + 1013904223u;
    return h >> 8;
}
double unit(uint32_t& h) { return (static_cast<double>(lcg(h) % 2000001u) / 1000000.0) - 1.0; }

double fp4(uint8_t nib) {
    static const double lut[16] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
    return lut[nib & 15];
}
double ue4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    return e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + (m / 8.0), e - 7);
}

template <typename T>
T* dev_upload(const std::vector<T>& v) {
    T* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, v.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return p;
}
template <typename T>
std::vector<T> dev_download(const void* p, size_t n) {
    std::vector<T> v(n);
    EXPECT_EQ(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return v;
}

// Row m >= 1 at block amax 2^-(14 - (m - 1) % 9), 2^-14 .. 2^-6 (all under the unit-scale UE4M3 normal floor
// 6 * 2^-6 = 0.094; one magnitude per row, as the GEMM row error is dominated by its largest blocks) and
// row 0 at amax ~1, past the calibration amax (saturates, as a checkpoint outlier).
std::vector<half> small_activations(int M, int K, uint32_t seed, float* amax_out) {
    std::vector<half> x(size_t(M) * K);
    uint32_t h = seed;
    float amax = 0.0f;
    for (int m = 0; m < M; ++m)
        for (int b = 0; b < K / 16; ++b) {
            const double mag = (m == 0) ? 1.0 : std::ldexp(1.0, -14 + ((m - 1) % 9));
            for (int i = 0; i < 16; ++i) {
                const half v = __float2half(static_cast<float>(unit(h) * mag));
                x[(size_t(m) * K) + (b * 16) + i] = v;
                amax = std::max(amax, std::fabs(__half2float(v)));
            }
        }
    *amax_out = amax;
    return x;
}

struct Weight {
    NvFP4QuantResult nv;
    std::vector<double> dq;  // decoded [N, K], tensor scale applied
};

Weight random_weight(int N, int K, uint32_t seed) {
    std::vector<uint8_t> packed(size_t(N) * K / 2), sf(size_t(N) * K / 16);
    uint32_t h = seed;
    for (auto& b : packed)
        b = static_cast<uint8_t>(lcg(h));
    for (auto& s : sf)
        s = static_cast<uint8_t>(((5 + (lcg(h) % 5)) << 3) | (lcg(h) & 7));
    Weight w;
    w.dq.resize(size_t(N) * K);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            const uint8_t byte = packed[(size_t(n) * K / 2) + (k / 2)];
            w.dq[(size_t(n) * K) + k] = fp4((k & 1) ? (byte >> 4) : byte) *
                                        ue4m3(sf[(size_t(n) * K / 16) + (k / 16)]) * kWTs;
        }
    w.nv.packed_data = dev_upload(packed);
    w.nv.micro_scales = dev_upload(sf);
    w.nv.tensor_scale = kWTs;
    w.nv.N = N;
    w.nv.K = K;
    return w;
}

// Relative Frobenius error of y [M, N] vs the double x . W^T, per row (row 0 is the amax ~1 row).
std::vector<double> row_rel_err(const std::vector<double>& x, const Weight& w, const std::vector<float>& y,
                                int M, int N, int K) {
    std::vector<double> e(M);
    for (int m = 0; m < M; ++m) {
        double num = 0.0, den = 0.0;
        for (int n = 0; n < N; ++n) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += x[(size_t(m) * K) + k] * w.dq[(size_t(n) * K) + k];
            const double d = y[(size_t(m) * N) + n] - ref;
            num += d * d;
            den += ref * ref;
        }
        e[m] = std::sqrt(num / std::max(den, 1e-300));
    }
    return e;
}

std::vector<double> to_double(const std::vector<half>& x) {
    std::vector<double> d(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        d[i] = static_cast<double>(__half2float(x[i]));
    return d;
}

double worst_small_row(const std::vector<double>& e) {
    double w = 0.0;
    for (size_t m = 1; m < e.size(); ++m)
        w = std::max(w, e[m]);
    return w;
}

enum class Arm { kScaled, kUnit, kMutantAlphaDropsScale, kMutantQuantDropsScale };

// Small-M v2: quantize_fp16_to_nvfp4_into with the quantize scale, Xq.tensor_scale = the alpha scale.
std::vector<float> smallm_y(const std::vector<half>& x, const Weight& w, int M, int N, int K, float s,
                            Arm arm) {
    const float qs = (arm == Arm::kScaled || arm == Arm::kMutantAlphaDropsScale) ? s : 1.0f;
    const float as = (arm == Arm::kScaled || arm == Arm::kMutantQuantDropsScale) ? s : 1.0f;
    half* dx = dev_upload(x);
    uint8_t *xq = nullptr, *xs = nullptr;
    half* dy = nullptr;
    void* ws = nullptr;
    EXPECT_EQ(cudaMalloc(&xq, size_t(M) * K / 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&xs, size_t(M) * K / 16), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dy, size_t(M) * N * sizeof(half)), cudaSuccess);
    const size_t wsb = gemm_nvfp4_smallm_v2_workspace_bytes(N, K);
    if (wsb > 0)
        EXPECT_EQ(cudaMalloc(&ws, wsb), cudaSuccess);
    quantize_fp16_to_nvfp4_into(dx, M, K, xq, xs, qs, nullptr);
    NvFP4QuantResult xr;
    xr.packed_data = xq;
    xr.micro_scales = xs;
    xr.tensor_scale = as;
    xr.N = M;
    xr.K = K;
    EXPECT_TRUE(gemm_nvfp4_smallm_v2_a4(w.nv, xr, dy, M, N, K, ws, nullptr));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto yh = dev_download<half>(dy, size_t(M) * N);
    std::vector<float> y(yh.size());
    for (size_t i = 0; i < y.size(); ++i)
        y[i] = __half2float(yh[i]);
    cudaFree(dx);
    cudaFree(xq);
    cudaFree(xs);
    cudaFree(dy);
    cudaFree(ws);
    return y;
}

std::vector<double> run_smallm(const std::vector<half>& x, const Weight& w, int M, int N, int K, float s,
                               Arm arm) {
    return row_rel_err(to_double(x), w, smallm_y(x, w, M, N, K, s, arm), M, N, K);
}

// Prompt tile: SfAtom quantize with the quantize scale, alpha = weight scale x the alpha scale.
std::vector<float> tile_y(const std::vector<half>& x, const Weight& w, int M, int N, int K, float s,
                          Arm arm) {
    const float qs = (arm == Arm::kScaled || arm == Arm::kMutantAlphaDropsScale) ? s : 1.0f;
    const float as = (arm == Arm::kScaled || arm == Arm::kMutantQuantDropsScale) ? s : 1.0f;
    CutlassNvFP4Weight cw;
    convert_nvfp4_to_cutlass(w.nv, cw, nullptr);
    half* dx = dev_upload(x);
    void *ad = nullptr, *asf = nullptr;
    half* dy = nullptr;
    const size_t sfb = cutlass_nvfp4_sf_size(M, K);
    EXPECT_EQ(cudaMalloc(&ad, size_t(M) * K / 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&asf, sfb), cudaSuccess);
    EXPECT_EQ(cudaMemset(asf, 0, sfb), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dy, size_t(M) * N * sizeof(half)), cudaSuccess);
    quantize_fp16_to_nvfp4_cutlass(dx, ad, asf, M, K, nullptr, qs);
    EXPECT_TRUE(
        gemm_nvfp4_tile(ad, asf, cw.data, cw.scale_factors, cw.tensor_scale * as, dy, M, N, K, nullptr));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto yh = dev_download<half>(dy, size_t(M) * N);
    std::vector<float> y(yh.size());
    for (size_t i = 0; i < y.size(); ++i)
        y[i] = __half2float(yh[i]);
    free_cutlass_nvfp4_weight(cw);
    cudaFree(dx);
    cudaFree(ad);
    cudaFree(asf);
    cudaFree(dy);
    return y;
}

std::vector<double> run_tile(const std::vector<half>& x, const Weight& w, int M, int N, int K, float s,
                             Arm arm) {
    return row_rel_err(to_double(x), w, tile_y(x, w, M, N, K, s, arm), M, N, K);
}

template <typename Run>
void check_arms(Run run, const char* what) {
    constexpr int K = 1024, N = 256;
    const int M = 16;
    float amax = 0.0f;
    const auto x = small_activations(M, K, 7u, &amax);
    // Calibration amax of the small rows only: row 0 (amax ~1) saturates past it, as a checkpoint outlier.
    const float s = std::ldexp(1.0f, -6) / 2688.0f;
    Weight w = random_weight(N, K, 11u);
    const auto scaled = run(x, w, M, N, K, s, Arm::kScaled);
    const auto unit_s = run(x, w, M, N, K, s, Arm::kUnit);
    const auto mut_a = run(x, w, M, N, K, s, Arm::kMutantAlphaDropsScale);
    const auto mut_q = run(x, w, M, N, K, s, Arm::kMutantQuantDropsScale);
    const double e_s = worst_small_row(scaled), e_1 = worst_small_row(unit_s);
    printf("[%s] worst small-row rel err: static scale %.4f, unit scale %.4f, mutants %.3g / %.3g\n", what,
           e_s, e_1, worst_small_row(mut_a), worst_small_row(mut_q));
    // NVFP4 W4A4 at normal block scales on uniform blocks: ~0.17 rel error per row; the unit scale rounds
    // most values of the 2^-14 .. 2^-10 rows to 0 or +-0.5 steps (block scale floor 2^-9, plain and SfAtom).
    EXPECT_LT(e_s, 0.2) << what;
    EXPECT_GT(e_1, 2.0 * e_s) << what;
    // Dropping the scale on one side moves every output by 1/s (alpha) or s (quantize): rel err ~1e5 / ~1.
    EXPECT_GT(worst_small_row(mut_a), 0.9) << what << ": alpha without the act scale must be red";
    EXPECT_GT(worst_small_row(mut_q), 0.9) << what << ": quantize without the act scale must be red";
    cudaFree(w.nv.packed_data);
    cudaFree(w.nv.micro_scales);
}

// Reference quantize of `x` (plain layout) at tensor scale s, for the producer comparisons.
void ref_plain(const half* dx, int M, int K, float s, std::vector<uint8_t>& p, std::vector<uint8_t>& sc) {
    uint8_t *xq = nullptr, *xs = nullptr;
    ASSERT_EQ(cudaMalloc(&xq, size_t(M) * K / 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&xs, size_t(M) * K / 16), cudaSuccess);
    quantize_fp16_to_nvfp4_into(dx, M, K, xq, xs, s, nullptr);
    p = dev_download<uint8_t>(xq, size_t(M) * K / 2);
    sc = dev_download<uint8_t>(xs, size_t(M) * K / 16);
    cudaFree(xq);
    cudaFree(xs);
}

}  // namespace

TEST(Nvfp4StaticActScaleTest, SmallMDoubleReference) { check_arms(run_smallm, "small-M v2"); }

TEST(Nvfp4StaticActScaleTest, PromptTileDoubleReference) { check_arms(run_tile, "prompt tile"); }

// Fused producers (rmsnorm / swiglu + quantize, plain and SfAtom) at a non-unit act scale write the bytes
// of the reference quantizer on their own FP16 output; scale 1.0 keeps the default-argument bytes.
TEST(Nvfp4StaticActScaleTest, ProducersMatchReferenceQuantize) {
    constexpr int M = 16, K = 2048;
    float amax = 0.0f;
    const auto xh = small_activations(M, K, 3u, &amax);
    std::vector<half> wv(K);
    uint32_t h = 5u;
    for (auto& v : wv)
        v = __float2half(static_cast<float>(0.5 + (0.5 * unit(h))));
    half* dx = dev_upload(xh);
    half* dw = dev_upload(wv);
    half* dg = dev_upload(small_activations(M, K, 13u, &amax));
    half* du = dev_upload(small_activations(M, K, 17u, &amax));
    half* dout = nullptr;
    uint8_t *xq = nullptr, *xs = nullptr;
    ASSERT_EQ(cudaMalloc(&dout, size_t(M) * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&xq, size_t(M) * K / 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&xs, size_t(M) * K / 16), cudaSuccess);
    const int64_t shp[2] = {M, K};
    const int64_t wshp[1] = {K};
    Tensor tx(dx, QType::F16, 2, shp, true), tw(dw, QType::F16, 1, wshp, true),
        to(dout, QType::F16, 2, shp, true);
    Tensor tg(dg, QType::F16, 2, shp, true), tu(du, QType::F16, 2, shp, true);
    for (const float s : {1.0f, std::ldexp(1.0f, -6) / 2688.0f, 0.37f}) {
        std::vector<uint8_t> rp, rs;
        // rmsnorm_nvfp4
        ASSERT_TRUE(rmsnorm_nvfp4(tx, tw, to, xq, xs, 1e-6f, nullptr, 0.0f, s));
        ref_plain(dout, M, K, s, rp, rs);
        EXPECT_EQ(dev_download<uint8_t>(xq, rp.size()), rp) << "rmsnorm_nvfp4 packed, s=" << s;
        EXPECT_EQ(dev_download<uint8_t>(xs, rs.size()), rs) << "rmsnorm_nvfp4 scales, s=" << s;
        // swiglu_quantize_nvfp4
        ASSERT_TRUE(swiglu_quantize_nvfp4(tg, tu, to, xq, xs, nullptr, s));
        ref_plain(dout, M, K, s, rp, rs);
        EXPECT_EQ(dev_download<uint8_t>(xq, rp.size()), rp) << "swiglu_quantize_nvfp4 packed, s=" << s;
        EXPECT_EQ(dev_download<uint8_t>(xs, rs.size()), rs) << "swiglu_quantize_nvfp4 scales, s=" << s;
        // SfAtom: swiglu_quantize_fp16_to_nvfp4_cutlass vs swiglu + quantize_fp16_to_nvfp4_cutlass
        const size_t sfb = cutlass_nvfp4_sf_size(M, K);
        void *a1 = nullptr, *s1 = nullptr, *a2 = nullptr, *s2 = nullptr;
        ASSERT_EQ(cudaMalloc(&a1, size_t(M) * K / 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&a2, size_t(M) * K / 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&s1, sfb), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&s2, sfb), cudaSuccess);
        ASSERT_EQ(cudaMemset(s1, 0, sfb), cudaSuccess);
        ASSERT_EQ(cudaMemset(s2, 0, sfb), cudaSuccess);
        swiglu_quantize_fp16_to_nvfp4_cutlass(dg, du, dout, a1, s1, M, K, nullptr, s);
        quantize_fp16_to_nvfp4_cutlass(dout, a2, s2, M, K, nullptr, s);
        EXPECT_EQ(dev_download<uint8_t>(a1, size_t(M) * K / 2), dev_download<uint8_t>(a2, size_t(M) * K / 2))
            << "SfAtom swiglu packed, s=" << s;
        EXPECT_EQ(dev_download<uint8_t>(s1, sfb), dev_download<uint8_t>(s2, sfb))
            << "SfAtom swiglu scales, s=" << s;
        if (s == 1.0f) {  // unit scale = the pre-existing default-argument bytes
            quantize_fp16_to_nvfp4_cutlass(dout, a2, s2, M, K, nullptr);
            EXPECT_EQ(dev_download<uint8_t>(a1, size_t(M) * K / 2),
                      dev_download<uint8_t>(a2, size_t(M) * K / 2));
        }
        cudaFree(a1);
        cudaFree(a2);
        cudaFree(s1);
        cudaFree(s2);
    }
    // A non-unit scale must move the bytes (guards a producer that ignores its argument).
    std::vector<uint8_t> p1, s1v, p2, s2v;
    ref_plain(dx, M, K, 1.0f, p1, s1v);
    ref_plain(dx, M, K, std::ldexp(1.0f, -6) / 2688.0f, p2, s2v);
    EXPECT_NE(s1v, s2v);
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dg);
    cudaFree(du);
    cudaFree(dout);
    cudaFree(xq);
    cudaFree(xs);
}

namespace {

// Tiny blocks at unit act scale (imp-quantize exports without a checkpoint scale): every block scale
// amax / 6 sits under the UE4M3 floor 2^-9. Block 0 amax 0, block 1 amax <= 6 * 2^-10 (scale <= 2^-10, rounds
// to 0 even in RNE), the rest amax 2^-(10 - m % 4) per row m (2^-10 .. 2^-7).
std::vector<half> tiny_activations(int M, int K, uint32_t seed) {
    std::vector<half> x(size_t(M) * K);
    uint32_t h = seed;
    for (int m = 0; m < M; ++m)
        for (int b = 0; b < K / 16; ++b) {
            const double mag = (b == 0)   ? 0.0
                               : (b == 1) ? 6.0 * std::ldexp(1.0, -10)
                                          : std::ldexp(1.0, -10 + (m % 4));
            for (int i = 0; i < 16; ++i)
                x[(size_t(m) * K) + (b * 16) + i] = __float2half(static_cast<float>(unit(h) * mag));
        }
    return x;
}

// Plain-layout NVFP4 [M, K] decoded to fp64 at tensor scale s.
std::vector<double> decode_plain(const std::vector<uint8_t>& p, const std::vector<uint8_t>& sc, int M, int K,
                                 float s) {
    std::vector<double> d(size_t(M) * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            const uint8_t byte = p[(size_t(m) * K / 2) + (k / 2)];
            d[(size_t(m) * K) + k] = fp4((k & 1) ? (byte >> 4) : byte) *
                                     ue4m3(sc[(size_t(m) * K / 16) + (k / 16)]) * s;
        }
    return d;
}

size_t sfatom_at(int row, int kg, int K) {  // mirrors sfatom_offset() in nvfp4_sfatom_quant.cu
    const int n_k_tiles = (K + 63) / 64, row_local = row % 128;
    return (static_cast<size_t>(((row / 128) * n_k_tiles) + (kg / 4)) * 512) +
           static_cast<size_t>(((row_local % 32) * 16) + ((row_local / 32) * 4) + (kg % 4));
}

}  // namespace

// One rule for a block scale under 2^-9: clamp to 2^-9 (nvfp4_encode_micro_scale). The prompt SfAtom quantize
// wrote the byte 0 there (block read as zero in the GEMM) while the plain quantize clamped: prompt rows and
// decode rows of one token differed. SfAtom bytes = plain bytes, at unit and static scales, across 2 atom
// rows.
TEST(Nvfp4TinyBlockScaleTest, SfAtomBytesEqualPlainQuantize) {
    constexpr int M = 144, K = 1024;
    const auto x = tiny_activations(M, K, 23u);
    half* dx = dev_upload(x);
    const size_t sfb = cutlass_nvfp4_sf_size(M, K);
    void *a = nullptr, *sf = nullptr;
    ASSERT_EQ(cudaMalloc(&a, size_t(M) * K / 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&sf, sfb), cudaSuccess);
    for (const float s : {1.0f, 0.37f, std::ldexp(1.0f, -6) / 2688.0f}) {
        std::vector<uint8_t> rp, rs;
        ref_plain(dx, M, K, s, rp, rs);
        ASSERT_EQ(cudaMemset(sf, 0xFF, sfb), cudaSuccess);
        quantize_fp16_to_nvfp4_cutlass(dx, a, sf, M, K, nullptr, s);
        const auto ap = dev_download<uint8_t>(a, rp.size());
        const auto as = dev_download<uint8_t>(sf, sfb);
        size_t bad_s = 0, floor_blocks = 0;
        for (int m = 0; m < M; ++m)
            for (int kg = 0; kg < K / 16; ++kg) {
                const uint8_t want = rs[(size_t(m) * K / 16) + kg];
                bad_s += as[sfatom_at(m, kg, K)] != want ? 1 : 0;
                floor_blocks += want == 0x01 ? 1 : 0;
            }
        printf("[tiny SfAtom] s=%g: sf bad %zu / %d, blocks at the 2^-9 floor %zu\n", s, bad_s, M * K / 16,
               floor_blocks);
        EXPECT_EQ(ap, rp) << "SfAtom packed vs plain, s=" << s;
        EXPECT_EQ(bad_s, 0u) << "SfAtom scale bytes vs plain, s=" << s;
        if (s == 1.0f)  // the regime under test: blocks 0 and 1 plus the 2^-10 .. 2^-7 rows clamp
            EXPECT_GT(floor_blocks, size_t(M) * 2);
    }
    cudaFree(dx);
    cudaFree(a);
    cudaFree(sf);
}

// Prompt tile and small-M v2 at unit scale on tiny blocks vs the fp64 GEMM over the clamp-rule quantized
// activations: both match to FP16 output rounding. A zero-byte encode reads 1.0 on every row.
TEST(Nvfp4TinyBlockScaleTest, PromptTileAndSmallMMatchDoubleReference) {
    constexpr int M = 16, K = 1024, N = 256;
    const auto x = tiny_activations(M, K, 29u);
    Weight w = random_weight(N, K, 31u);
    half* dx = dev_upload(x);
    std::vector<uint8_t> rp, rs;
    ref_plain(dx, M, K, 1.0f, rp, rs);
    const auto xq = decode_plain(rp, rs, M, K, 1.0f);
    const auto e_tile = row_rel_err(xq, w, tile_y(x, w, M, N, K, 1.0f, Arm::kUnit), M, N, K);
    const auto e_sm = row_rel_err(xq, w, smallm_y(x, w, M, N, K, 1.0f, Arm::kUnit), M, N, K);
    const auto e_x = row_rel_err(to_double(x), w, tile_y(x, w, M, N, K, 1.0f, Arm::kUnit), M, N, K);
    double wt = 0.0, ws = 0.0, wx = 0.0;
    for (int m = 0; m < M; ++m) {
        wt = std::max(wt, e_tile[m]);
        ws = std::max(ws, e_sm[m]);
        wx = std::max(wx, e_x[m]);
    }
    printf("[tiny GEMM] worst row rel err vs quantized fp64: tile %.3g, small-M %.3g; tile vs exact x %.4f\n",
           wt, ws, wx);
    EXPECT_LT(wt, 1e-2) << "prompt tile drops tiny blocks";
    EXPECT_LT(ws, 1e-2) << "small-M v2";
    EXPECT_LT(wx, 0.9) << "tile output vs the unquantized activations";
    cudaFree(dx);
    cudaFree(w.nv.packed_data);
    cudaFree(w.nv.micro_scales);
}
