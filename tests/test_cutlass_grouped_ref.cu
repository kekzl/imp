// =============================================================================
// TEST_AUDIT.md Re-Audit 2026-06-06 — P1.2 / R1.2 (issue #576).
// CUTLASS 3.x NVFP4 BlockScaled Grouped GEMM (src/compute/gemm_cutlass_grouped_3x.cu)
// — the #574 pp512-10x MoE prefill path — vs an INDEPENDENT fp64 CPU reference.
//
// WHY THIS EXISTS (audit §2 "CUTLASS grouped GEMM" = class B):
//   The sibling test (test_cutlass_grouped_3x_nvfp4.cu) compares the grouped
//   dispatch against the per-expert single GEMM, but BOTH run through the same
//   CUTLASS adapter ("only the staging buffer build differs") — it catches
//   staging bugs, not adapter/math bugs. The new dominant prefill path had no
//   independent reference. This test adds one: a per-expert fp64 CPU matmul
//   that decodes the SAME quantized bits the GPU GEMM consumes, then asserts
//   the f16-accumulation class tolerance, NOT the loose NVFP4 quant envelope.
//
// INDEPENDENCE (audit §3 — no tautologies, no transcribed kernel):
//   * The reference decodes the EXACT bits the GPU GEMM sees: the packed E2M1
//     nibbles (RowMajor [rows, K/2]) plus the per-16-block UE4M3 micro-scales
//     read back from device, de-swizzled from CUTLASS SfAtom layout. Decode is
//     from the published NVFP4 format definition (E2M1 magnitude LUT + UE4M3
//     scale), implemented independently in this file — never imp's dequant.
//   * Because the reference and the GPU consume the IDENTICAL quantized bits,
//     quantization error CANCELS: only GEMM accumulation + the f16 output store
//     differ. That justifies the f16-class tolerance below rather than the
//     1e-1 NVFP4 single-op floor.
//   * The grouped GEMM applies the per-expert tensor scale as the epilogue
//     alpha (B's micro-scales carry only the relative scale; A is quantized
//     with no separate tensor scale). The reference reproduces exactly:
//         D[m,n] = alpha_e * sum_k A_dec[m,k] * B_dec[n,k]
//     with alpha_e = B_e.tensor_scale and *_dec computed in fp64.
//
// THE BITS ARE GENUINELY IDENTICAL (not a re-quantization):
//   A: dA_packed[i] / dA_sf[i] are the very buffers passed as host_ptr_A /
//      host_ptr_SFA to gemm_grouped_cutlass_3x_nvfp4. We read them back and
//      decode them; the GPU read the same DRAM.
//   B: cutlass_w.data is BORROWED from the NVFP4 result (RowMajor packed
//      nibbles) and cutlass_w.scale_factors is the SfAtom conversion of the
//      native UE4M3 micro-scales (a value-preserving byte permutation:
//      convert_scales_sfatom drops the always-zero sign bit and re-encodes the
//      identical UE4M3 byte). We decode the SfAtom bytes the GPU actually uses.
//
// TOLERANCE (tests/refs/README.md policy):
//   f16-class accumulation: the GPU accumulates in fp32 and stores fp16; the
//   reference accumulates in fp64. Over a 256-term NVFP4 dot the fp16 OUTPUT
//   rounding (~2^-11 rel) dominates. Asserted <= 1e-2 rel (+ small abs floor
//   for near-zero outputs), MEASURED ~1e-3 (printed per case). This is the
//   tight GEMM/adapter class, an order of magnitude under the NVFP4 1e-1 floor.
//   Plus a hard no-NaN/Inf guard on every output element (the corruption assert,
//   e.g. an empty-expert staging bug must not poison neighbours).
//
// BOUNDARY DISTRIBUTIONS (the staging build must handle these):
//   - an EMPTY expert (M_i = 0)
//   - a 1-token expert (M_i = 1)
//   - a large-M expert that crosses the 128-row SfAtom tile boundary
//   - K = 256 crossing the 64-element atom-K boundary (4 atom-K-tiles)
// =============================================================================

#include <gtest/gtest.h>
#include "compute/gemm_cutlass_grouped_3x.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/tensor.h"
#include "quant/nvfp4_quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <random>

namespace imp {
namespace {

// ---------------------------------------------------------------------------
// Independent NVFP4 decode helpers (from the published format definition).
// Deliberately NOT imp's dequant kernel — this is the reference ground truth.
// ---------------------------------------------------------------------------

// E2M1 magnitude LUT, indexed by the 3-bit code (sign is the 4th nibble bit).
// {0, 0.5, 1, 1.5, 2, 3, 4, 6} — the 8 representable FP4 E2M1 magnitudes.
static double e2m1_code_to_mag(uint8_t code3) {
    static const double lut[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return lut[code3 & 0x7];
}

// Decode one packed byte into its two signed fp64 E2M1 magnitudes (no scale).
// Low nibble = even element, high nibble = odd element.
static void decode_packed_byte(uint8_t byte, double& lo, double& hi) {
    uint8_t n_lo = byte & 0x0F;
    uint8_t n_hi = (byte >> 4) & 0x0F;
    lo = e2m1_code_to_mag(n_lo & 0x7);
    if (n_lo & 0x8)
        lo = -lo;
    hi = e2m1_code_to_mag(n_hi & 0x7);
    if (n_hi & 0x8)
        hi = -hi;
}

// Decode a UE4M3 scale byte to fp64. UE4M3 = unsigned E4M3 (the scale-factor
// dtype): bit pattern is identical to IEEE FP8 E4M3 with the sign forced 0.
//   bias 7; normals 2^(e-7) * (1 + m/8); subnormals (e==0) m/8 * 2^-6.
//   e==15 with m==7 is the e4m3 NaN slot — not produced for scales; treat as 0.
static double ue4m3_to_double(uint8_t b) {
    uint8_t e = (b >> 3) & 0xF;  // 4 exponent bits (sign bit ignored)
    uint8_t m = b & 0x7;         // 3 mantissa bits
    if (e == 0)
        return std::ldexp(static_cast<double>(m) / 8.0, -6);  // subnormal: m/8 * 2^-6
    if (e == 15 && m == 7)
        return 0.0;  // NaN slot — n/a for scales
    return std::ldexp(1.0 + static_cast<double>(m) / 8.0, static_cast<int>(e) - 7);
}

// CUTLASS SfAtom byte offset for logical scale at (row, k_group).
// Reimplemented here from the layout spec (atom 128 rows x 4 k-groups = 512 B,
// K-tiles inner) so the reference does not borrow imp's sfatom_offset().
static int ref_sfatom_offset(int row, int k_group, int n_k_tiles) {
    const int kAtomRows = 128, kAtomKGroups = 4, kAtomSize = 512;
    int tile_row = row / kAtomRows;
    int tile_k = k_group / kAtomKGroups;
    int row_local = row % kAtomRows;
    int k_local = k_group % kAtomKGroups;
    int n0 = row_local % 32;
    int n1 = row_local / 32;
    int atom_offset = n0 * 16 + n1 * 4 + k_local;
    int tile_base = (tile_row * n_k_tiles + tile_k) * kAtomSize;
    return tile_base + atom_offset;
}

static int n_k_tiles_for(int K) {
    const int kAtomKElems = 64;  // 16 * 4
    return (K + kAtomKElems - 1) / kAtomKElems;
}

// Dequantize a [rows, K] NVFP4 operand (packed RowMajor nibbles + SfAtom UE4M3
// micro-scales) to a fp64 dense matrix. tensor_scale is left OUT (deferred to
// the GEMM alpha, exactly as the GPU does).
static std::vector<double> dequant_to_fp64(const std::vector<uint8_t>& packed,  // [rows, K/2]
                                           const std::vector<uint8_t>& sf,      // SfAtom UE4M3
                                           int rows, int K) {
    const int K_groups = K / 16;
    const int n_k_tiles = n_k_tiles_for(K);
    std::vector<double> out(static_cast<size_t>(rows) * K, 0.0);
    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < K_groups; ++g) {
            double ms = ue4m3_to_double(sf[ref_sfatom_offset(r, g, n_k_tiles)]);
            // 16 elements in this group = 8 packed bytes.
            for (int p = 0; p < 8; ++p) {
                int k0 = g * 16 + p * 2;
                uint8_t byte = packed[static_cast<size_t>(r) * (K / 2) + k0 / 2];
                double lo, hi;
                decode_packed_byte(byte, lo, hi);
                out[static_cast<size_t>(r) * K + k0] = lo * ms;
                out[static_cast<size_t>(r) * K + k0 + 1] = hi * ms;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Synthetic expert weights (same driving code as the sibling grouped test).
// ---------------------------------------------------------------------------
struct SyntheticExpert {
    std::vector<half> weight_fp16;  // [N, K] reference weights
    NvFP4QuantResult nvfp4{};
    CutlassNvFP4Weight cutlass_w{};
};

static void make_expert(SyntheticExpert& e, int N, int K, float wscale, uint64_t seed, cudaStream_t stream) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-wscale, wscale);
    e.weight_fp16.resize(static_cast<size_t>(N) * K);
    for (auto& v : e.weight_fp16)
        v = __float2half(dist(gen));

    void* d_w_fp16 = nullptr;
    cudaMalloc(&d_w_fp16, e.weight_fp16.size() * sizeof(half));
    cudaMemcpy(d_w_fp16, e.weight_fp16.data(), e.weight_fp16.size() * sizeof(half), cudaMemcpyHostToDevice);
    int64_t w_shape[2] = {N, K};
    Tensor w_input(d_w_fp16, QType::F16, 2, w_shape, true);
    quantize_fp16_to_nvfp4(w_input, e.nvfp4, stream);
    cudaFree(d_w_fp16);

    convert_nvfp4_to_cutlass(e.nvfp4, e.cutlass_w, stream);
    cudaStreamSynchronize(stream);
}

static void free_expert(SyntheticExpert& e) {
    free_cutlass_nvfp4_weight(e.cutlass_w);
    free_nvfp4_result(e.nvfp4);
}

class CutlassGroupedRefTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&stream_);
        int dev = 0;
        cudaGetDevice(&dev);
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        sm_ = major * 10 + minor;
    }
    void TearDown() override {
        cudaStreamDestroy(stream_);
        gemm_grouped_3x_nvfp4_cleanup();
    }
    cudaStream_t stream_ = nullptr;
    int sm_ = 0;

    // Run the grouped GEMM on the given per-expert token distribution and assert
    // its FP16 output matches the independent fp64 CPU reference.
    void run_case(const std::vector<int>& M_per, int N, int K, uint64_t seed) {
        const int ne = static_cast<int>(M_per.size());
        int M_total = 0;
        for (int m : M_per)
            M_total += m;
        ASSERT_GT(M_total, 0) << "all-empty distribution is meaningless";

        // ----- experts -----
        std::vector<SyntheticExpert> experts(ne);
        for (int i = 0; i < ne; ++i)
            make_expert(experts[i], N, K, /*wscale=*/0.5f, seed + 1000 + i, stream_);

        // ----- FP16 activations [M_total, K] -----
        std::mt19937 agen(static_cast<unsigned>(seed));
        std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
        std::vector<half> h_A(static_cast<size_t>(M_total) * K);
        for (auto& v : h_A)
            v = __float2half(adist(agen));
        void* d_A_fp16 = nullptr;
        cudaMalloc(&d_A_fp16, h_A.size() * sizeof(half));
        cudaMemcpy(d_A_fp16, h_A.data(), h_A.size() * sizeof(half), cudaMemcpyHostToDevice);

        // ----- Per-expert NVFP4 quantization of A -----
        std::vector<void*> dA_packed(ne, nullptr);
        std::vector<void*> dA_sf(ne, nullptr);
        int row_offset = 0;
        for (int i = 0; i < ne; ++i) {
            int M_i = M_per[i];
            if (M_i == 0)
                continue;  // empty expert: no A buffers (offset unchanged)
            size_t packed_bytes = static_cast<size_t>(M_i) * K / 2;
            size_t sfa_bytes = cutlass_nvfp4_sf_size(M_i, K);
            cudaMalloc(&dA_packed[i], packed_bytes);
            cudaMalloc(&dA_sf[i], sfa_bytes);
            const half* a_src = reinterpret_cast<const half*>(d_A_fp16) + static_cast<size_t>(row_offset) * K;
            quantize_fp16_to_nvfp4_cutlass(a_src, dA_packed[i], dA_sf[i], M_i, K, stream_);
            row_offset += M_i;
        }
        cudaStreamSynchronize(stream_);

        // ----- Grouped dispatch -----
        void* d_grp_out = nullptr;
        cudaMalloc(&d_grp_out, static_cast<size_t>(M_total) * N * sizeof(half));
        cudaMemset(d_grp_out, 0, static_cast<size_t>(M_total) * N * sizeof(half));

        std::vector<const void*> hA(ne), hSFA(ne), hB(ne), hSFB(ne);
        std::vector<void*> hD(ne);
        std::vector<float> hAlpha(ne);
        size_t grp_row_off = 0;
        for (int i = 0; i < ne; ++i) {
            hA[i] = dA_packed[i];
            hSFA[i] = dA_sf[i];
            hB[i] = experts[i].cutlass_w.data;
            hSFB[i] = experts[i].cutlass_w.scale_factors;
            hD[i] = reinterpret_cast<half*>(d_grp_out) + grp_row_off * N;
            hAlpha[i] = experts[i].cutlass_w.tensor_scale;
            grp_row_off += M_per[i];
        }

        ASSERT_TRUE(gemm_grouped_cutlass_3x_nvfp4(ne, M_per.data(), N, K, hA.data(), hSFA.data(), hB.data(),
                                                  hSFB.data(), hD.data(), hAlpha.data(), stream_))
            << "grouped dispatch failed";
        cudaStreamSynchronize(stream_);

        std::vector<half> grp_out(static_cast<size_t>(M_total) * N);
        cudaMemcpy(grp_out.data(), d_grp_out, grp_out.size() * sizeof(half), cudaMemcpyDeviceToHost);

        // ----- Read back B bits (per expert) once -----
        const size_t b_packed_bytes = static_cast<size_t>(N) * K / 2;
        const size_t b_sf_bytes = cutlass_nvfp4_sf_size(N, K);

        // ----- Independent fp64 reference, expert by expert -----
        double global_max_rel = 0.0;
        double global_max_abs = 0.0;
        int n_finite_checked = 0;
        size_t out_row = 0;
        for (int e = 0; e < ne; ++e) {
            int M_i = M_per[e];
            if (M_i == 0)
                continue;
            const double alpha = static_cast<double>(experts[e].cutlass_w.tensor_scale);

            // A bits for this expert.
            std::vector<uint8_t> a_packed(static_cast<size_t>(M_i) * K / 2);
            std::vector<uint8_t> a_sf(cutlass_nvfp4_sf_size(M_i, K));
            cudaMemcpy(a_packed.data(), dA_packed[e], a_packed.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(a_sf.data(), dA_sf[e], a_sf.size(), cudaMemcpyDeviceToHost);

            // B bits for this expert (the exact buffers the GEMM reads).
            std::vector<uint8_t> b_packed(b_packed_bytes);
            std::vector<uint8_t> b_sf(b_sf_bytes);
            cudaMemcpy(b_packed.data(), experts[e].cutlass_w.data, b_packed_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(b_sf.data(), experts[e].cutlass_w.scale_factors, b_sf_bytes, cudaMemcpyDeviceToHost);

            std::vector<double> A_dec = dequant_to_fp64(a_packed, a_sf, M_i, K);  // [M_i, K]
            std::vector<double> B_dec = dequant_to_fp64(b_packed, b_sf, N, K);    // [N, K]

            // D = alpha * A * B^T  (B stored [N, K], output [M_i, N]).
            for (int m = 0; m < M_i; ++m) {
                for (int n = 0; n < N; ++n) {
                    double acc = 0.0;
                    const double* a_row = &A_dec[static_cast<size_t>(m) * K];
                    const double* b_row = &B_dec[static_cast<size_t>(n) * K];
                    for (int k = 0; k < K; ++k)
                        acc += a_row[k] * b_row[k];
                    double ref = alpha * acc;

                    float got = __half2float(grp_out[(out_row + m) * N + n]);
                    ASSERT_TRUE(std::isfinite(got))
                        << "non-finite grouped output at expert " << e << " (m=" << m << ", n=" << n << ")";
                    double abs_err = std::fabs(got - ref);
                    double rel = abs_err / (1e-3 + std::fabs(ref));
                    global_max_abs = std::max(global_max_abs, abs_err);
                    global_max_rel = std::max(global_max_rel, rel);
                    ++n_finite_checked;
                    // f16-output accumulation class (+ abs floor for near-zero dots).
                    EXPECT_LT(rel, 1e-2) << "expert " << e << " m=" << m << " n=" << n << " ref=" << ref
                                         << " got=" << got << " abs_err=" << abs_err;
                }
            }
            out_row += M_i;
        }

        std::printf("[grouped-ref] ne=%d N=%d K=%d M_per=[", ne, N, K);
        for (int i = 0; i < ne; ++i)
            std::printf("%s%d", i ? "," : "", M_per[i]);
        std::printf("] checked=%d max_rel=%.3e max_abs=%.3e\n", n_finite_checked, global_max_rel,
                    global_max_abs);

        // ----- Cleanup -----
        for (int i = 0; i < ne; ++i) {
            if (dA_packed[i])
                cudaFree(dA_packed[i]);
            if (dA_sf[i])
                cudaFree(dA_sf[i]);
            free_expert(experts[i]);
        }
        cudaFree(d_A_fp16);
        cudaFree(d_grp_out);
    }
};

// Even token distribution, multiple experts.
TEST_F(CutlassGroupedRefTest, EvenDistribution) {
    if (sm_ < 120)
        GTEST_SKIP() << "SM120 required";
    if (!cutlass_sm120_nvfp4_available())
        GTEST_SKIP() << "CUTLASS NVFP4 disabled";
    if (!cutlass_grouped_3x_nvfp4_available())
        GTEST_SKIP() << "CUTLASS 3x grouped NVFP4 disabled";
    run_case({32, 16, 48, 64}, /*N=*/256, /*K=*/256, /*seed=*/42);
}

// Boundary distribution: an EMPTY expert (M=0), a 1-token expert, and a
// large-M expert that crosses the 128-row SfAtom tile boundary. K=256 spans
// 4 atom-K-tiles. This is the staging build's worst case.
TEST_F(CutlassGroupedRefTest, BoundaryDistribution) {
    if (sm_ < 120)
        GTEST_SKIP() << "SM120 required";
    if (!cutlass_sm120_nvfp4_available())
        GTEST_SKIP() << "CUTLASS NVFP4 disabled";
    if (!cutlass_grouped_3x_nvfp4_available())
        GTEST_SKIP() << "CUTLASS 3x grouped NVFP4 disabled";
    run_case({0, 1, 200, 7}, /*N=*/128, /*K=*/256, /*seed=*/7);
}

// Single active expert (degenerate group) — the dispatch must still produce a
// correct GEMM and not stage zero-length neighbours into the wrong slot.
TEST_F(CutlassGroupedRefTest, SingleActiveExpert) {
    if (sm_ < 120)
        GTEST_SKIP() << "SM120 required";
    if (!cutlass_sm120_nvfp4_available())
        GTEST_SKIP() << "CUTLASS NVFP4 disabled";
    if (!cutlass_grouped_3x_nvfp4_available())
        GTEST_SKIP() << "CUTLASS 3x grouped NVFP4 disabled";
    run_case({0, 33, 0}, /*N=*/192, /*K=*/256, /*seed=*/123);
}

}  // namespace
}  // namespace imp
