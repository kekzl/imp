#pragma once

// Independent fp64 reference for NVFP4 x NVFP4 grouped GEMMs: decodes the exact bits the GPU consumes
// (E2M1 nibbles + SfAtom UE4M3 micro-scales) from the published format definition, never imp's dequant.
// tensor_scale stays out (applied as the GEMM alpha, as on the GPU).

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace imp::nvfp4_ref {

// Signed E2M1 value of a 4-bit code (bit 3 = sign): magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
inline double e2m1(uint8_t code4) {
    static const double lut[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double v = lut[code4 & 0x7];
    return (code4 & 0x8) != 0 ? -v : v;
}

// UE4M3: bias 7, normals 2^(e-7)*(1+m/8), subnormals (e==0) m/8*2^-6; the NaN slot (e==15, m==7) reads 0.
inline double ue4m3(uint8_t b) {
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    if (e == 0)
        return std::ldexp(static_cast<double>(m) / 8.0, -6);
    if (e == 15 && m == 7)
        return 0.0;
    return std::ldexp(1.0 + (static_cast<double>(m) / 8.0), e - 7);
}

// CUTLASS SfAtom byte offset of (row, 16-element k group): 128 rows x 4 k groups = 512 B atoms, K tiles
// inner. Mirrors sfatom_offset() in nvfp4_sfatom_quant.cu (a layout permutation, not GEMM math).
inline size_t sfatom_offset(int row, int k_group, int n_k_tiles) {
    const int tile_row = row / 128;
    const int tile_k = k_group / 4;
    const int row_local = row % 128;
    const int atom = ((row_local % 32) * 16) + ((row_local / 32) * 4) + (k_group % 4);
    return (static_cast<size_t>((tile_row * n_k_tiles) + tile_k) * 512) + static_cast<size_t>(atom);
}

// [rows, K] NVFP4 operand (row-major packed nibbles, low nibble = even element; SfAtom scales) -> fp64.
inline std::vector<double> dequant(const uint8_t* packed, const uint8_t* sf, int rows, int K) {
    const int n_k_tiles = (K + 63) / 64;
    std::vector<double> out(static_cast<size_t>(rows) * K, 0.0);
    for (int r = 0; r < rows; ++r)
        for (int g = 0; g < K / 16; ++g) {
            const double s = ue4m3(sf[sfatom_offset(r, g, n_k_tiles)]);
            for (int k = g * 16; k < (g + 1) * 16; ++k) {
                const uint8_t byte = packed[(static_cast<size_t>(r) * (K / 2)) + (k / 2)];
                out[(static_cast<size_t>(r) * K) + k] = e2m1((k & 1) != 0 ? (byte >> 4) : (byte & 0xF)) * s;
            }
        }
    return out;
}

struct GroupedErr {
    double max_rel = 0.0, max_abs = 0.0, max_ref = 0.0;
    size_t checked = 0, bad = 0, nonfinite = 0;
};

// Host copy of a grouped problem: A / SFA slabs at offsets[e] rows / sfa_off[e] bytes, B / SFB per expert.
struct GroupedHost {
    int ne = 0, N = 0, K = 0;
    std::vector<int> offsets;     // [ne+1]
    std::vector<size_t> sfa_off;  // [ne]
    std::vector<uint8_t> a, a_sf;
    std::vector<std::vector<uint8_t>> b, b_sf;
    std::vector<float> alpha;
};

// D_e = alpha_e A_e B_e^T in fp64 against the GPU's fp16 D [offsets[ne], N]. An element is bad when
// |got - ref| / (1e-3 + |ref|) >= rel_tol or got is not finite (f16 output class: rel_tol 1e-2).
inline GroupedErr check_grouped(const GroupedHost& h, const std::vector<half>& d, double rel_tol) {
    GroupedErr r;
    const int N = h.N, K = h.K;
    for (int e = 0; e < h.ne; ++e) {
        const int m_e = h.offsets[e + 1] - h.offsets[e];
        if (m_e == 0)
            continue;
        const std::vector<double> A = dequant(h.a.data() + (static_cast<size_t>(h.offsets[e]) * (K / 2)),
                                              h.a_sf.data() + h.sfa_off[e], m_e, K);
        const std::vector<double> B = dequant(h.b[e].data(), h.b_sf[e].data(), N, K);
        for (int m = 0; m < m_e; ++m)
            for (int n = 0; n < N; ++n) {
                double acc = 0.0;
                const double* ar = &A[static_cast<size_t>(m) * K];
                const double* br = &B[static_cast<size_t>(n) * K];
                for (int k = 0; k < K; ++k)
                    acc += ar[k] * br[k];
                const double ref = static_cast<double>(h.alpha[e]) * acc;
                const double got = __half2float(d[(static_cast<size_t>(h.offsets[e] + m) * N) + n]);
                ++r.checked;
                r.max_ref = std::max(r.max_ref, std::fabs(ref));
                if (!std::isfinite(got)) {
                    ++r.nonfinite;
                    ++r.bad;
                    continue;
                }
                const double abs_err = std::fabs(got - ref);
                const double rel = abs_err / (1e-3 + std::fabs(ref));
                r.max_abs = std::max(r.max_abs, abs_err);
                r.max_rel = std::max(r.max_rel, rel);
                if (rel >= rel_tol) {
                    if (r.bad < 3)
                        std::fprintf(stderr, "  bad: expert %d m=%d n=%d ref=%.6g got=%.6g\n", e, m, n, ref,
                                     got);
                    ++r.bad;
                }
            }
    }
    return r;
}

}  // namespace imp::nvfp4_ref
