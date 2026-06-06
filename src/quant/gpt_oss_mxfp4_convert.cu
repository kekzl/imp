#include "quant/gpt_oss_mxfp4_convert.h"
#include "core/logging.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace imp {

bool gpt_oss_convert_experts_to_nvfp4(const uint8_t* h_blocks, const uint8_t* h_scales, int ne,
                                      int64_t n_rows_total, int64_t K, int row_offset, int row_stride,
                                      NvFP4MoEQuantResult& out) {
    const int64_t N = n_rows_total / row_stride;  // rows per expert in the output
    const int64_t kb32 = K / 32;                  // MXFP4 blocks per row
    const int64_t row_bytes = K / 2;              // packed nibbles per row
    const int64_t ms_per_row = K / 16;            // NVFP4 micro-scales per row

    std::vector<uint8_t> packed(static_cast<size_t>(ne) * N * row_bytes);
    std::vector<uint8_t> mscales(static_cast<size_t>(ne) * N * ms_per_row);
    std::vector<float> tscales(ne, 1.0f);

    int clamped_lo = 0, clamped_hi = 0;
    for (int e = 0; e < ne; e++) {
        // Pass 1: per-expert max ue8m0 exponent → tensor scale.
        // tensor_scale = 2^(max_u - 127 - 8): the largest block scale maps to
        // e4m3 value 2^8 = 256 (≤ 448), leaving e4m3's ~18 octaves of range
        // downwards (min subnormal 2^-9 → covers blocks within 2^17 of max).
        int max_u = 0;
        const uint8_t* es_base = h_scales + static_cast<size_t>(e) * n_rows_total * kb32;
        for (int64_t r = 0; r < N; r++) {
            const uint8_t* sr = es_base + static_cast<size_t>(row_offset + r * row_stride) * kb32;
            for (int64_t b = 0; b < kb32; b++)
                max_u = std::max(max_u, static_cast<int>(sr[b]));
        }
        const int ts_exp = max_u - 127 - 8;
        tscales[e] = std::ldexp(1.0f, ts_exp);

        // Pass 2: nibbles copy + scale expansion.
        const uint8_t* eb_base = h_blocks + static_cast<size_t>(e) * n_rows_total * row_bytes;
        uint8_t* op = packed.data() + static_cast<size_t>(e) * N * row_bytes;
        uint8_t* om = mscales.data() + static_cast<size_t>(e) * N * ms_per_row;
        for (int64_t r = 0; r < N; r++) {
            const int64_t src_row = row_offset + r * row_stride;
            std::memcpy(op + r * row_bytes, eb_base + static_cast<size_t>(src_row) * row_bytes,
                        static_cast<size_t>(row_bytes));
            const uint8_t* sr = es_base + static_cast<size_t>(src_row) * kb32;
            uint8_t* mr = om + r * ms_per_row;
            for (int64_t b = 0; b < kb32; b++) {
                // block scale (absolute) = 2^(u-127); stored e4m3 = scale / ts
                float rel = std::ldexp(1.0f, static_cast<int>(sr[b]) - 127 - ts_exp);
                if (rel > 448.0f) {
                    rel = 448.0f;
                    clamped_hi++;
                } else if (rel > 0.0f && rel < 0.001953125f) {  // < 2^-9 (e4m3 min subnormal)
                    rel = 0.001953125f;
                    clamped_lo++;
                }
                __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(rel);
                uint8_t byte;
                std::memcpy(&byte, &f8, 1);
                mr[2 * b] = byte;
                mr[2 * b + 1] = byte;  // one 32-block scale covers two 16-blocks
            }
        }
    }
    if (clamped_lo || clamped_hi)
        IMP_LOG_WARN("gpt-oss MXFP4→NVFP4: %d block scales clamped low, %d high (range loss)", clamped_lo,
                     clamped_hi);

    // Device upload.
    void* d_packed = nullptr;
    void* d_ms = nullptr;
    float* d_ts = nullptr;
    if (cudaMalloc(&d_packed, packed.size()) != cudaSuccess)
        return false;
    if (cudaMalloc(&d_ms, mscales.size()) != cudaSuccess) {
        cudaFree(d_packed);
        return false;
    }
    if (cudaMalloc(&d_ts, sizeof(float) * ne) != cudaSuccess) {
        cudaFree(d_packed);
        cudaFree(d_ms);
        return false;
    }
    if (cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_ms, mscales.data(), mscales.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_ts, tscales.data(), sizeof(float) * ne, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_packed);
        cudaFree(d_ms);
        cudaFree(d_ts);
        return false;
    }

    out.packed_data = d_packed;
    out.micro_scales = d_ms;
    out.tensor_scales = d_ts;
    out.n_experts = ne;
    out.N = N;
    out.K = K;
    out.expert_stride_packed = static_cast<size_t>(N) * row_bytes;
    out.expert_stride_ms = static_cast<size_t>(N) * ms_per_row;
    out.borrowed = false;
    return true;
}

}  // namespace imp
