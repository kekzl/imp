// weight_upload_traits.h (the one traits-templated upload path, #2212) against the 7+2 per-format
// upload functions it replaced. OLD = verbatim origin/main bodies (3ef8c3fc, src/model/weight_upload.cpp),
// kept only here. Both drive one fake device; every alloc/copy/free/track/dequant and the result
// tensor must match byte for byte, including the alloc/copy failure paths.

#include "model/weight_upload_traits.h"
#include "quant/dequant_gpu.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <deque>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

using imp::QType;
using imp::Tensor;

// One fake device: allocations are host buffers, every op is recorded in order.
struct Event {
    char kind = 0;  // A alloc, C copy, R release, D dequant
    int a = -1, b = -1;
    size_t n = 0, off = 0;
    int q = 0, rows = 0, cols = 0;
    std::vector<uint8_t> payload;
    bool operator==(const Event&) const = default;
};

Event event(char kind, int a, int b = -1, size_t n = 0, size_t off = 0) {
    Event e;
    e.kind = kind;
    e.a = a;
    e.b = b;
    e.n = n;
    e.off = off;
    return e;
}

struct FakeDevice {
    std::deque<std::vector<uint8_t>> bufs;
    std::vector<Event> ev;
    int n_alloc = 0, n_copy = 0;
    int fail_alloc_at = -1, fail_copy_at = -1;

    int index_of(const void* p, size_t* off = nullptr) const {
        const auto* c = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < bufs.size(); i++) {
            const uint8_t* base = bufs[i].data();
            if (c >= base && c < base + bufs[i].size()) {
                if (off)
                    *off = static_cast<size_t>(c - base);
                return static_cast<int>(i);
            }
        }
        return -1;
    }
    void* alloc(size_t n) {
        const int k = n_alloc++;
        ev.push_back(event('A', k, -1, n));
        if (k == fail_alloc_at)
            return nullptr;
        bufs.emplace_back(n + 1, 0xCD);  // +1: a 0-byte alloc still gets a unique address
        return bufs.back().data();
    }
    int copy(void* dst, const void* src, size_t n) {
        const int k = n_copy++;
        size_t off = 0;
        const int idx = index_of(dst, &off);
        Event e = event('C', idx, k, n, off);
        EXPECT_GE(idx, 0) << "copy into unknown device pointer";
        if (idx >= 0) {
            EXPECT_LE(off + n, bufs[idx].size() - 1) << "copy past the allocation";
            if (n > 0 && off + n <= bufs[idx].size() - 1)  // memcpy(_, nullptr, 0) is UB (UBSan)
                std::memcpy(bufs[idx].data() + off, src, n);
        }
        e.payload.assign(static_cast<const uint8_t*>(src), static_cast<const uint8_t*>(src) + n);
        ev.push_back(std::move(e));
        return k == fail_copy_at ? 700 : 0;
    }
    void release(void* d) { ev.push_back(event('R', index_of(d))); }
    std::vector<void*> tracked;  // = gpu_allocs
    void track(void* d) { tracked.push_back(d); }
    void dequant(const void* raw, void* out, QType q, int rows, int cols) {
        Event e = event('D', index_of(raw), index_of(out));
        e.q = static_cast<int>(q);
        e.rows = rows;
        e.cols = cols;
        ev.push_back(std::move(e));
    }
    static bool dequant_supported(QType q) { return imp::dequant_gpu_supported(q); }
    static const char* err_str(int) { return "fake device error"; }
};

FakeDevice* g_old_dev = nullptr;

}  // namespace

// ---- OLD: stubs for the CUDA calls the verbatim bodies make, then the bodies ----
namespace old_impl {
using namespace imp;
using cudaError_t = int;
using cudaStream_t = void*;
constexpr cudaError_t cudaSuccess = 0;
inline const char* cudaGetErrorString(cudaError_t) { return "fake device error"; }
inline cudaError_t checked_cuda_malloc(void** ptr, size_t size, cudaStream_t) {
    *ptr = g_old_dev->alloc(size);
    return *ptr ? cudaSuccess : 2;
}
inline cudaError_t h2d_copy(void* dst, const void* src, size_t n, cudaStream_t) {
    return g_old_dev->copy(dst, src, n);
}
inline cudaError_t cudaFreeAsync(void* d, cudaStream_t) {
    g_old_dev->release(d);
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline void dequant_gpu(const void* src, void* dst, QType qtype, int rows, int cols, cudaStream_t) {
    g_old_dev->dequant(src, dst, qtype, rows, cols);
}
// gpu_allocs: the old bodies push_back directly; compared as the final vector.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
// ==== BEGIN verbatim origin/main src/model/weight_upload.cpp:238-779, 787-809 ====
static float fp16_to_float(uint16_t h) {
    uint16_t sign = (h >> 15) & 1;
    uint16_t exp = (h >> 10) & 0x1F;
    uint16_t man = h & 0x3FF;

    float result;
    if (exp == 0) {
        // Subnormal or zero
        if (man == 0) {
            result = 0.0f;
        } else {
            result = std::ldexp(static_cast<float>(man) / 1024.0f, -14);
        }
    } else if (exp == 31) {
        // Inf or NaN -- clamp to 0 for safety in weight dequant
        result = 0.0f;
    } else {
        result = std::ldexp(1.0f + static_cast<float>(man) / 1024.0f, exp - 15);
    }
    return sign ? -result : result;
}

static uint16_t float_to_fp16(float val) {
    uint32_t fbits;
    std::memcpy(&fbits, &val, 4);
    uint32_t f_sign = (fbits >> 31) & 1;
    int f_exp = static_cast<int>((fbits >> 23) & 0xFF) - 127;
    uint32_t f_man = fbits & 0x7FFFFF;

    // Zero (positive or negative)
    if ((fbits & 0x7FFFFFFF) == 0) {
        return static_cast<uint16_t>(f_sign << 15);
    }

    // Overflow -> Inf
    if (f_exp > 15) {
        return static_cast<uint16_t>((f_sign << 15) | 0x7C00);
    }

    // Underflow -> flush to zero
    if (f_exp < -24) {
        return static_cast<uint16_t>(f_sign << 15);
    }

    // Subnormal in FP16
    if (f_exp < -14) {
        // Convert to subnormal
        int shift = -14 - f_exp;
        uint32_t subnormal_man = (0x800000 | f_man) >> (shift + 13);
        return static_cast<uint16_t>((f_sign << 15) | (subnormal_man & 0x3FF));
    }

    // Normal -- round-to-nearest-even (matching __float2half behavior)
    uint16_t h_exp = static_cast<uint16_t>(f_exp + 15);
    uint32_t round_bit = (f_man >> 12) & 1;  // bit 12 (first discarded bit)
    uint32_t sticky = f_man & 0xFFF;         // bits 11..0 (remaining discarded bits)
    uint16_t h_man = static_cast<uint16_t>(f_man >> 13);
    // Round to nearest even: round up if round_bit=1 AND (sticky!=0 OR lsb=1)
    if (round_bit && (sticky || (h_man & 1))) {
        h_man++;
        if (h_man > 0x3FF) {
            h_man = 0;
            h_exp++;
            if (h_exp > 30) {
                // Overflow to infinity
                return static_cast<uint16_t>((f_sign << 15) | 0x7C00);
            }
        }
    }
    return static_cast<uint16_t>((f_sign << 15) | (h_exp << 10) | h_man);
}

// upload_weight: uploads one weight tensor from host (mmap) to GPU. Q4_0 splits into
// packed_nibbles [N,K/2] + scales [N,K/32] on GPU (dtype->INT4). Q8_0/Q6_K raw_quant=true
// uploads raw bytes (executor dequants on-the-fly before GEMM); raw_quant=false dequants to
// FP16 on host first. F16/BF16 upload direct; F32 converts to FP16 on host. scales_out is
// empty except for Q4_0.


// Per-qtype upload handler extracted from upload_weight: mxfp4 path.
static bool upload_qtype_mxfp4_(Tensor& weight, QType qtype, QType compute_dtype,
                                 cudaStream_t stream, std::vector<void*>& gpu_allocs,
                                 bool raw_quant, float weight_offset) {
    if (weight.ndim < 2)
        return false;
    int64_t N = weight.shape[0];
    int64_t K = weight.shape[1];
    int blocks_per_row = static_cast<int>((K + 31) / 32);
    int total_blocks = static_cast<int>(N) * blocks_per_row;
    size_t data_bytes = static_cast<size_t>(N) * blocks_per_row * 16;  // packed nibbles only

    // CPU-side split: [data_0..data_N | scale_0..scale_N] contiguous layout. Source block
    // layout depends on the GGUF type: legacy (31) is [data(16)|scale(1)] per block, modern (39)
    // is [scale(1)|data(16)] per block (llama.cpp standard). weight.mxfp4_layout_v2 tracks the
    // modern layout (set by gguf_loader).
    size_t scale_bytes = static_cast<size_t>(total_blocks);  // 1 byte per block
    size_t total_bytes = data_bytes + scale_bytes;
    const uint8_t* src = static_cast<const uint8_t*>(weight.data);
    std::vector<uint8_t> h_buf(total_bytes);
    if (weight.mxfp4_layout_v2) {
        // GGML type-39 blocks differ from imp's legacy type 31 in TWO ways: the scale byte leads,
        // AND the nibble order in qs[16] is SPLIT (elem j=low nibble of qs[j], j+16=high), not the
        // LINEAR pair order (2i=low/2i+1=high of byte i) every imp consumer assumes. Copying bytes
        // verbatim permutes all 32 elements of every block (#551, Qwen3.5-4B-mxfp4). Normalize to
        // linear order once, here.
        for (int i = 0; i < total_blocks; i++) {
            const uint8_t* qs = src + static_cast<size_t>(i) * 17 + 1;
            h_buf[data_bytes + i] = src[static_cast<size_t>(i) * 17];
            uint8_t* dst = h_buf.data() + static_cast<size_t>(i) * 16;
            for (int b = 0; b < 16; b++) {
                const int e0 = 2 * b, e1 = 2 * b + 1;
                const uint8_t n0 = (e0 < 16) ? (qs[e0] & 0xF) : (qs[e0 - 16] >> 4);
                const uint8_t n1 = (e1 < 16) ? (qs[e1] & 0xF) : (qs[e1 - 16] >> 4);
                dst[b] = static_cast<uint8_t>(n0 | (n1 << 4));
            }
        }
    } else {
        for (int i = 0; i < total_blocks; i++) {
            memcpy(h_buf.data() + static_cast<size_t>(i) * 16, src + static_cast<size_t>(i) * 17, 16);
            h_buf[data_bytes + i] = src[static_cast<size_t>(i) * 17 + 16];
        }
    }

    void* d_data = nullptr;
    checked_cuda_malloc(&d_data, total_bytes, stream);
    if (!d_data)
        return false;
    h2d_copy(d_data, h_buf.data(), total_bytes, stream);
    gpu_allocs.push_back(d_data);
    int64_t new_shape[4] = {N, K, 0, 0};
    weight = Tensor(d_data, qtype, 2, new_shape, true);
    IMP_LOG_DEBUG("  MXFP4 upload: [%lld, %lld] %.2f MiB (data+scales split)", (long long)N, (long long)K,
                  total_bytes / (1024.0 * 1024.0));
    return true;
}

// Per-qtype upload handler extracted from upload_weight: q4_0 path.
static bool upload_qtype_q4_0_(Tensor& weight, QType qtype, QType compute_dtype,
                                 cudaStream_t stream, std::vector<void*>& gpu_allocs,
                                 bool raw_quant, float weight_offset) {
    if (weight.ndim < 2) {
        IMP_LOG_WARN("Q4_0 weight has < 2 dims, skipping upload");
        return false;
    }

    int64_t N = weight.shape[0];  // out_features (rows)
    int64_t K = weight.shape[1];  // in_features (cols), logical

    // Raw upload: keep quantized bytes on GPU for dp4a GEMV decode path.
    // Prefill uses fp16_cache or on-the-fly dequant_gpu → cuBLAS GEMM.
    if (raw_quant) {
        size_t raw_bytes = static_cast<size_t>(N) * qtype_row_bytes(qtype, K);
        void* d_data = nullptr;
        checked_cuda_malloc(&d_data, raw_bytes, stream);
        if (!d_data)
            return false;
        h2d_copy(d_data, weight.data, raw_bytes, stream);
        gpu_allocs.push_back(d_data);

        // Logical shape [N, K] — qtype tells executor data is raw quantized
        int64_t new_shape[4] = {N, K, 0, 0};
        weight = Tensor(d_data, qtype, 2, new_shape, true);
        return true;
    }

    // Split upload fallback: separate nibbles + scales for quant_gemm_int4.
    int blocks_per_row = static_cast<int>(K) / 32;
    int num_groups = blocks_per_row;
    int half_K = static_cast<int>(K) / 2;

    // GGML Q4_0 block format: 18 bytes per block (2 fp16 scale + 16 nibbles)
    static constexpr size_t Q4_0_BLOCK_SIZE = 18;

    size_t nibbles_bytes = static_cast<size_t>(N) * half_K;
    size_t scales_count = static_cast<size_t>(N) * num_groups;

    std::vector<uint8_t> h_nibbles(nibbles_bytes);
    std::vector<uint16_t> h_scales(scales_count);  // raw FP16 bits

    const uint8_t* raw = static_cast<const uint8_t*>(weight.data);

    for (int64_t n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* block_ptr = raw + (n * blocks_per_row + b) * Q4_0_BLOCK_SIZE;

            // Scale: first 2 bytes (fp16)
            uint16_t scale_bits;
            std::memcpy(&scale_bits, block_ptr, 2);
            h_scales[n * num_groups + b] = scale_bits;

            // Nibbles: next 16 bytes (copied as-is)
            std::memcpy(&h_nibbles[n * half_K + b * 16], block_ptr + 2, 16);
        }
    }

    // Upload packed nibbles to GPU
    void* d_nibbles = nullptr;
    checked_cuda_malloc(&d_nibbles, nibbles_bytes, stream);
    if (!d_nibbles)
        return false;
    h2d_copy(d_nibbles, h_nibbles.data(), nibbles_bytes, stream);
    gpu_allocs.push_back(d_nibbles);

    // Upload scales to GPU
    void* d_scales = nullptr;
    size_t scales_bytes = scales_count * sizeof(uint16_t);
    checked_cuda_malloc(&d_scales, scales_bytes, stream);
    if (!d_scales) {
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_nibbles, stream));
        return false;
    }
    h2d_copy(d_scales, h_scales.data(), scales_bytes, stream);
    gpu_allocs.push_back(d_scales);

    // Update weight tensor to point to packed nibbles on GPU; the
    // FP16 scales buffer rides along as Tensor::scales (sidecar).
    int64_t new_shape[4] = {N, static_cast<int64_t>(half_K), 0, 0};
    weight = Tensor(d_nibbles, qtype, 2, new_shape, true);
    weight.scales = d_scales;

    return true;
}

// Per-qtype upload handler extracted from upload_weight: q8_0 path.
static bool upload_qtype_q8_0_(Tensor& weight, QType qtype, QType compute_dtype,
                                 cudaStream_t stream, std::vector<void*>& gpu_allocs,
                                 bool raw_quant, float weight_offset) {
    if (weight.ndim < 2) {
        IMP_LOG_WARN("Q8_0 weight has < 2 dims, skipping upload");
        return false;
    }

    int64_t N = weight.shape[0];
    int64_t K = weight.shape[1];

    // Raw upload: keep quantized bytes on GPU, dequant on-the-fly in executor
    if (raw_quant) {
        size_t raw_bytes = static_cast<size_t>(N) * qtype_row_bytes(qtype, K);
        void* d_data = nullptr;
        checked_cuda_malloc(&d_data, raw_bytes, stream);
        if (!d_data)
            return false;
        h2d_copy(d_data, weight.data, raw_bytes, stream);
        gpu_allocs.push_back(d_data);

        // Logical shape [N, K] — qtype tells executor data is raw quantized
        int64_t new_shape[4] = {N, K, 0, 0};
        weight = Tensor(d_data, qtype, 2, new_shape, true);
        return true;
    }

    // CPU dequant fallback: decode to FP16 on host, upload
    int blocks_per_row = static_cast<int>(K) / 32;
    static constexpr size_t Q8_0_BLOCK_SIZE = 34;  // 2 (fp16 scale) + 32 (int8 quants)

    size_t fp16_count = static_cast<size_t>(N * K);
    std::vector<uint16_t> h_fp16(fp16_count);

    const uint8_t* raw = static_cast<const uint8_t*>(weight.data);

    for (int64_t n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* block_ptr = raw + (n * blocks_per_row + b) * Q8_0_BLOCK_SIZE;

            uint16_t scale_bits;
            std::memcpy(&scale_bits, block_ptr, 2);
            float scale_f = fp16_to_float(scale_bits);

            const int8_t* quants = reinterpret_cast<const int8_t*>(block_ptr + 2);
            for (int q = 0; q < 32; ++q) {
                float val = static_cast<float>(quants[q]) * scale_f;
                h_fp16[n * K + b * 32 + q] = float_to_fp16(val);
            }
        }
    }

    size_t bytes = fp16_count * sizeof(uint16_t);
    void* d_data = nullptr;
    checked_cuda_malloc(&d_data, bytes, stream);
    if (!d_data)
        return false;
    h2d_copy(d_data, h_fp16.data(), bytes, stream);
    gpu_allocs.push_back(d_data);

    int64_t new_shape[4] = {N, K, 0, 0};
    weight = Tensor(d_data, QType::F16, 2, new_shape, true);
    return true;
}

// Per-qtype upload handler extracted from upload_weight: q6_k path.
static bool upload_qtype_q6_k_(Tensor& weight, QType qtype, QType compute_dtype,
                                 cudaStream_t stream, std::vector<void*>& gpu_allocs,
                                 bool raw_quant, float weight_offset) {
    if (weight.ndim < 2) {
        IMP_LOG_WARN("Q6_K weight has < 2 dims, skipping upload");
        return false;
    }

    int64_t N = weight.shape[0];
    int64_t K = weight.shape[1];

    // Raw upload: keep quantized bytes on GPU, dequant on-the-fly in executor
    if (raw_quant) {
        size_t raw_bytes = static_cast<size_t>(N) * qtype_row_bytes(qtype, K);
        void* d_data = nullptr;
        checked_cuda_malloc(&d_data, raw_bytes, stream);
        if (!d_data)
            return false;
        h2d_copy(d_data, weight.data, raw_bytes, stream);
        gpu_allocs.push_back(d_data);

        int64_t new_shape[4] = {N, K, 0, 0};
        weight = Tensor(d_data, qtype, 2, new_shape, true);
        return true;
    }

    // CPU dequant fallback: decode to FP16 on host, upload
    int blocks_per_row = static_cast<int>(K) / 256;
    static constexpr size_t Q6_K_BLOCK_SIZE = 210;

    size_t fp16_count = static_cast<size_t>(N * K);
    std::vector<uint16_t> h_fp16(fp16_count);

    const uint8_t* raw = static_cast<const uint8_t*>(weight.data);

    for (int64_t n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* block_ptr = raw + (n * blocks_per_row + b) * Q6_K_BLOCK_SIZE;

            const uint8_t* ql = block_ptr;
            const uint8_t* qh = block_ptr + 128;
            const int8_t* scales = reinterpret_cast<const int8_t*>(block_ptr + 192);
            uint16_t d_bits;
            std::memcpy(&d_bits, block_ptr + 208, 2);
            float d = fp16_to_float(d_bits);

            for (int i = 0; i < 256; ++i) {
                int group = i / 128;
                int within = i % 128;
                int quad = within / 32;
                int l = within % 32;

                int ql_idx = group * 64 + (quad & 1) * 32 + l;
                int qh_idx = group * 32 + l;

                uint8_t ql_byte = ql[ql_idx];
                uint8_t low4 = (quad >= 2) ? ((ql_byte >> 4) & 0xF) : (ql_byte & 0xF);
                uint8_t high2 = (qh[qh_idx] >> (quad * 2)) & 0x3;
                int q6 = static_cast<int>((high2 << 4) | low4) - 32;
                float val = d * static_cast<float>(scales[i / 16]) * static_cast<float>(q6);
                h_fp16[n * K + b * 256 + i] = float_to_fp16(val);
            }
        }
    }

    size_t bytes = fp16_count * sizeof(uint16_t);
    void* d_data = nullptr;
    checked_cuda_malloc(&d_data, bytes, stream);
    if (!d_data)
        return false;
    h2d_copy(d_data, h_fp16.data(), bytes, stream);
    gpu_allocs.push_back(d_data);

    int64_t new_shape[4] = {N, K, 0, 0};
    weight = Tensor(d_data, QType::F16, 2, new_shape, true);
    return true;
}

// Per-qtype upload handler extracted from upload_weight: general_quant path.
static bool upload_qtype_general_quant_(Tensor& weight, QType qtype, QType compute_dtype,
                                 cudaStream_t stream, std::vector<void*>& gpu_allocs,
                                 bool raw_quant, float weight_offset) {
    int64_t N = weight.shape[0];
    int64_t K = weight.shape[1];

    if (raw_quant) {
        // Upload raw quantized bytes — executor dequants on-the-fly
        size_t raw_bytes = static_cast<size_t>(N) * qtype_row_bytes(qtype, K);
        void* d_data = nullptr;
        checked_cuda_malloc(&d_data, raw_bytes, stream);
        if (!d_data)
            return false;
        cudaError_t cpy_err = h2d_copy(d_data, weight.data, raw_bytes, stream);
        if (cpy_err != cudaSuccess) {
            IMP_LOG_ERROR("h2d_copy failed for qtype=%u [%ldx%ld] %zu bytes: %s", (unsigned)qtype,
                          (long)N, (long)K, raw_bytes, cudaGetErrorString(cpy_err));
        }
        gpu_allocs.push_back(d_data);
        IMP_LOG_DEBUG("Upload raw qtype=%u [%ldx%ld] %zu bytes -> GPU %p", (unsigned)qtype, (long)N,
                      (long)K, raw_bytes, d_data);
        int64_t new_shape[4] = {N, K, 0, 0};
        weight = Tensor(d_data, qtype, 2, new_shape, true);
        return true;
    } else {
        // Dequant on GPU: upload raw → dequant to FP16 → free raw
        size_t raw_bytes = static_cast<size_t>(N) * qtype_row_bytes(qtype, K);
        void* d_raw = nullptr;
        checked_cuda_malloc(&d_raw, raw_bytes, stream);
        if (!d_raw)
            return false;
        h2d_copy(d_raw, weight.data, raw_bytes, stream);

        size_t fp16_bytes = static_cast<size_t>(N) * K * sizeof(uint16_t);
        void* d_fp16 = nullptr;
        checked_cuda_malloc(&d_fp16, fp16_bytes, stream);
        if (!d_fp16) {
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_raw, stream));
            return false;
        }

        dequant_gpu(d_raw, d_fp16, qtype, static_cast<int>(N), static_cast<int>(K), stream);
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_raw, stream));
        gpu_allocs.push_back(d_fp16);

        weight = Tensor(d_fp16, QType::F16, weight.ndim, weight.shape, true);
        return true;
    }
}

// Per-qtype upload handler extracted from upload_weight: f16 path.
static bool upload_qtype_f16_(Tensor& weight, QType qtype, QType compute_dtype,
                                 cudaStream_t stream, std::vector<void*>& gpu_allocs,
                                 bool raw_quant, float weight_offset) {
    size_t bytes = weight.nbytes();
    void* d_data = nullptr;
    checked_cuda_malloc(&d_data, bytes, stream);
    if (!d_data)
        return false;
    h2d_copy(d_data, weight.data, bytes, stream);
    gpu_allocs.push_back(d_data);

    weight.data = d_data;
    weight.on_device = true;
    return true;
}

// Per-qtype upload handler extracted from upload_weight: bf16 path.
static bool upload_qtype_bf16_(Tensor& weight, QType qtype, QType compute_dtype,
                                 cudaStream_t stream, std::vector<void*>& gpu_allocs,
                                 bool raw_quant, float weight_offset) {
    int64_t n_elem = weight.numel();
    const uint16_t* src = static_cast<const uint16_t*>(weight.data);
    std::vector<uint16_t> h_fp16(static_cast<size_t>(n_elem));
    for (int64_t i = 0; i < n_elem; ++i) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        f += weight_offset;
        h_fp16[i] = float_to_fp16(f);
    }
    size_t bytes = static_cast<size_t>(n_elem) * sizeof(uint16_t);
    void* d_data = nullptr;
    checked_cuda_malloc(&d_data, bytes, stream);
    if (!d_data)
        return false;
    h2d_copy(d_data, h_fp16.data(), bytes, stream);
    gpu_allocs.push_back(d_data);
    weight = Tensor(d_data, QType::F16, weight.ndim, weight.shape, true);
    return true;
}

// Per-qtype upload handler extracted from upload_weight: f32 path.
static bool upload_qtype_f32_(Tensor& weight, QType qtype, QType compute_dtype,
                                 cudaStream_t stream, std::vector<void*>& gpu_allocs,
                                 bool raw_quant, float weight_offset) {
    // BF16 (SafeTensors non-quantized weights): convert to FP16
    if (weight.qtype == QType::BF16) {
        int64_t n_elem = weight.numel();
        const uint16_t* src = static_cast<const uint16_t*>(weight.data);
        std::vector<uint16_t> h_fp16(static_cast<size_t>(n_elem));
        for (int64_t i = 0; i < n_elem; ++i) {
            // BF16 → float: zero-fill lower mantissa bits
            uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(float));
            f += weight_offset;
            h_fp16[i] = float_to_fp16(f);
        }
        size_t bytes = static_cast<size_t>(n_elem) * sizeof(uint16_t);
        void* d_data = nullptr;
        checked_cuda_malloc(&d_data, bytes, stream);
        if (!d_data)
            return false;
        h2d_copy(d_data, h_fp16.data(), bytes, stream);
        gpu_allocs.push_back(d_data);
        weight = Tensor(d_data, QType::F16, weight.ndim, weight.shape, true);
        return true;
    }
    // NONE maps to F32 (both are enum value 0)
    if (weight.qtype != QType::F32) {
        // If it's not actually FP32 data (e.g. INT8/U8 packed FP4), direct upload
        size_t bytes = weight.nbytes();
        void* d_data = nullptr;
        checked_cuda_malloc(&d_data, bytes, stream);
        if (!d_data)
            return false;
        h2d_copy(d_data, weight.data, bytes, stream);
        gpu_allocs.push_back(d_data);
        weight.data = d_data;
        weight.on_device = true;
        return true;
    }

    int64_t n_elem = weight.numel();
    const float* src = static_cast<const float*>(weight.data);
    std::vector<uint16_t> h_fp16(static_cast<size_t>(n_elem));

    for (int64_t i = 0; i < n_elem; ++i) {
        h_fp16[i] = float_to_fp16(src[i]);
    }

    size_t bytes = static_cast<size_t>(n_elem) * sizeof(uint16_t);
    void* d_data = nullptr;
    checked_cuda_malloc(&d_data, bytes, stream);
    if (!d_data)
        return false;
    h2d_copy(d_data, h_fp16.data(), bytes, stream);
    gpu_allocs.push_back(d_data);

    weight = Tensor(d_data, QType::F16, weight.ndim, weight.shape, true);
    return true;
}

// Raw-byte fallback upload (NVFP4/MXFP4/FP4_E2M1/INT8/INT4 packed payloads)
static bool upload_qtype_raw_fallback_(Tensor& weight, QType qtype, cudaStream_t stream,
                                       std::vector<void*>& gpu_allocs) {
    size_t bytes = weight.nbytes();
    if (bytes == 0) {
        IMP_LOG_WARN("Empty raw weight for qtype %u, skipping", std::to_underlying(qtype));
        return false;
    }
    void* d_data = nullptr;
    checked_cuda_malloc(&d_data, bytes, stream);
    if (!d_data)
        return false;
    h2d_copy(d_data, weight.data, bytes, stream);
    gpu_allocs.push_back(d_data);
    weight.data = d_data;
    weight.on_device = true;
    return true;
}

static bool upload_weight_dispatch_(Tensor& weight, QType qtype, QType compute_dtype,
                                    cudaStream_t stream, std::vector<void*>& gpu_allocs,
                                    bool raw_quant, float weight_offset) {
    if (qtype == QType::MXFP4)
        return upload_qtype_mxfp4_(weight, qtype, compute_dtype, stream, gpu_allocs, raw_quant, weight_offset);
    if (qtype == QType::Q4_0)
        return upload_qtype_q4_0_(weight, qtype, compute_dtype, stream, gpu_allocs, raw_quant, weight_offset);
    if (qtype == QType::Q8_0)
        return upload_qtype_q8_0_(weight, qtype, compute_dtype, stream, gpu_allocs, raw_quant, weight_offset);
    if (qtype == QType::Q6_K)
        return upload_qtype_q6_k_(weight, qtype, compute_dtype, stream, gpu_allocs, raw_quant, weight_offset);
    if (dequant_gpu_supported(qtype) && weight.ndim >= 2)
        return upload_qtype_general_quant_(weight, qtype, compute_dtype, stream, gpu_allocs, raw_quant, weight_offset);
    if (qtype == QType::F16)
        return upload_qtype_f16_(weight, qtype, compute_dtype, stream, gpu_allocs, raw_quant, weight_offset);
    if (qtype == QType::BF16)
        return upload_qtype_bf16_(weight, qtype, compute_dtype, stream, gpu_allocs, raw_quant, weight_offset);
    if (qtype == QType::F32 || qtype == QType::NONE)
        return upload_qtype_f32_(weight, qtype, compute_dtype, stream, gpu_allocs, raw_quant, weight_offset);

    // Fallback: raw direct upload of opaque bytes (preserves qtype).
    return upload_qtype_raw_fallback_(weight, qtype, stream, gpu_allocs);
}
// ==== END verbatim ====
#pragma GCC diagnostic pop
}  // namespace old_impl

namespace {

// Everything observable after one upload: device ops, gpu_allocs, the result tensor.
struct Outcome {
    bool ok = false;
    std::vector<Event> ev;
    std::vector<int> tracked;
    std::vector<std::vector<uint8_t>> device;
    int data = -9, scales = -9;  // device alloc index, -2 = unchanged source, -1 = nullptr
    Tensor t;
};

int where(const FakeDevice& d, const void* p, const void* src) {
    if (p == nullptr)
        return -1;
    if (p == src)
        return -2;
    return d.index_of(p);
}

struct Case {
    QType qtype;
    QType tensor_qtype;
    int ndim;
    int64_t shape[4];
    bool raw_quant;
    bool v2;
    float offset;
    int fail_alloc_at;
    int fail_copy_at;
    uint32_t seed;
};

std::string name(const Case& c) {
    std::string s = "qtype=" + std::to_string(static_cast<int>(c.qtype)) +
                    " tensor_qtype=" + std::to_string(static_cast<int>(c.tensor_qtype)) + " shape=[";
    for (int i = 0; i < c.ndim; i++)
        s += std::to_string(c.shape[i]) + (i + 1 < c.ndim ? "," : "");
    return s + "] raw=" + std::to_string(c.raw_quant) + " v2=" + std::to_string(c.v2) +
           " fail_alloc=" + std::to_string(c.fail_alloc_at) + " fail_copy=" + std::to_string(c.fail_copy_at);
}

// Source bytes: enough for every reader (row bytes, 4-byte elements) plus slack.
std::vector<uint8_t> make_source(const Case& c) {
    int64_t numel = 1;
    for (int i = 0; i < c.ndim; i++)
        numel *= c.shape[i];
    size_t n = static_cast<size_t>(numel) * 8 + 4096;
    if (c.ndim >= 2)
        n += static_cast<size_t>(c.shape[0]) * imp::qtype_row_bytes(c.qtype, c.shape[1]) * 2;
    std::vector<uint8_t> v(n);
    std::mt19937 rng(c.seed);
    for (auto& b : v)
        b = static_cast<uint8_t>(rng());
    return v;
}

Tensor make_tensor(const Case& c, std::vector<uint8_t>& src) {
    Tensor t(src.data(), c.tensor_qtype, c.ndim, c.shape, false);
    t.mxfp4_layout_v2 = c.v2;
    t.tensor_scale = 0.5f;
    t.kind = static_cast<imp::TensorKind>(1);
    return t;
}

template <class Run>
Outcome run(const Case& c, Run&& fn) {
    std::vector<uint8_t> src = make_source(c);
    Tensor w = make_tensor(c, src);
    FakeDevice dev;
    dev.fail_alloc_at = c.fail_alloc_at;
    dev.fail_copy_at = c.fail_copy_at;
    Outcome o;
    o.ok = fn(w, dev);
    o.ev = dev.ev;
    for (void* p : dev.tracked)
        o.tracked.push_back(dev.index_of(p));
    for (auto& b : dev.bufs)
        o.device.push_back(b);
    o.data = where(dev, w.data, src.data());
    o.scales = where(dev, w.scales, src.data());
    o.t = w;
    return o;
}

Outcome run_old(const Case& c) {
    return run(c, [&](Tensor& w, FakeDevice& dev) {
        g_old_dev = &dev;
        std::vector<void*> gpu_allocs;
        const bool ok = old_impl::upload_weight_dispatch_(w, c.qtype, QType::F16, nullptr, gpu_allocs, c.raw_quant,
                                                          c.offset);
        dev.tracked = gpu_allocs;
        g_old_dev = nullptr;
        return ok;
    });
}

Outcome run_new(const Case& c) {
    return run(c, [&](Tensor& w, FakeDevice& dev) {
        return imp::wupload::upload_dispatch(w, c.qtype, c.raw_quant, c.offset, dev);
    });
}

void expect_same(const Outcome& a, const Outcome& b, const std::string& what) {
    EXPECT_EQ(a.ok, b.ok) << what;
    ASSERT_EQ(a.ev.size(), b.ev.size()) << what;
    for (size_t i = 0; i < a.ev.size(); i++)
        EXPECT_TRUE(a.ev[i] == b.ev[i]) << what << " event " << i << " kind " << a.ev[i].kind << "/" << b.ev[i].kind
                                        << " n " << a.ev[i].n << "/" << b.ev[i].n;
    EXPECT_EQ(a.tracked, b.tracked) << what;
    EXPECT_TRUE(a.device == b.device) << what;
    EXPECT_EQ(a.data, b.data) << what;
    EXPECT_EQ(a.scales, b.scales) << what;
    EXPECT_EQ(a.t.qtype, b.t.qtype) << what;
    EXPECT_EQ(a.t.ndim, b.t.ndim) << what;
    for (int i = 0; i < imp::kMaxDims; i++) {
        EXPECT_EQ(a.t.shape[i], b.t.shape[i]) << what << " shape " << i;
        EXPECT_EQ(a.t.stride[i], b.t.stride[i]) << what << " stride " << i;
    }
    EXPECT_EQ(a.t.on_device, b.t.on_device) << what;
    EXPECT_EQ(a.t.kind, b.t.kind) << what;
    EXPECT_EQ(a.t.tensor_scale, b.t.tensor_scale) << what;
    EXPECT_EQ(a.t.mxfp4_layout_v2, b.t.mxfp4_layout_v2) << what;
    EXPECT_EQ(a.t.dropped_source, b.t.dropped_source) << what;
}

std::vector<Case> all_cases() {
    // (qtype, tensor qtype): the 7 named formats + F16 + the raw fallback; GENERAL via Q4_K/Q5_K/IQ4_XS.
    const std::vector<std::pair<QType, QType>> fmts = {
        {QType::MXFP4, QType::MXFP4}, {QType::Q4_0, QType::Q4_0},   {QType::Q8_0, QType::Q8_0},
        {QType::Q6_K, QType::Q6_K},   {QType::Q4_K, QType::Q4_K},   {QType::Q5_K, QType::Q5_K},
        {QType::IQ4_XS, QType::IQ4_XS}, {QType::F16, QType::F16},   {QType::BF16, QType::BF16},
        {QType::F32, QType::F32},     {QType::F32, QType::BF16},    {QType::F32, QType::INT8},
        {QType::NONE, QType::F32},    {QType::NONE, QType::BF16},   {QType::NVFP4, QType::NVFP4},
        {QType::INT8, QType::INT8},
    };
    const std::vector<std::vector<int64_t>> shapes = {
        {1, 256}, {3, 512}, {5, 1024}, {2, 96}, {4, 40}, {7}, {2, 3, 256}, {1, 1},
    };
    std::vector<Case> out;
    uint32_t seed = 1;
    for (auto [q, tq] : fmts)
        for (const auto& sh : shapes)
            for (int raw = 0; raw < 2; raw++)
                for (int v2 = 0; v2 < 2; v2++)
                    for (auto [fa, fc] : {std::pair{-1, -1}, std::pair{0, -1}, std::pair{1, -1}, std::pair{-1, 0}}) {
                        Case c{q, tq, static_cast<int>(sh.size()), {0, 0, 0, 0}, raw == 1, v2 == 1,
                               v2 ? 1.0f : 0.0f, fa, fc, seed++};
                        for (size_t i = 0; i < sh.size(); i++)
                            c.shape[i] = sh[i];
                        out.push_back(c);
                    }
    return out;
}

TEST(WeightUploadTraits, MatchesPerFormatUploadersByteForByte) {
    int compared = 0, uploaded = 0;
    size_t bytes = 0;
    for (const Case& c : all_cases()) {
        const Outcome a = run_old(c);
        const Outcome b = run_new(c);
        expect_same(a, b, name(c));
        compared++;
        uploaded += a.ok;
        for (const auto& e : a.ev)
            bytes += e.payload.size();
        if (HasFailure())
            break;
    }
    // Every format uploaded something and the comparison saw real payload bytes.
    EXPECT_GT(uploaded, compared / 4);
    EXPECT_GT(bytes, size_t{1} << 20);
    std::printf("[weight-upload-traits] cases=%d ok=%d payload_bytes=%zu\n", compared, uploaded, bytes);
}

// The oracle is not vacuous: a one-byte source change moves the staged Q8_0 host bytes.
TEST(WeightUploadTraits, ComparisonSeesAOneByteChange) {
    Case c{QType::Q8_0, QType::Q8_0, 2, {2, 64, 0, 0}, false, false, 0.0f, -1, -1, 7};
    const Outcome a = run_new(c);
    std::vector<uint8_t> src = make_source(c);
    src[3] ^= 0x10;  // first block, first quant
    Tensor w = make_tensor(c, src);
    FakeDevice dev;
    ASSERT_TRUE(imp::wupload::upload_dispatch(w, c.qtype, c.raw_quant, c.offset, dev));
    ASSERT_EQ(dev.ev.size(), a.ev.size());
    EXPECT_FALSE(dev.ev[1] == a.ev[1]);
}

}  // namespace
