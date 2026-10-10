#pragma once
// Host staging for upload_weight (weight_upload.cpp): one traits struct per source format builds
// a StagePlan (host bytes + resulting tensor), commit_plan() does the device side for all of them.
// CUDA-free: the device ops come in through Dev, so tests/test_weight_upload_staging.cpp runs it on CPU.

#include "core/logging.h"
#include "core/qtype.h"
#include "core/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace imp::wupload {

// Host-side FP16<->FP32 conversion: CUDA device intrinsics (__half2float, __float2half)
// aren't available on the host, so these are bitwise conversions.

inline float fp16_to_float(uint16_t h) {
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

inline uint16_t float_to_fp16(float val) {
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

// What one upload does: up to 2 H2D segments, then the tensor it leaves behind.
struct StagePlan {
    enum class Out : uint8_t {
        kRebuild,     // weight = Tensor(seg0, qtype, ndim, shape); seg1 (if any) -> weight.scales
        kKeep,        // weight.data = seg0, on_device = true; every other field kept
        kGpuDequant,  // seg0 raw on device -> dequant to F16 (dequant_bytes) -> free raw
    };
    enum class Log : uint8_t { kNone, kMxfp4Split, kRawRows };

    std::vector<uint8_t> h8;    // byte staging (MXFP4 data+scales, Q4_0 nibbles)
    std::vector<uint16_t> h16;  // FP16 staging (host dequant/convert, Q4_0 scales)
    const void* src[2] = {};
    size_t bytes[2] = {};
    int n_seg = 0;
    Out out = Out::kRebuild;
    Log log = Log::kNone;
    QType qtype = QType::NONE;
    int ndim = 0;
    int64_t shape[kMaxDims] = {};
    size_t dequant_bytes = 0;

    StagePlan() = default;
    StagePlan(const StagePlan&) = delete;  // src[] points into h8/h16
    StagePlan& operator=(const StagePlan&) = delete;

    void add(const void* p, size_t n) {
        src[n_seg] = p;
        bytes[n_seg] = n;
        n_seg++;
    }
    void result(QType q, int nd, const int64_t* sh) {
        qtype = q;
        ndim = nd;
        for (int i = 0; i < nd && i < kMaxDims; i++)
            shape[i] = sh[i];
    }
    void result_2d(QType q, int64_t n, int64_t k) {
        const int64_t sh[2] = {n, k};
        result(q, 2, sh);
    }
};

// Uploaded as-is, tensor fields kept (F16, non-F32 bytes under F32/NONE, opaque payloads).
inline void stage_keep(const Tensor& w, StagePlan& p) {
    p.add(w.data, w.nbytes());
    p.out = StagePlan::Out::kKeep;
}

// Raw quantized rows, dequantized on the fly by the executor. Logical shape [N, K].
inline void stage_rows_raw(const Tensor& w, QType qtype, StagePlan& p) {
    const int64_t N = w.shape[0];
    const int64_t K = w.shape[1];
    p.add(w.data, static_cast<size_t>(N) * qtype_row_bytes(qtype, K));
    p.result_2d(qtype, N, K);
}

// BF16 -> FP16 on host; weight_offset added before the FP16 rounding.
inline void stage_bf16(const Tensor& w, float weight_offset, StagePlan& p) {
    int64_t n_elem = w.numel();
    const uint16_t* src = static_cast<const uint16_t*>(w.data);
    p.h16.resize(static_cast<size_t>(n_elem));
    for (int64_t i = 0; i < n_elem; ++i) {
        // BF16 → float: zero-fill lower mantissa bits
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        f += weight_offset;
        p.h16[i] = float_to_fp16(f);
    }
    p.add(p.h16.data(), static_cast<size_t>(n_elem) * sizeof(uint16_t));
    p.result(QType::F16, w.ndim, w.shape);
}

// Block quant -> FP16 on host, row-major [N, K]: Fmt::decode writes Fmt::kBlockElems values.
template <class Fmt>
void stage_host_dequant(const Tensor& w, StagePlan& p) {
    const int64_t N = w.shape[0];
    const int64_t K = w.shape[1];
    const int blocks_per_row = static_cast<int>(K) / Fmt::kBlockElems;
    p.h16.resize(static_cast<size_t>(N * K));
    const uint8_t* raw = static_cast<const uint8_t*>(w.data);
    for (int64_t n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* block_ptr = raw + (n * blocks_per_row + b) * Fmt::kBlockBytes;
            Fmt::decode(block_ptr, p.h16.data() + n * K + b * Fmt::kBlockElems);
        }
    }
    p.add(p.h16.data(), p.h16.size() * sizeof(uint16_t));
    p.result_2d(QType::F16, N, K);
}

// Shared shape for Q4_0/Q8_0/Q6_K: < 2 dims refused, raw_quant uploads rows, else Fmt::stage_host.
template <class Fmt>
[[nodiscard]] bool stage_block_quant(const Tensor& w, QType qtype, bool raw_quant, StagePlan& p) {
    if (w.ndim < 2) {
        IMP_LOG_WARN("%s weight has < 2 dims, skipping upload", Fmt::kName);
        return false;
    }
    if (raw_quant) {
        stage_rows_raw(w, qtype, p);
        return true;
    }
    Fmt::stage_host(w, qtype, p);
    return true;
}

// MXFP4: [data_0..data_N | scale_0..scale_N]. Legacy type 31 blocks are [data(16)|scale(1)],
// type 39 (w.mxfp4_layout_v2) [scale(1)|data(16)] with SPLIT nibble order, normalized to linear (#551).
struct Mxfp4Fmt {
    static constexpr int kBlockElems = 32;
    static constexpr size_t kBlockBytes = 17;
    static constexpr size_t kDataBytes = 16;

    static void repack_v2(const uint8_t* src, int total_blocks, size_t data_bytes, uint8_t* h) {
        for (int i = 0; i < total_blocks; i++) {
            const uint8_t* qs = src + static_cast<size_t>(i) * kBlockBytes + 1;
            h[data_bytes + i] = src[static_cast<size_t>(i) * kBlockBytes];
            uint8_t* dst = h + static_cast<size_t>(i) * kDataBytes;
            for (int b = 0; b < 16; b++) {
                const int e0 = 2 * b, e1 = 2 * b + 1;
                const uint8_t n0 = (e0 < 16) ? (qs[e0] & 0xF) : (qs[e0 - 16] >> 4);
                const uint8_t n1 = (e1 < 16) ? (qs[e1] & 0xF) : (qs[e1 - 16] >> 4);
                dst[b] = static_cast<uint8_t>(n0 | (n1 << 4));
            }
        }
    }
    static void repack_legacy(const uint8_t* src, int total_blocks, size_t data_bytes, uint8_t* h) {
        for (int i = 0; i < total_blocks; i++) {
            memcpy(h + static_cast<size_t>(i) * kDataBytes, src + static_cast<size_t>(i) * kBlockBytes, kDataBytes);
            h[data_bytes + i] = src[static_cast<size_t>(i) * kBlockBytes + kDataBytes];
        }
    }
    [[nodiscard]] static bool stage(const Tensor& w, QType qtype, bool, float, StagePlan& p) {
        if (w.ndim < 2)
            return false;
        int64_t N = w.shape[0];
        int64_t K = w.shape[1];
        int blocks_per_row = static_cast<int>((K + kBlockElems - 1) / kBlockElems);
        int total_blocks = static_cast<int>(N) * blocks_per_row;
        size_t data_bytes = static_cast<size_t>(N) * blocks_per_row * kDataBytes;
        p.h8.resize(data_bytes + static_cast<size_t>(total_blocks));  // + 1 scale byte per block
        const uint8_t* src = static_cast<const uint8_t*>(w.data);
        if (w.mxfp4_layout_v2)
            repack_v2(src, total_blocks, data_bytes, p.h8.data());
        else
            repack_legacy(src, total_blocks, data_bytes, p.h8.data());
        p.add(p.h8.data(), p.h8.size());
        p.result_2d(qtype, N, K);
        p.log = StagePlan::Log::kMxfp4Split;
        return true;
    }
};

// Q4_0 split (raw_quant=false): packed nibbles [N, K/2] + FP16 scales [N, K/32] -> weight.scales.
struct Q4_0Fmt {
    static constexpr const char* kName = "Q4_0";
    static constexpr int kBlockElems = 32;
    static constexpr size_t kBlockBytes = 18;  // 2 (fp16 scale) + 16 nibbles

    static void stage_host(const Tensor& w, QType qtype, StagePlan& p) {
        int64_t N = w.shape[0];
        int64_t K = w.shape[1];
        int blocks_per_row = static_cast<int>(K) / kBlockElems;
        int half_K = static_cast<int>(K) / 2;
        p.h8.resize(static_cast<size_t>(N) * half_K);
        p.h16.resize(static_cast<size_t>(N) * blocks_per_row);
        const uint8_t* raw = static_cast<const uint8_t*>(w.data);
        for (int64_t n = 0; n < N; ++n) {
            for (int b = 0; b < blocks_per_row; ++b) {
                const uint8_t* block_ptr = raw + (n * blocks_per_row + b) * kBlockBytes;
                std::memcpy(&p.h16[n * blocks_per_row + b], block_ptr, 2);
                std::memcpy(&p.h8[n * half_K + static_cast<int64_t>(b) * 16], block_ptr + 2, 16);
            }
        }
        p.add(p.h8.data(), p.h8.size());
        p.add(p.h16.data(), p.h16.size() * sizeof(uint16_t));
        p.result_2d(qtype, N, half_K);
    }
    [[nodiscard]] static bool stage(const Tensor& w, QType qtype, bool raw_quant, float, StagePlan& p) {
        return stage_block_quant<Q4_0Fmt>(w, qtype, raw_quant, p);
    }
};

struct Q8_0Fmt {
    static constexpr const char* kName = "Q8_0";
    static constexpr int kBlockElems = 32;
    static constexpr size_t kBlockBytes = 34;  // 2 (fp16 scale) + 32 (int8 quants)

    static void decode(const uint8_t* block_ptr, uint16_t* out) {
        uint16_t scale_bits;
        std::memcpy(&scale_bits, block_ptr, 2);
        float scale_f = fp16_to_float(scale_bits);
        const int8_t* quants = reinterpret_cast<const int8_t*>(block_ptr + 2);
        for (int q = 0; q < kBlockElems; ++q)
            out[q] = float_to_fp16(static_cast<float>(quants[q]) * scale_f);
    }
    static void stage_host(const Tensor& w, QType, StagePlan& p) { stage_host_dequant<Q8_0Fmt>(w, p); }
    [[nodiscard]] static bool stage(const Tensor& w, QType qtype, bool raw_quant, float, StagePlan& p) {
        return stage_block_quant<Q8_0Fmt>(w, qtype, raw_quant, p);
    }
};

struct Q6_KFmt {
    static constexpr const char* kName = "Q6_K";
    static constexpr int kBlockElems = 256;
    static constexpr size_t kBlockBytes = 210;  // ql[128] qh[64] scales[16] d(fp16)

    static void decode(const uint8_t* block_ptr, uint16_t* out) {
        const uint8_t* ql = block_ptr;
        const uint8_t* qh = block_ptr + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(block_ptr + 192);
        uint16_t d_bits;
        std::memcpy(&d_bits, block_ptr + 208, 2);
        float d = fp16_to_float(d_bits);
        for (int i = 0; i < kBlockElems; ++i) {
            int group = i / 128;
            int within = i % 128;
            int quad = within / 32;
            int l = within % 32;
            uint8_t ql_byte = ql[group * 64 + (quad & 1) * 32 + l];
            uint8_t low4 = (quad >= 2) ? ((ql_byte >> 4) & 0xF) : (ql_byte & 0xF);
            uint8_t high2 = (qh[group * 32 + l] >> (quad * 2)) & 0x3;
            int q6 = static_cast<int>((high2 << 4) | low4) - 32;
            const int8_t scale = scales[i / 16];  // index math, not a float division
            out[i] = float_to_fp16(d * static_cast<float>(scale) * static_cast<float>(q6));
        }
    }
    static void stage_host(const Tensor& w, QType, StagePlan& p) { stage_host_dequant<Q6_KFmt>(w, p); }
    [[nodiscard]] static bool stage(const Tensor& w, QType qtype, bool raw_quant, float, StagePlan& p) {
        return stage_block_quant<Q6_KFmt>(w, qtype, raw_quant, p);
    }
};

// Any dequant_gpu-supported qtype: raw rows, or raw rows dequantized to F16 on device.
struct GeneralQuantFmt {
    [[nodiscard]] static bool stage(const Tensor& w, QType qtype, bool raw_quant, float, StagePlan& p) {
        stage_rows_raw(w, qtype, p);
        if (raw_quant) {
            p.log = StagePlan::Log::kRawRows;
            return true;
        }
        p.out = StagePlan::Out::kGpuDequant;
        p.dequant_bytes = static_cast<size_t>(w.shape[0]) * w.shape[1] * sizeof(uint16_t);
        p.result(QType::F16, w.ndim, w.shape);
        return true;
    }
};

struct F16Fmt {
    [[nodiscard]] static bool stage(const Tensor& w, QType, bool, float, StagePlan& p) {
        stage_keep(w, p);
        return true;
    }
};

struct Bf16Fmt {
    [[nodiscard]] static bool stage(const Tensor& w, QType, bool, float weight_offset, StagePlan& p) {
        stage_bf16(w, weight_offset, p);
        return true;
    }
};

// qtype F32/NONE: the tensor's own qtype decides. BF16 source (SafeTensors) -> FP16 with offset;
// anything not F32 (e.g. INT8/U8 packed FP4) as-is; F32 -> FP16 without offset.
struct F32Fmt {
    [[nodiscard]] static bool stage(const Tensor& w, QType, bool, float weight_offset, StagePlan& p) {
        if (w.qtype == QType::BF16) {
            stage_bf16(w, weight_offset, p);
            return true;
        }
        // NONE maps to F32 (both are enum value 0)
        if (w.qtype != QType::F32) {
            stage_keep(w, p);
            return true;
        }
        int64_t n_elem = w.numel();
        const float* src = static_cast<const float*>(w.data);
        p.h16.resize(static_cast<size_t>(n_elem));
        for (int64_t i = 0; i < n_elem; ++i)
            p.h16[i] = float_to_fp16(src[i]);
        p.add(p.h16.data(), static_cast<size_t>(n_elem) * sizeof(uint16_t));
        p.result(QType::F16, w.ndim, w.shape);
        return true;
    }
};

// Raw-byte fallback (NVFP4/MXFP4/FP4_E2M1/INT8/INT4 packed payloads), qtype preserved.
struct RawFmt {
    [[nodiscard]] static bool stage(const Tensor& w, QType qtype, bool, float, StagePlan& p) {
        if (w.nbytes() == 0) {
            IMP_LOG_WARN("Empty raw weight for qtype %u, skipping", std::to_underlying(qtype));
            return false;
        }
        stage_keep(w, p);
        return true;
    }
};

// Device side of a StagePlan. Dev: void* alloc(size_t) (nullptr = fail), int copy(dst, src, n)
// (0 = ok, result only logged), release(void*, size_t), track(void*) (-> gpu_allocs),
// dequant(raw, out, qtype, rows, cols) (dequant + sync), const char* err_str(int).
template <class Dev>
[[nodiscard]] bool commit_gpu_dequant(Tensor& w, QType qtype, const StagePlan& p, Dev& dev) {
    void* d_raw = dev.alloc(p.bytes[0]);
    if (!d_raw)
        return false;
    dev.copy(d_raw, p.src[0], p.bytes[0]);
    void* d_fp16 = dev.alloc(p.dequant_bytes);
    if (!d_fp16) {
        dev.release(d_raw, p.bytes[0]);
        return false;
    }
    dev.dequant(d_raw, d_fp16, qtype, static_cast<int>(w.shape[0]), static_cast<int>(w.shape[1]));
    dev.release(d_raw, p.bytes[0]);
    dev.track(d_fp16);
    w = Tensor(d_fp16, p.qtype, p.ndim, p.shape, true);
    return true;
}

template <class Dev>
void log_commit(const Tensor& w, QType qtype, const StagePlan& p, int copy_err, Dev& dev) {
    const long long N = w.shape[0], K = w.shape[1];
    if (p.log == StagePlan::Log::kMxfp4Split) {
        IMP_LOG_DEBUG("  MXFP4 upload: [%lld, %lld] %.2f MiB (data+scales split)", N, K,
                      p.bytes[0] / (1024.0 * 1024.0));
    } else if (p.log == StagePlan::Log::kRawRows) {
        if (copy_err != 0)
            IMP_LOG_ERROR("h2d_copy failed for qtype=%u [%ldx%ld] %zu bytes: %s", (unsigned)qtype, (long)N,
                          (long)K, p.bytes[0], dev.err_str(copy_err));
        IMP_LOG_DEBUG("Upload raw qtype=%u [%ldx%ld] %zu bytes -> GPU %p", (unsigned)qtype, (long)N, (long)K,
                      p.bytes[0], w.data);
    }
}

template <class Dev>
[[nodiscard]] bool commit_plan(Tensor& w, QType qtype, const StagePlan& p, Dev& dev) {
    if (p.out == StagePlan::Out::kGpuDequant)
        return commit_gpu_dequant(w, qtype, p, dev);
    void* d[2] = {};
    int copy_err = 0;
    for (int i = 0; i < p.n_seg; ++i) {
        d[i] = dev.alloc(p.bytes[i]);
        if (!d[i]) {
            for (int j = 0; j < i; ++j)
                dev.release(d[j], p.bytes[j]);
            return false;
        }
        const int e = dev.copy(d[i], p.src[i], p.bytes[i]);
        if (i == 0)
            copy_err = e;
        dev.track(d[i]);
    }
    if (p.out == StagePlan::Out::kKeep) {
        w.data = d[0];
        w.on_device = true;
        return true;
    }
    w = Tensor(d[0], p.qtype, p.ndim, p.shape, true);
    if (p.n_seg > 1)
        w.scales = d[1];
    log_commit(w, qtype, p, copy_err, dev);
    return true;
}

// One upload path for every format: Fmt stages on host, commit_plan does the device side.
template <class Fmt, class Dev>
[[nodiscard]] bool upload_fmt(Tensor& w, QType qtype, bool raw_quant, float weight_offset, Dev& dev) {
    StagePlan p;
    if (!Fmt::stage(w, qtype, raw_quant, weight_offset, p))
        return false;
    return commit_plan(w, qtype, p, dev);
}

// Per-qtype dispatch (upload_weight). Dev::dequant_supported(qtype) = dequant_gpu_supported.
template <class Dev>
[[nodiscard]] bool upload_dispatch(Tensor& w, QType qtype, bool raw_quant, float weight_offset, Dev& dev) {
    switch (qtype) {
    case QType::MXFP4: return upload_fmt<Mxfp4Fmt>(w, qtype, raw_quant, weight_offset, dev);
    case QType::Q4_0: return upload_fmt<Q4_0Fmt>(w, qtype, raw_quant, weight_offset, dev);
    case QType::Q8_0: return upload_fmt<Q8_0Fmt>(w, qtype, raw_quant, weight_offset, dev);
    case QType::Q6_K: return upload_fmt<Q6_KFmt>(w, qtype, raw_quant, weight_offset, dev);
    default: break;
    }
    if (dev.dequant_supported(qtype) && w.ndim >= 2)
        return upload_fmt<GeneralQuantFmt>(w, qtype, raw_quant, weight_offset, dev);
    if (qtype == QType::F16)
        return upload_fmt<F16Fmt>(w, qtype, raw_quant, weight_offset, dev);
    if (qtype == QType::BF16)
        return upload_fmt<Bf16Fmt>(w, qtype, raw_quant, weight_offset, dev);
    if (qtype == QType::F32 || qtype == QType::NONE)
        return upload_fmt<F32Fmt>(w, qtype, raw_quant, weight_offset, dev);
    // Fallback: raw direct upload of opaque bytes (preserves qtype).
    return upload_fmt<RawFmt>(w, qtype, raw_quant, weight_offset, dev);
}

// Per-row FP8 checkpoint head (#2479) the FP8 head GEMV serves: on-device F8_E4M3 [V, D % 256 == 0]
// codes, F32 [V] row scales, F16 final norm. Anything else refuses the load.
[[nodiscard]] inline bool lm_head_row_scales_ok(const Tensor& head, const Tensor& norm,
                                                const Tensor& scales) {
    return head.data && head.on_device && head.qtype == QType::FP8_E4M3 && head.ndim == 2 &&
           head.shape[1] % 256 == 0 && scales.qtype == QType::F32 && scales.ndim == 1 &&
           scales.shape[0] == head.shape[0] && (!norm.data || norm.qtype == QType::F16);
}

}  // namespace imp::wupload
