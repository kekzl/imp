#include "model/model.h"
#include "model/gguf_loader.h"
#include "quant/dequant_gpu.h"
#include "quant/dequant_gptq.h"
#include "core/logging.h"
#include "runtime/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>

#ifdef __linux__
#include <fstream>
#include <string>
#endif

namespace imp {

// ---------------------------------------------------------------------------
// Checked GPU allocation: prevents CUDA memory oversubscription by verifying
// enough free GPU memory exists before allocating.  Without this, cudaMalloc
// on Linux silently succeeds by backing with system RAM (unified memory),
// which causes cuBLASLt INTERNAL_ERROR on the resulting pointers.
// ---------------------------------------------------------------------------
// Cached VRAM state — refreshed once per upload pass instead of per-tensor.
// Eliminates ~500+ cudaMemGetInfo roundtrips during weight upload.
static size_t g_cached_free_mem = 0;
static size_t g_total_allocated = 0;
static size_t g_vram_reserve = 0;  // set from Engine's computed reserve

static cudaError_t checked_cuda_malloc(void** ptr, size_t size, cudaStream_t stream = nullptr) {
    size_t reserve = g_vram_reserve;
    // Use cached free memory (updated at start of each upload pass)
    if (g_cached_free_mem > 0) {
        if (g_total_allocated + size + reserve > g_cached_free_mem) {
            *ptr = nullptr;
            return cudaErrorMemoryAllocation;
        }
        cudaError_t err = cudaMallocAsync(ptr, size, stream);
        if (err == cudaSuccess)
            g_total_allocated += size;
        return err;
    }
    // Fallback: per-tensor check (used outside upload passes)
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (size + reserve > free_mem) {
        *ptr = nullptr;
        return cudaErrorMemoryAllocation;
    }
    return cudaMallocAsync(ptr, size, stream);
}

// ---------------------------------------------------------------------------
// Double-buffered pinned staging for fast H2D transfers.
// On WSL2, mmap'd memory cannot be pinned (cudaHostRegister fails/corrupts),
// so cudaMemcpyAsync from mmap'd memory falls back to synchronous staging
// inside the CUDA driver (~8 GB/s on PCIe 5.0 x16).
// This stager pre-allocates two pinned buffers and pipelines:
//   CPU: memcpy(pinned[i], mmap_data)  ←→  GPU: DMA(gpu, pinned[i^1])
// Achieves true async DMA at full PCIe bandwidth (~25 GB/s on PCIe 5.0).
// ---------------------------------------------------------------------------
struct PinnedStager {
    // Ring of N pinned buffers — deeper pipeline lets CPU memcpy stay ahead of
    // queued DMAs, smoothing per-tensor stalls (each tensor wakes one DMA, ring
    // depth N keeps N-1 DMAs queued while CPU fills the next).
    static constexpr int kRing = 4;
    static constexpr size_t kChunkSize = 128 << 20;  // 128 MiB per buffer (4 × 128 MiB = 512 MiB)
    void* buf[kRing] = {};
    cudaEvent_t done[kRing] = {};
    int idx = 0;

    bool init() {
        for (int i = 0; i < kRing; i++) {
            if (cudaHostAlloc(&buf[i], kChunkSize, cudaHostAllocDefault) != cudaSuccess) {
                destroy();
                return false;
            }
            cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming);
        }
        return true;
    }

    cudaError_t copy(void* dst, const void* src, size_t n, cudaStream_t s) {
        cudaError_t last = cudaSuccess;
        for (size_t off = 0; off < n;) {
            size_t chunk = std::min(n - off, kChunkSize);
            int b = idx % kRing;
            cudaEventSynchronize(done[b]);
            memcpy(buf[b], static_cast<const char*>(src) + off, chunk);
            last = cudaMemcpyAsync(static_cast<char*>(dst) + off, buf[b], chunk, cudaMemcpyHostToDevice, s);
            cudaEventRecord(done[b], s);
            off += chunk;
            idx++;
        }
        return last;
    }

    void destroy() {
        for (int i = 0; i < kRing; i++) {
            if (done[i]) {
                cudaEventSynchronize(done[i]);
                cudaEventDestroy(done[i]);
                done[i] = nullptr;
            }
            if (buf[i]) {
                IMP_CUDA_CHECK_LOG(cudaFreeHost(buf[i]));
                buf[i] = nullptr;
            }
        }
    }
};

// Active stager for current upload pass (nullptr = use plain cudaMemcpyAsync)
static PinnedStager* g_stager = nullptr;

// H2D copy that routes through pinned staging when available
static cudaError_t h2d_copy(void* dst, const void* src, size_t n, cudaStream_t s) {
    if (g_stager)
        return g_stager->copy(dst, src, n, s);
    return cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, s);
}

// ---------------------------------------------------------------------------
// WSL2 detection: cudaHostRegister on mmap'd memory can succeed but produce
// corrupted DMA transfers on WSL2 (stale data from GPU reads).  Detect WSL2
// at runtime so we can skip pinning and fall back to pageable H2D copies.
// ---------------------------------------------------------------------------
static bool is_wsl2() {
#ifdef __linux__
    static int cached = -1;
    if (cached >= 0)
        return cached;
    std::ifstream f("/proc/version");
    if (f) {
        std::string line;
        std::getline(f, line);
        cached = (line.find("microsoft") != std::string::npos ||
                  line.find("Microsoft") != std::string::npos || line.find("WSL") != std::string::npos)
                     ? 1
                     : 0;
    } else {
        cached = 0;
    }
    return cached;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Host-side FP16 <-> FP32 conversion helpers.
// We cannot use CUDA device intrinsics (__half2float, __float2half) on the
// host, so we implement bitwise conversions.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// upload_weight: upload a single weight tensor from host (mmap) to GPU.
//
// For Q4_0: splits into packed_nibbles [N, K/2] + scales [N, K/32] on GPU.
//           Updates weight tensor to point to packed_nibbles (dtype=INT4),
//           fills scales_out tensor.
// For Q8_0/Q6_K (raw_quant=true): uploads raw quantized bytes to GPU.
//           Executor dequants on-the-fly into a scratch buffer before GEMM.
// For Q8_0/Q6_K (raw_quant=false): dequants to FP16 on host, uploads as FP16.
// For F16/BF16: direct upload. scales_out stays empty.
// For F32: converts to FP16 on host, uploads. scales_out stays empty.
// ---------------------------------------------------------------------------


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

    // CPU-side split: [data_0..data_N | scale_0..scale_N] contiguous layout.
    // Source block layout depends on the originating GGUF type:
    //   legacy (type 31): [data (16) | scale (1)] per block
    //   modern (type 39): [scale (1) | data (16)] per block (llama.cpp standard)
    // weight.mxfp4_layout_v2 tracks the modern layout (set by gguf_loader).
    size_t scale_bytes = static_cast<size_t>(total_blocks);  // 1 byte per block
    size_t total_bytes = data_bytes + scale_bytes;
    const uint8_t* src = static_cast<const uint8_t*>(weight.data);
    std::vector<uint8_t> h_buf(total_bytes);
    if (weight.mxfp4_layout_v2) {
        for (int i = 0; i < total_blocks; i++) {
            h_buf[data_bytes + i] = src[static_cast<size_t>(i) * 17];
            memcpy(h_buf.data() + static_cast<size_t>(i) * 16, src + static_cast<size_t>(i) * 17 + 1, 16);
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
        IMP_LOG_WARN("Empty raw weight for qtype %u, skipping", static_cast<unsigned>(qtype));
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

static bool upload_weight(Tensor& weight, QType qtype, QType compute_dtype, cudaStream_t stream,
                          std::vector<void*>& gpu_allocs, bool raw_quant = true, float weight_offset = 0.0f) {
    // weight_offset: added to each FP32 element BEFORE FP16 conversion. Only
    // applied on BF16-source paths (qtype==BF16 or qtype==F32/NONE with
    // weight.qtype==BF16). F32-source paths leave it unused — GGUF norms
    // already carry the offset baked in by the converter; SafeTensors stores
    // the delta `W` (where actual gamma = 1 + W) for Qwen3.5/3.6 block norms.
    if (weight.data == nullptr || weight.on_device)
        return true;
    if (weight.ndim < 1)
        return true;

    int64_t n_elements = weight.numel();
    if (n_elements == 0)
        return true;

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

// ---------------------------------------------------------------------------
// Helper: upload a weight tensor that has no associated quant type
// (e.g., norm weights, embedding). We detect the dtype from the tensor.
// ---------------------------------------------------------------------------

static bool upload_unquantized_weight(Tensor& weight, QType qtype, QType compute_dtype, cudaStream_t stream,
                                      std::vector<void*>& gpu_allocs, bool raw_quant = true) {
    return upload_weight(weight, qtype, compute_dtype, stream, gpu_allocs, raw_quant);
}

// ---------------------------------------------------------------------------
// Model::estimate_expert_bytes
// ---------------------------------------------------------------------------

size_t Model::estimate_expert_bytes() const {
    size_t total = 0;
    for (int i = 0; i < n_layers(); ++i) {
        const TransformerLayer& L = layers_[i];
        auto add_packed = [&](const Tensor& p, QType qt) {
            if (!p.data || p.ndim < 3 || !dequant_gpu_supported(qt))
                return;
            size_t row_bytes = qtype_row_bytes(qt, p.shape[2]);
            total += static_cast<size_t>(p.shape[0]) * p.shape[1] * row_bytes;
        };
        add_packed(L.expert_gate_packed, L.expert_gate_packed.qtype);
        add_packed(L.expert_up_packed, L.expert_up_packed.qtype);
        add_packed(L.expert_down_packed, L.expert_down_packed.qtype);
    }
    return total;
}

// ---------------------------------------------------------------------------
// Upload context: bundles the repeated parameters needed by all upload helpers.
// Passed by reference to avoid >8 params on every helper call.
// ---------------------------------------------------------------------------
struct UploadCtx {
    QType compute_dtype;
    cudaStream_t stream;
    std::vector<void*>& gpu_allocs;
    std::vector<void*>& host_pinned;
    std::vector<void*>& host_pinned_allocs;
    // Architecture-specific norm-weight offset. Qwen3.5/3.6 SafeTensors stores
    // block-norm gammas as deltas (gamma = 1 + W) while GGUF bakes the +1 in
    // at conversion time. Applied only on BF16-source paths in upload_weight().
    // Set to 1.0f for QWEN35[_MOE]/QWEN36_MOE, 0.0f otherwise.
    float arch_norm_offset = 0.0f;
};

// ---------------------------------------------------------------------------
// UPLOAD_OR_FAIL / UPLOAD_UNQUANT_OR_FAIL: reduces the per-weight boilerplate
// of calling upload_weight() + error log + early return.
// ---------------------------------------------------------------------------
#define UPLOAD_OR_FAIL(tensor, qtype, msg, layer_idx, ctx)                                            \
    do {                                                                                              \
        if (!upload_weight((tensor), (qtype), (ctx).compute_dtype, (ctx).stream, (ctx).gpu_allocs)) { \
            IMP_LOG_ERROR("Failed to upload " msg " for layer %d", (layer_idx));                      \
            return false;                                                                             \
        }                                                                                             \
    } while (0)

#define UPLOAD_OR_FAIL_RAW(tensor, qtype, raw, msg, layer_idx, ctx)                                          \
    do {                                                                                                     \
        if (!upload_weight((tensor), (qtype), (ctx).compute_dtype, (ctx).stream, (ctx).gpu_allocs, (raw))) { \
            IMP_LOG_ERROR("Failed to upload " msg " for layer %d", (layer_idx));                             \
            return false;                                                                                    \
        }                                                                                                    \
    } while (0)

#define UPLOAD_UNQUANT_OR_FAIL(tensor, msg, layer_idx, ctx)                                          \
    do {                                                                                             \
        if ((tensor).data && !(tensor).on_device) {                                                  \
            if (!upload_unquantized_weight((tensor), QType::NONE, (ctx).compute_dtype, (ctx).stream, \
                                           (ctx).gpu_allocs)) {                                      \
                IMP_LOG_ERROR("Failed to upload " msg " for layer %d", (layer_idx));                 \
                return false;                                                                        \
            }                                                                                        \
        }                                                                                            \
    } while (0)

// ---------------------------------------------------------------------------
// upload_embeddings_and_output: token embedding, output norm, output projection
// ---------------------------------------------------------------------------
static bool upload_embeddings_and_output(Tensor& tok_emb, Tensor& out_norm, Tensor& out_proj,
                                         const UploadCtx& ctx) {
    // Upload token embedding
    // Embedding lookup only supports Q8_0/Q6_K natively; other quant types
    // need to be dequanted to FP16 (raw_quant=false) so the standard FP16
    // embedding gather works. tok_emb.qtype is updated in-place by
    // upload_weight if a host-side dequant occurs.
    const void* tok_emb_host_ptr = tok_emb.data;  // save for weight-tying check below
    const QType tok_emb_orig_qtype = tok_emb.qtype;
    if (tok_emb.data && !tok_emb.on_device) {
        const bool emb_raw = (tok_emb.qtype == QType::Q8_0 || tok_emb.qtype == QType::Q6_K);
        if (!upload_unquantized_weight(tok_emb, tok_emb.qtype, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs,
                                       emb_raw)) {
            IMP_LOG_ERROR("Failed to upload token embedding");
            return false;
        }
    }

    // Upload output norm
    if (out_norm.data && !out_norm.on_device) {
        if (!upload_unquantized_weight(out_norm, out_norm.qtype, ctx.compute_dtype, ctx.stream,
                                       ctx.gpu_allocs)) {
            IMP_LOG_ERROR("Failed to upload output norm");
            return false;
        }
    }

    // Upload output projection — raw Q6_K/Q8_0 for dp4a GEMV (saves ~60% VRAM).
    // Falls back to FP16 dequant for unsupported quant types.
    // For weight-tied models (out_proj == tok_emb), share the GPU data directly.
    if (out_proj.data && !out_proj.on_device) {
        // Weight tying: share GPU data only if both point to the same host tensor
        // (i.e. GGUF had no output.weight and the loader aliased out_proj = tok_emb).
        // Compare against the ORIGINAL pre-upload qtype since out_proj still holds it.
        const bool actually_tied = (out_proj.data == tok_emb_host_ptr &&
                                    out_proj.qtype == tok_emb_orig_qtype);
        if (actually_tied && tok_emb.on_device) {
            out_proj = tok_emb;
            IMP_LOG_INFO("Output projection shares GPU data with token embedding (weight tying)");
        } else {
            const bool raw_ok = (out_proj.qtype == QType::Q6_K || out_proj.qtype == QType::Q8_0 ||
                                 out_proj.qtype == QType::Q4_0);
            if (!upload_unquantized_weight(out_proj, out_proj.qtype, ctx.compute_dtype, ctx.stream,
                                           ctx.gpu_allocs,
                                           /*raw_quant=*/raw_ok)) {
                IMP_LOG_ERROR("Failed to upload output projection");
                return false;
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// upload_mtp_weights: BF16 → FP16 upload of the MTP head (Phase 2 prereq).
// Walks all 19 named tensors in MtpHead and uploads each via upload_weight().
// All tensors are stored BF16 on disk and run as FP16 on GPU. The MoE expert
// tensors (3D [n_experts, ...]) are uploaded raw as 3D FP16 — slicing per-
// expert is the forward kernel's concern (Phase 2 compute).
//
// CRITICAL — norm-weight offset: Qwen3.5/3.6 SafeTensors stores RMSNorm
// gammas as deltas `W` (actual gamma = 1 + W). Without ctx.arch_norm_offset
// applied during BF16→FP16, MTP norms run with scale ≈ 0 instead of ≈ 1,
// producing zero output that locks the LM-head argmax to a deterministic
// noise token. Pass the offset on norm tensors only.
// ---------------------------------------------------------------------------
static bool upload_mtp_weights(MtpHead& head, const UploadCtx& ctx) {
    if (!head.loaded) return true;  // nothing to upload

    // Norm uploader: applies arch_norm_offset for Qwen3.5/3.6's `gamma = 1 + W`
    // convention. Non-norm tensors use the no-offset path.
    auto up_norm = [&](Tensor& t, const char* name) -> bool {
        if (t.data == nullptr || t.on_device)
            return true;
        if (!upload_weight(t, t.qtype, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs,
                           /*raw_quant=*/true, ctx.arch_norm_offset)) {
            IMP_LOG_ERROR("Failed to upload MTP norm: %s", name);
            return false;
        }
        return true;
    };
    auto up = [&](Tensor& t, const char* name) -> bool {
        if (t.data == nullptr || t.on_device)
            return true;
        if (!upload_unquantized_weight(t, t.qtype, ctx.compute_dtype,
                                       ctx.stream, ctx.gpu_allocs)) {
            IMP_LOG_ERROR("Failed to upload MTP tensor: %s", name);
            return false;
        }
        return true;
    };

    bool ok = true;
    // Norm weights — REQUIRE arch_norm_offset for Qwen3.5/3.6
    ok &= up_norm(head.pre_fc_norm_embedding,    "pre_fc_norm_embedding");
    ok &= up_norm(head.pre_fc_norm_hidden,       "pre_fc_norm_hidden");
    ok &= up_norm(head.input_layernorm,          "input_layernorm");
    ok &= up_norm(head.post_attention_layernorm, "post_attention_layernorm");
    ok &= up_norm(head.q_norm,                   "q_norm");
    ok &= up_norm(head.k_norm,                   "k_norm");
    ok &= up_norm(head.final_norm,               "final_norm");
    // Projection / MoE weights — no offset
    ok &= up(head.fc,                        "fc");
    ok &= up(head.q_proj,                   "q_proj");
    ok &= up(head.k_proj,                   "k_proj");
    ok &= up(head.v_proj,                   "v_proj");
    ok &= up(head.o_proj,                   "o_proj");
    ok &= up(head.router,                   "router");
    ok &= up(head.experts_gate_up_packed,   "experts_gate_up_packed");
    ok &= up(head.experts_down_packed,      "experts_down_packed");
    ok &= up(head.shared_expert_gate_proj,  "shared_expert_gate_proj");
    ok &= up(head.shared_expert_up_proj,    "shared_expert_up_proj");
    ok &= up(head.shared_expert_down_proj,  "shared_expert_down_proj");
    ok &= up(head.shared_expert_gate,       "shared_expert_gate");
    return ok;
}

// ---------------------------------------------------------------------------
// upload_gptq_weight: dequantize a GPTQ-packed weight to FP16 on GPU.
// Uploads qweight/qzeros/scales/g_idx to temporary GPU buffers, runs the
// dequant kernel, then frees the temporaries.  Sets output tensor to point
// to the resulting FP16 weight on GPU.
// ---------------------------------------------------------------------------
static bool upload_gptq_weight(const TransformerLayer::GPTQWeight& gptq, Tensor& output, cudaStream_t stream,
                               std::vector<void*>& gpu_allocs) {
    if (!gptq.qweight.data || !gptq.scales.data)
        return false;
    if (gptq.bits != 4) {
        IMP_LOG_ERROR("GPTQ: only 4-bit supported (got %d)", gptq.bits);
        return false;
    }

    // qweight shape: [K/8, N] for 4-bit (8 values packed per INT32)
    int pack_factor = 32 / gptq.bits;  // 8 for 4-bit
    int K_packed = static_cast<int>(gptq.qweight.shape[0]);
    int N = static_cast<int>(gptq.qweight.shape[1]);
    int K = K_packed * pack_factor;

    // 1. Upload qweight to GPU
    size_t qw_bytes = static_cast<size_t>(K_packed) * N * sizeof(int32_t);
    int32_t* d_qweight = nullptr;
    if (checked_cuda_malloc(reinterpret_cast<void**>(&d_qweight), qw_bytes, stream) != cudaSuccess || !d_qweight) {
        IMP_LOG_ERROR("GPTQ: failed to allocate qweight (%zu bytes)", qw_bytes);
        return false;
    }
    h2d_copy(d_qweight, gptq.qweight.data, qw_bytes, stream);

    // 2. Upload qzeros to GPU
    int32_t* d_qzeros = nullptr;
    if (gptq.qzeros.data) {
        size_t qz_bytes = static_cast<size_t>(gptq.qzeros.shape[0]) * gptq.qzeros.shape[1] * sizeof(int32_t);
        if (checked_cuda_malloc(reinterpret_cast<void**>(&d_qzeros), qz_bytes, stream) != cudaSuccess || !d_qzeros) {
            IMP_LOG_ERROR("GPTQ: failed to allocate qzeros");
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qweight, stream));
            return false;
        }
        h2d_copy(d_qzeros, gptq.qzeros.data, qz_bytes, stream);
    }

    // 3. Upload scales to GPU
    size_t sc_bytes = static_cast<size_t>(gptq.scales.shape[0]) * gptq.scales.shape[1] * sizeof(half);
    half* d_scales = nullptr;
    if (checked_cuda_malloc(reinterpret_cast<void**>(&d_scales), sc_bytes, stream) != cudaSuccess || !d_scales) {
        IMP_LOG_ERROR("GPTQ: failed to allocate scales");
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qweight, stream));
        if (d_qzeros)
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qzeros, stream));
        return false;
    }
    h2d_copy(d_scales, gptq.scales.data, sc_bytes, stream);

    // 4. Upload g_idx to GPU (optional, for desc_act reordering)
    int32_t* d_g_idx = nullptr;
    if (gptq.g_idx.data) {
        size_t gi_bytes = static_cast<size_t>(K) * sizeof(int32_t);
        if (checked_cuda_malloc(reinterpret_cast<void**>(&d_g_idx), gi_bytes, stream) != cudaSuccess || !d_g_idx) {
            IMP_LOG_WARN("GPTQ: failed to allocate g_idx, falling back to sequential groups");
        } else {
            h2d_copy(d_g_idx, gptq.g_idx.data, gi_bytes, stream);
        }
    } else if (gptq.desc_act) {
        // Activation-reordered model export but no g_idx tensor → kernel will
        // run sequential grouping, which silently produces wrong outputs. Warn
        // once per process so this isn't lost in the per-layer upload spam.
        static bool warned = false;
        if (!warned) {
            warned = true;
            IMP_LOG_WARN(
                "GPTQ: config declares desc_act=true but g_idx tensor is "
                "absent. Dequant will use sequential grouping; if the model "
                "was exported with activation reordering, outputs will be "
                "incorrect. (Logged once.)");
        }
    }

    // 5. Allocate FP16 output [N, K]
    size_t out_bytes = static_cast<size_t>(N) * K * sizeof(half);
    half* d_out = nullptr;
    if (checked_cuda_malloc(reinterpret_cast<void**>(&d_out), out_bytes, stream) != cudaSuccess || !d_out) {
        IMP_LOG_ERROR("GPTQ: failed to allocate output (%zu bytes)", out_bytes);
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qweight, stream));
        if (d_qzeros)
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qzeros, stream));
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_scales, stream));
        if (d_g_idx)
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_g_idx, stream));
        return false;
    }

    // 6. Run dequantization kernel
    dequant_gptq4(d_out, d_qweight, d_qzeros, d_scales, d_g_idx, N, K, gptq.group_size, stream);

    // 7. Sync and free temporary GPU buffers
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qweight, stream));
    if (d_qzeros)
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_qzeros, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_scales, stream));
    if (d_g_idx)
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_g_idx, stream));

    // 8. Set output tensor
    int64_t out_shape[4] = {N, K, 0, 0};
    output = Tensor(d_out, QType::F16, 2, out_shape, true);
    gpu_allocs.push_back(d_out);

    return true;
}

// ---------------------------------------------------------------------------
// upload_layer_attention_weights: wq/wk/wv/wo + norms + biases for one layer
// ---------------------------------------------------------------------------
static bool upload_layer_attention_weights(TransformerLayer& L, int i, const UploadCtx& ctx) {
    // Attention weights — try regular upload first, fall back to GPTQ dequant
    UPLOAD_OR_FAIL(L.wq, L.wq.qtype, "wq", i, ctx);
    UPLOAD_OR_FAIL(L.wk, L.wk.qtype, "wk", i, ctx);
    UPLOAD_OR_FAIL(L.wv, L.wv.qtype, "wv", i, ctx);
    UPLOAD_OR_FAIL(L.wo, L.wo.qtype, "wo", i, ctx);

    // GPTQ fallback: if regular weight is missing but GPTQ tensors are present
    struct {
        Tensor& w;
        TransformerLayer::GPTQWeight& gptq;
        const char* name;
    } attn_gptq[] = {
        {L.wq, L.gptq_q, "q_proj"},
        {L.wk, L.gptq_k, "k_proj"},
        {L.wv, L.gptq_v, "v_proj"},
        {L.wo, L.gptq_o, "o_proj"},
    };
    for (auto& [w, gptq, name] : attn_gptq) {
        if (!w.on_device && gptq.qweight.data) {
            if (!upload_gptq_weight(gptq, w, ctx.stream, ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to dequant GPTQ %s for layer %d", name, i);
                return false;
            }
            IMP_LOG_DEBUG("GPTQ dequant %s layer %d -> [%lld, %lld] FP16", name, i, w.shape[0], w.shape[1]);
        }
    }

    // Attention norm (typically F32/F16, no quant). For Qwen3.5/3.6
    // SafeTensors, the BF16 weight stores `W` (delta from 1.0) and the actual
    // gamma is `1 + W`; ctx.arch_norm_offset bakes that +1 in during BF16→FP16
    // conversion. GGUF F32 norms already carry the offset and are unaffected.
    if (L.attn_norm.data && !L.attn_norm.on_device) {
        if (!upload_weight(L.attn_norm, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs, true,
                           ctx.arch_norm_offset)) {
            IMP_LOG_ERROR("Failed to upload attn_norm for layer %d", i);
            return false;
        }
    }

    // QK-norm weights (Qwen3-style per-head RMSNorm, F32 [head_dim]).
    // Qwen3.5/3.6 also use the `1 + W` convention here.
    if (L.attn_q_norm.data && !L.attn_q_norm.on_device) {
        if (!upload_weight(L.attn_q_norm, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs, true,
                           ctx.arch_norm_offset)) {
            IMP_LOG_ERROR("Failed to upload attn_q_norm for layer %d", i);
            return false;
        }
    }
    if (L.attn_k_norm.data && !L.attn_k_norm.on_device) {
        if (!upload_weight(L.attn_k_norm, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs, true,
                           ctx.arch_norm_offset)) {
            IMP_LOG_ERROR("Failed to upload attn_k_norm for layer %d", i);
            return false;
        }
    }

    // Attention biases (Qwen2-style Q/K/V biases, F32)
    for (auto* bias : {&L.q_bias, &L.k_bias, &L.v_bias}) {
        if (bias->data && !bias->on_device) {
            if (!upload_unquantized_weight(*bias, QType::NONE, ctx.compute_dtype, ctx.stream,
                                           ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload attention bias for layer %d", i);
                return false;
            }
        }
    }

    // Post-layer norms (Gemma-3/4)
    for (auto* norm : {&L.post_attn_norm, &L.post_ffn_norm, &L.ffn_pre_norm_2, &L.ffn_post_norm_1,
                       &L.ffn_post_norm_2, &L.ffn_gate_inp_scale, &L.layer_out_scale, &L.expert_down_scale}) {
        if (norm->data && !norm->on_device) {
            if (!upload_unquantized_weight(*norm, QType::NONE, ctx.compute_dtype, ctx.stream,
                                           ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload post-layer norm for layer %d", i);
                return false;
            }
        }
    }

    // rope_freqs: upload as raw FP32 (NOT converted to FP16).
    // The RoPE kernel reads these as float* — FP16 conversion would corrupt them.
    if (L.rope_freqs.data && !L.rope_freqs.on_device && L.rope_freqs.qtype == QType::F32) {
        IMP_LOG_INFO("Layer %d: uploading rope_freqs as raw FP32 (%lld elements)", i, L.rope_freqs.numel());
        size_t bytes = L.rope_freqs.nbytes();
        void* d_data = nullptr;
        IMP_CUDA_CHECK_LOG(cudaMallocAsync(&d_data, bytes, ctx.stream));
        if (!d_data) {
            IMP_LOG_ERROR("Failed to allocate GPU memory for rope_freqs layer %d", i);
            return false;
        }
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(d_data, L.rope_freqs.data, bytes, cudaMemcpyHostToDevice, ctx.stream));
        ctx.gpu_allocs.push_back(d_data);
        L.rope_freqs.data = d_data;
        L.rope_freqs.on_device = true;
    }

    return true;
}

// ---------------------------------------------------------------------------
// upload_layer_ffn_weights: w_gate/w_up/w_down + norms + MoE routing +
//                           shared experts for one layer
// ---------------------------------------------------------------------------
static bool upload_layer_ffn_weights(TransformerLayer& L, int i, const UploadCtx& ctx) {
    // FFN weights (dense path)
    UPLOAD_OR_FAIL(L.w_gate, L.w_gate.qtype, "w_gate", i, ctx);
    UPLOAD_OR_FAIL(L.w_up, L.w_up.qtype, "w_up", i, ctx);
    UPLOAD_OR_FAIL(L.w_down, L.w_down.qtype, "w_down", i, ctx);

    // GPTQ fallback for FFN weights
    struct {
        Tensor& w;
        TransformerLayer::GPTQWeight& gptq;
        const char* name;
    } ffn_gptq[] = {
        {L.w_gate, L.gptq_gate, "gate_proj"},
        {L.w_up, L.gptq_up, "up_proj"},
        {L.w_down, L.gptq_down, "down_proj"},
    };
    for (auto& [w, gptq, name] : ffn_gptq) {
        if (!w.on_device && gptq.qweight.data) {
            if (!upload_gptq_weight(gptq, w, ctx.stream, ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to dequant GPTQ %s for layer %d", name, i);
                return false;
            }
            IMP_LOG_DEBUG("GPTQ dequant %s layer %d -> [%lld, %lld] FP16", name, i, w.shape[0], w.shape[1]);
        }
    }

    // FFN norm (typically F32/F16, no quant). Qwen3.5/3.6 SafeTensors stores
    // it as delta `W` (actual gamma = 1 + W); ctx.arch_norm_offset adds the +1
    // during BF16→FP16 conversion. GGUF F32 norms unaffected.
    if (L.ffn_norm.data && !L.ffn_norm.on_device) {
        if (!upload_weight(L.ffn_norm, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs, true,
                           ctx.arch_norm_offset)) {
            IMP_LOG_ERROR("Failed to upload ffn_norm for layer %d", i);
            return false;
        }
    }

    // MoE gate (routing weights, typically F32/F16)
    UPLOAD_UNQUANT_OR_FAIL(L.moe_gate, "moe_gate", i, ctx);

    // Router bias (Nemotron MoE)
    UPLOAD_UNQUANT_OR_FAIL(L.moe_router_bias, "moe_router_bias", i, ctx);

    // Shared expert weights (Nemotron/DeepSeek style)
    if (L.w_up_shared.data && !L.w_up_shared.on_device) {
        UPLOAD_OR_FAIL(L.w_up_shared, L.w_up_shared.qtype, "w_up_shared", i, ctx);
    }
    if (L.w_down_shared.data && !L.w_down_shared.on_device) {
        UPLOAD_OR_FAIL(L.w_down_shared, L.w_down_shared.qtype, "w_down_shared", i, ctx);
    }
    if (L.w_gate_shared.data && !L.w_gate_shared.on_device) {
        UPLOAD_OR_FAIL(L.w_gate_shared, L.w_gate_shared.qtype, "w_gate_shared", i, ctx);
    }
    // Qwen3-Next / Qwen3.6 shared-expert input gate (FP32 [d_model]).
    if (L.shared_expert_gate_inp.data && !L.shared_expert_gate_inp.on_device) {
        UPLOAD_OR_FAIL(L.shared_expert_gate_inp, QType::F32, "shared_expert_gate_inp", i, ctx);
    }

    return true;
}

// ---------------------------------------------------------------------------
// upload_layer_ssm_weights: SSM weights for one layer (Mamba2/Nemotron-H)
// ---------------------------------------------------------------------------
static bool upload_layer_ssm_weights(TransformerLayer& L, int i, const UploadCtx& ctx) {
    // SSM weights (Mamba2)
    if (L.ssm_in.data && !L.ssm_in.on_device) {
        UPLOAD_OR_FAIL(L.ssm_in, L.ssm_in.qtype, "ssm_in", i, ctx);
    }
    if (L.ssm_out.data && !L.ssm_out.on_device) {
        UPLOAD_OR_FAIL(L.ssm_out, L.ssm_out.qtype, "ssm_out", i, ctx);
    }
    // SSM tensors that convert to compute_dtype (FP16): conv1d weights, norm
    for (Tensor* t : {&L.ssm_conv1d_w, &L.ssm_conv1d_b, &L.ssm_norm_w}) {
        if (t->data && !t->on_device) {
            if (!upload_unquantized_weight(*t, QType::NONE, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload SSM tensor for layer %d", i);
                return false;
            }
        }
    }
    // SSM scalar/vector tensors that MUST end up as FP32 on device, because
    // the GDN/Mamba scan kernels read them as `const float*` (A_log, D,
    // dt_bias). GGUF emits them as F32 — direct upload. SafeTensors with
    // bfloat16 model dtype emits them as BF16, and a previous engine version
    // simply h2d_copy'd the bytes — the scan kernel then reinterpreted BF16
    // as F32 (sign/exponent/mantissa all wrong) and produced NaN within the
    // first GDN layer. Convert on host before upload so the scan always sees
    // real FP32 values.
    // ssm_a needs an extra HF-vs-GGUF transform: imp's GDN scan kernel computes
    //   g = exp(A_log * softplus(alpha + dt_bias))
    // which matches GGUF semantics — the Unsloth/llama.cpp converter pre-applies
    // `-exp()` to the original HF A_log so kernel reads the post-transform value.
    // HF SafeTensors carries the RAW HF `A_log` (mean ~3.3 positive). Without
    // applying `-exp()` at load time, the kernel produces exp(positive*positive)
    // and the recurrent state grows exponentially → garbage decode.
    // Verified: GGUF[i] == -exp(NVFP4_A_log_HF[head_perm(i)]) elementwise on L0.
    for (Tensor* t : {&L.ssm_a, &L.ssm_d, &L.ssm_dt_b}) {
        if (!t->data || t->on_device)
            continue;
        const bool is_ssm_a_hf = (t == &L.ssm_a) && (t->qtype == QType::BF16 || t->qtype == QType::F16);
        const int64_t n_elem = t->numel();
        const size_t fp32_bytes = static_cast<size_t>(n_elem) * sizeof(float);
        void* d_data = nullptr;
        checked_cuda_malloc(&d_data, fp32_bytes, ctx.stream);
        if (!d_data) {
            IMP_LOG_ERROR("Failed to allocate GPU memory for SSM F32 tensor in layer %d", i);
            return false;
        }

        std::vector<float> h_fp32(static_cast<size_t>(n_elem));
        if (t->qtype == QType::F32 || t->qtype == QType::NONE) {
            // GGUF path — already FP32, ssm_a already pre-transformed by converter.
            h2d_copy(d_data, t->data, fp32_bytes, ctx.stream);
            ctx.gpu_allocs.push_back(d_data);
            t->data = d_data;
            t->qtype = QType::F32;
            t->on_device = true;
            continue;
        } else if (t->qtype == QType::BF16) {
            const uint16_t* src = static_cast<const uint16_t*>(t->data);
            for (int64_t k = 0; k < n_elem; ++k) {
                uint32_t bits = static_cast<uint32_t>(src[k]) << 16;
                std::memcpy(&h_fp32[k], &bits, sizeof(float));
            }
        } else if (t->qtype == QType::F16) {
            const uint16_t* src = static_cast<const uint16_t*>(t->data);
            for (int64_t k = 0; k < n_elem; ++k) {
                h_fp32[k] = fp16_to_float(src[k]);
            }
        } else {
            IMP_LOG_ERROR("SSM scalar tensor has unexpected qtype %u (layer %d)",
                          static_cast<unsigned>(t->qtype), i);
            return false;
        }

        // Apply HF-to-GGUF A_log transform: A_log_GGUF = -exp(A_log_HF).
        // Only for ssm_a, only when source is BF16/F16 (= HF SafeTensors path).
        if (is_ssm_a_hf) {
            for (int64_t k = 0; k < n_elem; ++k) {
                h_fp32[k] = -std::exp(h_fp32[k]);
            }
            if (i == 0) {
                IMP_LOG_INFO(
                    "HF GDN A_log: applied -exp() transform to ssm_a (layer 0 first 4 values: %.4f %.4f %.4f "
                    "%.4f)",
                    h_fp32[0], h_fp32[1], h_fp32[2], h_fp32[3]);
            }
        }
        h2d_copy(d_data, h_fp32.data(), fp32_bytes, ctx.stream);

        ctx.gpu_allocs.push_back(d_data);
        t->data = d_data;
        t->qtype = QType::F32;
        t->on_device = true;
    }

    // Gated DeltaNet (GDN) weights (Qwen3.5).
    // GDN alpha/beta: dispatched via gemm_dispatch like every other quantized
    // weight. Earlier code used raw_quant=false to pre-dequant to FP16 on host,
    // but upload_weight() does not update L.gdn_*_qtype after conversion, so
    // gemm_dispatch still saw qtype=Q8_0 and re-interpreted the FP16 bytes as
    // Q8_0 blocks → ~80× too-large alpha/beta and immediate state collapse.
    // Uploading raw Q8_0 keeps the qtype consistent with the bytes on device.
    if (L.gdn_gate.data && !L.gdn_gate.on_device) {
        UPLOAD_OR_FAIL(L.gdn_gate, L.gdn_gate.qtype, "gdn_gate", i, ctx);
    }
    if (L.gdn_alpha.data && !L.gdn_alpha.on_device) {
        UPLOAD_OR_FAIL(L.gdn_alpha, L.gdn_alpha.qtype, "gdn_alpha", i, ctx);
    }
    if (L.gdn_beta.data && !L.gdn_beta.on_device) {
        UPLOAD_OR_FAIL(L.gdn_beta, L.gdn_beta.qtype, "gdn_beta", i, ctx);
    }

    // GDN input projection fusion (M=1 decode GEMV). Tries the full 4-way pack
    // first (ssm_in + gdn_gate + gdn_alpha + gdn_beta → one [total_out, d_model]
    // weight); falls back to the alpha+beta-only 2-way pack if ssm_in / gdn_gate
    // aren't FP16/BF16 (e.g. raw-quant Q*_K or NVFP4 prequant paths). Originals
    // stay live so prefill (n>1) keeps the 4-call path unchanged. Decode opt-in
    // via the executor: when n==1 it slices the fused output instead of running
    // 4 separate matmuls.
    auto& a = L.ssm_in;
    auto& b = L.gdn_gate;
    auto& c = L.gdn_alpha;
    auto& d = L.gdn_beta;

    auto fp_uniform = [](QType q, QType expect) {
        return q == expect && (q == QType::F16 || q == QType::BF16);
    };

    bool four_way_ok =
        a.data && a.on_device && b.data && b.on_device && c.data && c.on_device && d.data && d.on_device &&
        a.ndim == 2 && b.ndim == 2 && c.ndim == 2 && d.ndim == 2 &&
        (a.qtype == QType::F16 || a.qtype == QType::BF16) && fp_uniform(b.qtype, a.qtype) &&
        fp_uniform(c.qtype, a.qtype) && fp_uniform(d.qtype, a.qtype) && a.shape[1] == b.shape[1] &&
        a.shape[1] == c.shape[1] && a.shape[1] == d.shape[1] && c.shape[0] == d.shape[0];

    if (four_way_ok) {
        int64_t conv_channels = a.shape[0];
        int64_t inner = b.shape[0];
        int64_t n_heads = c.shape[0];
        int64_t d_model = a.shape[1];
        int64_t total_out = conv_channels + inner + 2 * n_heads;
        size_t es = 2;  // F16 / BF16 = 2 bytes
        size_t total_bytes = static_cast<size_t>(total_out) * d_model * es;
        void* d_packed = nullptr;
        if (cudaMallocAsync(&d_packed, total_bytes, ctx.stream) == cudaSuccess && d_packed) {
            ctx.gpu_allocs.push_back(d_packed);
            char* base = static_cast<char*>(d_packed);
            // Concat in N (rows): each weight is a contiguous [out, d_model] block,
            // so plain cudaMemcpyAsync into the packed buffer's row range is the
            // entire pack op.
            size_t bytes_a = static_cast<size_t>(conv_channels) * d_model * es;
            size_t bytes_b = static_cast<size_t>(inner) * d_model * es;
            size_t bytes_c = static_cast<size_t>(n_heads) * d_model * es;
            IMP_CUDA_CHECK_LOG(
                cudaMemcpyAsync(base, a.data, bytes_a, cudaMemcpyDeviceToDevice, ctx.stream));
            IMP_CUDA_CHECK_LOG(
                cudaMemcpyAsync(base + bytes_a, b.data, bytes_b, cudaMemcpyDeviceToDevice, ctx.stream));
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(base + bytes_a + bytes_b, c.data, bytes_c,
                                                cudaMemcpyDeviceToDevice, ctx.stream));
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(base + bytes_a + bytes_b + bytes_c, d.data, bytes_c,
                                                cudaMemcpyDeviceToDevice, ctx.stream));
            int64_t packed_shape[2] = {total_out, d_model};
            L.gdn_input_packed = Tensor(d_packed, a.qtype, 2, packed_shape, true);
            L.gdn_packed_conv_channels = static_cast<int>(conv_channels);
            L.gdn_packed_inner = static_cast<int>(inner);
            L.gdn_packed_n_heads = static_cast<int>(n_heads);
            IMP_LOG_DEBUG("  layer %d: gdn_input_packed [%lld, %lld] %.2f MiB", i, (long long)total_out,
                          (long long)d_model, total_bytes / (1024.0 * 1024.0));
        } else {
            IMP_LOG_WARN(
                "layer %d: gdn_input_packed alloc failed (%zu bytes), falling back to 2-way / 4-call", i,
                total_bytes);
        }
    }

    // Step-1 fallback: alpha+beta only (decode 2-way fusion). Skipped if
    // gdn_input_packed already covers the same case.
    if (!L.gdn_input_packed.data && c.data && c.on_device && d.data && d.on_device && c.ndim == 2 &&
        d.ndim == 2 && c.qtype == d.qtype && (c.qtype == QType::F16 || c.qtype == QType::BF16) &&
        c.shape[0] == d.shape[0] && c.shape[1] == d.shape[1]) {
        int64_t d_model = c.shape[0];
        int64_t n_heads = c.shape[1];
        size_t es = 2;
        size_t row_bytes = static_cast<size_t>(n_heads) * es;
        size_t total_bytes = static_cast<size_t>(d_model) * 2 * row_bytes;
        void* d_packed = nullptr;
        if (cudaMallocAsync(&d_packed, total_bytes, ctx.stream) == cudaSuccess && d_packed) {
            ctx.gpu_allocs.push_back(d_packed);
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(d_packed, 2 * row_bytes, c.data, row_bytes, row_bytes,
                                                 d_model, cudaMemcpyDeviceToDevice, ctx.stream));
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(static_cast<char*>(d_packed) + row_bytes, 2 * row_bytes,
                                                 d.data, row_bytes, row_bytes, d_model,
                                                 cudaMemcpyDeviceToDevice, ctx.stream));
            int64_t packed_shape[2] = {d_model, 2 * n_heads};
            L.gdn_alpha_beta_packed = Tensor(d_packed, c.qtype, 2, packed_shape, true);
            IMP_LOG_DEBUG("  layer %d: gdn_alpha_beta_packed [%lld, %lld] %.2f KiB (Step-1 fallback)", i,
                          (long long)d_model, (long long)(2 * n_heads), total_bytes / 1024.0);
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// upload_expert_weights: MoE expert weight upload for all layers (Pass 2).
// Handles packed 3D tensors and per-expert 2D tensors.
// ---------------------------------------------------------------------------
// Phase 1 of upload_expert_weights: compute per-layer expert-tensor byte cost
// for the packed 3-D tensors PLUS per-expert 2-D tensors (NVFP4 llm-compressor
// format). Returns total_expert_bytes; fills layer_expert_bytes in place.
static size_t compute_expert_layer_costs_(const std::vector<TransformerLayer>& layers, int n_layers,
                                          std::vector<size_t>& layer_expert_bytes) {
    size_t total_expert_bytes = 0;
    layer_expert_bytes.assign(n_layers, 0);
    for (int i = 0; i < n_layers; ++i) {
        const TransformerLayer& L = layers[i];
        auto add_packed = [&](const Tensor& p, QType qt) {
            if (!p.data || p.ndim < 3 || !dequant_gpu_supported(qt))
                return;
            size_t row_bytes = qtype_row_bytes(qt, p.shape[2]);
            size_t bytes = static_cast<size_t>(p.shape[0]) * p.shape[1] * row_bytes;
            layer_expert_bytes[i] += bytes;
            total_expert_bytes += bytes;
        };
        add_packed(L.expert_gate_packed, L.expert_gate_packed.qtype);
        add_packed(L.expert_up_packed, L.expert_up_packed.qtype);
        add_packed(L.expert_down_packed, L.expert_down_packed.qtype);

        // Also account for per-expert 2D tensors (e.g. NVFP4 llm-compressor format).
        // These are NOT packed 3D, so add_packed misses them.
        auto add_2d_expert = [&](const std::vector<Tensor>& vt) {
            for (const auto& t : vt) {
                if (!t.data || t.on_device || t.ndim < 2)
                    continue;
                layer_expert_bytes[i] += t.nbytes();
                total_expert_bytes += t.nbytes();
            }
        };
        add_2d_expert(L.expert_w_gate);
        add_2d_expert(L.expert_w_up);
        add_2d_expert(L.expert_w_down);
    }
    return total_expert_bytes;
}

// Phase 2 of upload_expert_weights: pick which MoE layers' experts stay on
// GPU vs go to host. Honors:
//   - VRAM reserve passed by Engine (KV cache + workspaces + FP16 cache),
//   - WSL2/WDDM driver overhead (auto-pick 10 % aggressive if all fit, else
//     30 % conservative), explicit overhead_pct override (0..50),
//   - moe.force_host_experts = N debug flag (last N MoE layers off-GPU).
// Also re-arms the g_cached_free_mem / g_total_allocated / g_vram_reserve
// trio so per-expert checked_cuda_malloc calls don't double-count the
// reserve.
static void decide_expert_layer_placement_(const std::vector<size_t>& layer_expert_bytes,
                                           size_t total_expert_bytes, size_t expert_reserve_bytes,
                                           int n_layers, std::vector<bool>& experts_upload_layer) {
    if (total_expert_bytes == 0)
        return;

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Auto-pick default: use 10% (aggressive) if ALL experts would fit with
    // that overhead, else 30% (conservative). This saves users from a
    // silent 3× perf penalty on Qwen3-Coder-30B / Qwen3.6-35B-class MoE
    // models where the conservative default unnecessarily offloads experts
    // to host (measured: 77 → 237 tok/s with 10% vs 30% on Qwen3-Coder-30B
    // Q6_K, RTX 5090 native Linux).
    int overhead_pct = process_diag_moe_expert_overhead_pct();
    if (overhead_pct < 0 || overhead_pct > 50) {
        overhead_pct = 30;
    } else if (overhead_pct == 10) {
        size_t aggressive_overhead = static_cast<size_t>(free_mem * 10 / 100);
        size_t aggressive_reserve = expert_reserve_bytes + aggressive_overhead;
        size_t aggressive_budget = (free_mem > aggressive_reserve) ? (free_mem - aggressive_reserve) : 0;
        if (aggressive_budget < total_expert_bytes) {
            overhead_pct = 30;
        } else {
            IMP_LOG_INFO(
                "Expert offload: all experts fit with 10%% overhead "
                "(%.2f GiB experts, %.2f GiB free) — picking aggressive.",
                total_expert_bytes / (1024.0 * 1024.0 * 1024.0), free_mem / (1024.0 * 1024.0 * 1024.0));
        }
    }
    size_t overhead = static_cast<size_t>(free_mem * overhead_pct / 100);
    size_t total_reserve = expert_reserve_bytes + overhead;
    size_t budget = (free_mem > total_reserve) ? (free_mem - total_reserve) : 0;

    int force_host_n = process_diag_moe_force_host_experts();
    if (force_host_n < 0)
        force_host_n = 0;

    if (budget >= total_expert_bytes && force_host_n == 0) {
        for (int i = 0; i < n_layers; ++i) {
            if (layer_expert_bytes[i] > 0)
                experts_upload_layer[i] = true;
        }
        IMP_LOG_INFO(
            "Expert weights: %.2f GiB -> uploading ALL to GPU "
            "(%.2f GiB free, %.2f GiB reserve)",
            total_expert_bytes / (1024.0 * 1024.0 * 1024.0), free_mem / (1024.0 * 1024.0 * 1024.0),
            expert_reserve_bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (force_host_n > 0) {
        // Debug: force last N MoE layers to host. Still respect budget for
        // the ones we do upload.
        std::vector<int> moe_layer_idxs;
        for (int i = 0; i < n_layers; ++i)
            if (layer_expert_bytes[i] > 0)
                moe_layer_idxs.push_back(i);
        int skip_from = std::max(0, (int)moe_layer_idxs.size() - force_host_n);
        size_t uploaded = 0;
        int n_uploaded = 0;
        for (int k = 0; k < skip_from; ++k) {
            int i = moe_layer_idxs[k];
            if (uploaded + layer_expert_bytes[i] <= budget) {
                experts_upload_layer[i] = true;
                uploaded += layer_expert_bytes[i];
                n_uploaded++;
            }
        }
        IMP_LOG_INFO(
            "Expert weights (IMP_FORCE_HOST_EXPERTS=%d): uploading %d/%zu MoE layers, %d forced to host",
            force_host_n, n_uploaded, moe_layer_idxs.size(), force_host_n);
    } else {
        // Partial upload: greedily upload layers until budget exhausted
        size_t uploaded = 0;
        int n_uploaded = 0, n_total_moe = 0;
        for (int i = 0; i < n_layers; ++i) {
            if (layer_expert_bytes[i] == 0)
                continue;
            n_total_moe++;
            if (uploaded + layer_expert_bytes[i] <= budget) {
                experts_upload_layer[i] = true;
                uploaded += layer_expert_bytes[i];
                n_uploaded++;
            }
        }
        IMP_LOG_INFO(
            "Expert weights: %.2f GiB total, uploading %d/%d MoE layers "
            "(%.2f GiB on GPU, %.2f GiB on host, %.2f GiB free, "
            "%.2f GiB reserve)",
            total_expert_bytes / (1024.0 * 1024.0 * 1024.0), n_uploaded, n_total_moe,
            uploaded / (1024.0 * 1024.0 * 1024.0),
            (total_expert_bytes - uploaded) / (1024.0 * 1024.0 * 1024.0),
            free_mem / (1024.0 * 1024.0 * 1024.0), expert_reserve_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    // Re-arm the cached free-memory window so per-expert checked_cuda_malloc
    // calls don't fall back to a sync cudaMemGetInfo on every tensor.
    g_cached_free_mem = free_mem;
    g_total_allocated = 0;
    g_vram_reserve = 0;  // budget already accounted for above
}

static bool upload_expert_weights(std::vector<TransformerLayer>& layers, int n_layers,
                                  size_t expert_reserve_bytes, const UploadCtx& ctx) {
    std::vector<size_t> layer_expert_bytes;
    size_t total_expert_bytes = compute_expert_layer_costs_(layers, n_layers, layer_expert_bytes);

    std::vector<bool> experts_upload_layer(n_layers, false);
    decide_expert_layer_placement_(layer_expert_bytes, total_expert_bytes, expert_reserve_bytes, n_layers,
                                   experts_upload_layer);

    // Upload expert weights for each layer
    for (int i = 0; i < n_layers; ++i) {
        TransformerLayer& L = layers[i];

        // MoE expert weights -- two paths:
        // A) Packed 3D tensors (*_exps):
        //    - For quantized types (Q6_K, Q8_0, Q4_0): upload raw bytes to GPU,
        //      keep packed tensor. Dequant happens on-the-fly in run_moe_ffn.
        //    - For F16/BF16/F32: dequant/upload and slice into per-expert views.
        // B) Per-expert 2D tensors: upload individually (legacy per-expert GGUF format)

        auto upload_packed_experts = [&](Tensor& packed, QType qtype, std::vector<Tensor>& expert_vec,
                                         const char* name) -> bool {
            if (!packed.data || packed.ndim < 3)
                return true;  // nothing to do
            if (packed.on_device)
                return true;  // already on GPU (e.g. from Gemma 4 fused split)

            int n_experts = static_cast<int>(packed.shape[0]);
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];

            // Path A1: Quantized types -- upload raw bytes to GPU if they fit,
            // otherwise keep on host (mmap'd) with optional pinning for H2D.
            if (dequant_gpu_supported(qtype)) {
                size_t row_bytes = qtype_row_bytes(qtype, cols);
                size_t expert_raw = static_cast<size_t>(rows) * row_bytes;
                size_t total_raw = static_cast<size_t>(n_experts) * expert_raw;

                if (experts_upload_layer[i]) {
                    // Upload raw quantized bytes to GPU (respects VRAM reserve)
                    void* gpu_ptr = nullptr;
                    cudaError_t err = checked_cuda_malloc(&gpu_ptr, total_raw, ctx.stream);
                    if (err == cudaSuccess) {
                        cudaError_t cpy_err = h2d_copy(gpu_ptr, packed.data, total_raw, ctx.stream);
                        if (cpy_err != cudaSuccess) {
                            IMP_LOG_ERROR("  %s: h2d_copy failed: %s", name, cudaGetErrorString(cpy_err));
                            IMP_CUDA_CHECK_LOG(cudaFreeAsync(gpu_ptr, ctx.stream));
                            return false;
                        }
                        packed.data = gpu_ptr;
                        packed.on_device = true;
                        ctx.gpu_allocs.push_back(gpu_ptr);
                        IMP_LOG_DEBUG("  %s: %d experts uploaded to GPU (%.2f MiB)", name, n_experts,
                                      total_raw / (1024.0 * 1024.0));
                        return true;
                    }
                    // cudaMalloc failed — fall through to host path
                    IMP_LOG_WARN("  %s: cudaMalloc failed for %.2f MiB, falling back to host", name,
                                 total_raw / (1024.0 * 1024.0));
                }

                // Host path: pin memory for fast async DMA H2D during decode.
                if (is_wsl2()) {
                    // WSL2: cudaHostRegister fails on mmap'd memory. Instead,
                    // allocate fresh pinned memory and copy mmap'd data there.
                    // This enables true async DMA H2D (no per-token CPU memcpy).
                    void* pinned_buf = nullptr;
                    cudaError_t pin_err = cudaHostAlloc(&pinned_buf, total_raw, cudaHostAllocDefault);
                    if (pin_err == cudaSuccess) {
                        memcpy(pinned_buf, packed.data, total_raw);
                        packed.data = pinned_buf;
                        ctx.host_pinned_allocs.push_back(pinned_buf);
                        IMP_LOG_INFO("  %s: WSL2 pinned copy (%.2f MiB, DMA-ready)", name,
                                     total_raw / (1024.0 * 1024.0));
                    } else {
                        IMP_LOG_DEBUG("Cleared WSL2 cudaHostAlloc error: %s", cudaGetErrorString(pin_err));
                        cudaGetLastError();  // clear sticky CUDA error state
                        IMP_LOG_INFO(
                            "  %s: WSL2 cudaHostAlloc failed, falling back to "
                            "unpinned mmap (%.2f MiB)",
                            name, total_raw / (1024.0 * 1024.0));
                    }
                } else {
                    cudaError_t pin_err = cudaHostRegister(packed.data, total_raw, cudaHostRegisterReadOnly);
                    if (pin_err == cudaSuccess) {
                        ctx.host_pinned.push_back(packed.data);
                        IMP_LOG_DEBUG("  %s: %d experts, raw %s pinned on host (%.2f MiB)", name, n_experts,
                                      qtype == QType::Q6_K   ? "Q6_K"
                                      : qtype == QType::Q8_0 ? "Q8_0"
                                                             : "Q4_0",
                                      total_raw / (1024.0 * 1024.0));
                    } else {
                        IMP_LOG_WARN("  %s: cudaHostRegister failed (%s), H2D will be slower", name,
                                     cudaGetErrorString(pin_err));
                    }
                }

                return true;
            }

            // Path A2: Unquantized (F16/BF16/F32) -- dequant to FP16, slice per-expert.
            int64_t flat_shape[4] = {static_cast<int64_t>(n_experts) * rows, cols, 0, 0};
            Tensor flat(packed.data, packed.qtype, 2, flat_shape, packed.on_device);

            if (!upload_weight(flat, qtype, ctx.compute_dtype, ctx.stream, ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload packed %s for layer %d", name, i);
                return false;
            }

            expert_vec.resize(n_experts);
            size_t expert_bytes = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
            for (int e = 0; e < n_experts; e++) {
                char* ptr = static_cast<char*>(flat.data) + e * expert_bytes;
                int64_t eshape[4] = {rows, cols, 0, 0};
                expert_vec[e] = Tensor(ptr, QType::F16, 2, eshape, true);
            }
            packed = Tensor();
            return true;
        };

        if (!upload_packed_experts(L.expert_gate_packed, L.expert_gate_packed.qtype, L.expert_w_gate,
                                   "expert_gate_exps"))
            return false;

        // Gemma 4: split fused ffn_gate_up_exps into separate gate and up packed tensors.
        // The original tensor (in expert_gate_packed) has shape [n_exp, 2*n_ff_exp, d_model].
        // Layout per expert: rows [0, n_ff_exp) = gate, rows [n_ff_exp, 2*n_ff_exp) = up.
        // We split physically on GPU via cudaMemcpy2D, then free the fused buffer.
        // Host-resident fused gate_up split (Gemma-4 + partial upload):
        // When the fused tensor is not uploaded to GPU, we still need to split
        // it into gate and up tensors so the MoE dispatch can find
        // expert_up_packed. Without this, host-resident MoE layers have
        // nullptr expert_up_packed → use_packed_dequant=0 → fallback to
        // uninitialized expert_w_up[eidx] → garbage output.
        //
        // Gate this on `!experts_upload_layer[i]` — only for layers that won't
        // be uploaded. Upload-destined layers use the GPU split code below,
        // which runs after upload_packed_experts has set on_device=true.
        if (!experts_upload_layer[i] && L.expert_gate_packed.data && !L.expert_gate_packed.on_device &&
            L.expert_up_packed.data == nullptr && L.expert_gate_packed.ndim >= 3 &&
            (L.expert_gate_packed.shape[1] & 1) == 0 && dequant_gpu_supported(L.expert_gate_packed.qtype)) {
            int64_t n_exp = L.expert_gate_packed.shape[0];
            int64_t fused_rows = L.expert_gate_packed.shape[1];
            int64_t cols = L.expert_gate_packed.shape[2];
            int64_t half_rows = fused_rows / 2;
            size_t row_bytes = qtype_row_bytes(L.expert_gate_packed.qtype, cols);
            size_t half_raw = static_cast<size_t>(n_exp) * half_rows * row_bytes;
            size_t src_pitch = static_cast<size_t>(fused_rows) * row_bytes;
            size_t dst_pitch = static_cast<size_t>(half_rows) * row_bytes;

            // Allocate two pinned host buffers for the split halves.
            void* gate_buf = nullptr;
            void* up_buf = nullptr;
            cudaError_t eg = cudaHostAlloc(&gate_buf, half_raw, cudaHostAllocDefault);
            cudaError_t eu = cudaHostAlloc(&up_buf, half_raw, cudaHostAllocDefault);
            if (eg != cudaSuccess || eu != cudaSuccess) {
                IMP_LOG_ERROR("Gemma 4: host split cudaHostAlloc failed (layer %d): %s/%s", i,
                              cudaGetErrorString(eg), cudaGetErrorString(eu));
                if (gate_buf)
                    cudaFreeHost(gate_buf);
                if (up_buf)
                    cudaFreeHost(up_buf);
                return false;
            }

            const char* src_base = static_cast<const char*>(L.expert_gate_packed.data);
            for (int64_t e = 0; e < n_exp; ++e) {
                // gate half = rows [0, half_rows)
                memcpy(static_cast<char*>(gate_buf) + e * dst_pitch, src_base + e * src_pitch, dst_pitch);
                // up half = rows [half_rows, fused_rows)
                memcpy(static_cast<char*>(up_buf) + e * dst_pitch, src_base + e * src_pitch + dst_pitch,
                       dst_pitch);
            }

            int64_t split_shape[4] = {n_exp, half_rows, cols, 0};
            L.expert_gate_packed = Tensor(gate_buf, L.expert_gate_packed.qtype, 3, split_shape, false);
            L.expert_up_packed = Tensor(up_buf, L.expert_gate_packed.qtype, 3, split_shape, false);
            L.expert_up_packed.qtype = L.expert_gate_packed.qtype;
            ctx.host_pinned_allocs.push_back(gate_buf);
            ctx.host_pinned_allocs.push_back(up_buf);
            IMP_LOG_INFO("Gemma 4: host-split fused gate_up_exps layer %d (n_ff_exp=%ld, %.1f MiB each)", i,
                         (long)half_rows, half_raw / (1024.0 * 1024.0));
        }

        if (L.expert_gate_packed.data && L.expert_gate_packed.on_device &&
            L.expert_up_packed.data == nullptr && L.expert_gate_packed.ndim >= 3 &&
            (L.expert_gate_packed.shape[1] & 1) == 0 && dequant_gpu_supported(L.expert_gate_packed.qtype)) {
            int64_t n_exp = L.expert_gate_packed.shape[0];
            int64_t fused_rows = L.expert_gate_packed.shape[1];
            int64_t cols = L.expert_gate_packed.shape[2];
            int64_t half_rows = fused_rows / 2;
            size_t row_bytes = qtype_row_bytes(L.expert_gate_packed.qtype, cols);
            size_t half_raw = static_cast<size_t>(n_exp) * half_rows * row_bytes;

            // Memory-efficient split: allocate only ONE half-sized buffer for the
            // up half, copy it out, then reuse the fused buffer in-place for the
            // gate half (its rows are already at the front; the trailing half is
            // simply ignored via the new shape). Peak overhead = 0.5x fused
            // instead of 1.0x for a two-buffer split.
            void* up_buf = nullptr;
            cudaError_t e2 = checked_cuda_malloc(&up_buf, half_raw, ctx.stream);
            if (e2 != cudaSuccess) {
                IMP_LOG_ERROR("Gemma 4: cudaMalloc failed for fused expert split (layer %d, %.1f MiB)", i,
                              half_raw / (1024.0 * 1024.0));
                return false;
            }

            size_t dst_pitch = static_cast<size_t>(half_rows) * row_bytes;
            size_t src_pitch = static_cast<size_t>(fused_rows) * row_bytes;
            const char* src_base = static_cast<const char*>(L.expert_gate_packed.data);

            // Copy up half (rows [half_rows, fused_rows)) into the new up buffer.
            cudaError_t cp = cudaMemcpy2DAsync(up_buf, dst_pitch, src_base + dst_pitch, src_pitch, dst_pitch,
                                               n_exp, cudaMemcpyDeviceToDevice, ctx.stream);
            if (cp != cudaSuccess) {
                IMP_LOG_ERROR("Gemma 4: cudaMemcpy2DAsync failed (layer %d): %s", i, cudaGetErrorString(cp));
                cudaFreeAsync(up_buf, ctx.stream);
                return false;
            }
            // Compact the gate half in-place: row e at offset e*src_pitch must
            // move to offset e*dst_pitch. Walk experts forward — for forward
            // copy, dst[e] starts BEFORE src[e] (e*dst_pitch < e*src_pitch +
            // half_pitch for e>=1, but src[e] of expert e is read fully before
            // expert e+1's dst is written, so this is safe as a sequential 2D
            // copy from a single launch only when there is no inter-expert
            // overlap. With dst_pitch < src_pitch the expert-1 dst region
            // overlaps with expert-0 src — so we must serialize per expert.
            for (int64_t e = 1; e < n_exp; ++e) {  // e=0 already at the right offset
                cudaError_t cp_e = cudaMemcpyAsync(const_cast<char*>(src_base) + e * dst_pitch,
                                                   src_base + e * src_pitch, dst_pitch,
                                                   cudaMemcpyDeviceToDevice, ctx.stream);
                if (cp_e != cudaSuccess) {
                    IMP_LOG_ERROR("Gemma 4: gate compact memcpy failed (layer %d, expert %ld): %s", i,
                                  (long)e, cudaGetErrorString(cp_e));
                    cudaFreeAsync(up_buf, ctx.stream);
                    return false;
                }
            }

            // Reuse fused buffer (now compacted to half size) as gate_packed.
            int64_t split_shape[4] = {n_exp, half_rows, cols, 0};
            void* gate_buf = L.expert_gate_packed.data;
            L.expert_gate_packed = Tensor(gate_buf, L.expert_gate_packed.qtype, 3, split_shape, true);
            L.expert_up_packed = Tensor(up_buf, L.expert_gate_packed.qtype, 3, split_shape, true);
            L.expert_up_packed.qtype = L.expert_gate_packed.qtype;
            // gate_buf is already in ctx.gpu_allocs from the original upload.
            ctx.gpu_allocs.push_back(up_buf);
            IMP_LOG_INFO("Gemma 4: split fused gate_up_exps layer %d (n_ff_exp=%ld, %.1f MiB each)", i,
                         (long)half_rows, half_raw / (1024.0 * 1024.0));
        }

        if (!upload_packed_experts(L.expert_up_packed, L.expert_up_packed.qtype, L.expert_w_up,
                                   "expert_up_exps"))
            return false;
        if (!upload_packed_experts(L.expert_down_packed, L.expert_down_packed.qtype, L.expert_w_down,
                                   "expert_down_exps"))
            return false;

        // Path B: per-expert 2D tensors (from per-expert GGUF or llm-compressor NVFP4 format).
        // Respect the experts_upload_layer budget flag — skip if experts for this layer
        // don't fit in the remaining VRAM budget.
        if (experts_upload_layer[i]) {
            for (size_t e = 0; e < L.expert_w_gate.size(); ++e) {
                if (!L.expert_w_gate[e].data || L.expert_w_gate[e].on_device)
                    continue;
                if (!upload_weight(L.expert_w_gate[e], L.expert_gate_packed.qtype, ctx.compute_dtype,
                                   ctx.stream, ctx.gpu_allocs)) {
                    IMP_LOG_ERROR("Failed to upload expert_w_gate[%zu] for layer %d", e, i);
                    return false;
                }
            }
            for (size_t e = 0; e < L.expert_w_up.size(); ++e) {
                if (!L.expert_w_up[e].data || L.expert_w_up[e].on_device)
                    continue;
                if (!upload_weight(L.expert_w_up[e], L.expert_up_packed.qtype, ctx.compute_dtype, ctx.stream,
                                   ctx.gpu_allocs)) {
                    IMP_LOG_ERROR("Failed to upload expert_w_up[%zu] for layer %d", e, i);
                    return false;
                }
            }
            for (size_t e = 0; e < L.expert_w_down.size(); ++e) {
                if (!L.expert_w_down[e].data || L.expert_w_down[e].on_device)
                    continue;
                if (!upload_weight(L.expert_w_down[e], L.expert_down_packed.qtype, ctx.compute_dtype,
                                   ctx.stream, ctx.gpu_allocs)) {
                    IMP_LOG_ERROR("Failed to upload expert_w_down[%zu] for layer %d", e, i);
                    return false;
                }
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Model::upload_weights_gpu
// ---------------------------------------------------------------------------

bool Model::upload_weights_gpu(QType compute_dtype, cudaStream_t stream, size_t expert_reserve_bytes) {
    if (gpu_weights_ready_) {
        IMP_LOG_WARN("Weights already uploaded to GPU");
        return true;
    }

    IMP_LOG_INFO("Uploading model weights to GPU (%d layers)...", n_layers());

    // Initialize pinned staging for fast H2D (especially on WSL2 where mmap can't be pinned).
    // StagingGuard provides RAII cleanup on all exit paths (including early return false).
    struct StagingGuard {
        PinnedStager stager;
        ~StagingGuard() {
            g_stager = nullptr;
            stager.destroy();
            g_cached_free_mem = 0;
            g_total_allocated = 0;
            g_vram_reserve = 0;
        }
    } staging_guard;

    // Use Engine's computed reserve (workspace + KV cache + SSM state + safety)
    // directly — no additional margin needed here.
    g_vram_reserve = expert_reserve_bytes;

    if (staging_guard.stager.init()) {
        g_stager = &staging_guard.stager;
        IMP_LOG_INFO("Pinned staging enabled (%dx %.0f MiB ring)", PinnedStager::kRing,
                     PinnedStager::kChunkSize / (1024.0 * 1024.0));
    } else {
        IMP_LOG_WARN("Pinned staging alloc failed, using default H2D path");
    }

    // Cache VRAM state to avoid per-tensor cudaMemGetInfo calls
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        g_cached_free_mem = free_mem;
        g_total_allocated = 0;
        IMP_LOG_DEBUG("VRAM at upload start: %.2f GiB free / %.2f GiB total",
                      free_mem / (1024.0 * 1024.0 * 1024.0), total_mem / (1024.0 * 1024.0 * 1024.0));
    }

    // Qwen3.5/3.6 SafeTensors stores block-norm gammas as deltas (gamma = 1+W).
    // GGUF stores the post-+1 values directly. We bake the +1 in during the
    // BF16→FP16 upload conversion; F32-source norms (GGUF) are unaffected.
    const float arch_norm_offset = (config_.arch == ModelArch::QWEN35 ||
                                    config_.arch == ModelArch::QWEN35_MOE ||
                                    config_.arch == ModelArch::QWEN36_MOE)
                                       ? 1.0f
                                       : 0.0f;
    UploadCtx ctx{compute_dtype,       stream,          gpu_allocations_, host_pinned_,
                  host_pinned_allocs_, arch_norm_offset};

    // --- Embeddings, output norm, output projection ---
    if (!upload_embeddings_and_output(tok_emb_, out_norm_, out_proj_, ctx)) {
        return false;
    }

    // =========================================================================
    // Two-pass upload strategy:
    // Pass 1: Upload all non-expert per-layer weights (attention, FFN, norms,
    //         SSM, shared experts, routing). This consumes a variable amount
    //         of VRAM that's hard to estimate accurately.
    // Pass 2: After non-expert weights are on GPU, cudaMemGetInfo gives us
    //         the actual remaining VRAM. We then greedily upload expert
    //         layers until the budget is exhausted.
    // =========================================================================

    // --- Pass 1: Non-expert per-layer weights ---
    for (int i = 0; i < n_layers(); ++i) {
        TransformerLayer& L = layers_[i];

        if (!upload_layer_attention_weights(L, i, ctx))
            return false;
        if (!upload_layer_ffn_weights(L, i, ctx))
            return false;

        // (Expert weights are uploaded in Pass 2 below)

        if (!upload_layer_ssm_weights(L, i, ctx))
            return false;

        IMP_LOG_DEBUG("Layer %d/%d non-expert weights uploaded", i + 1, n_layers());
    }

    // Sync Pass 1 before measuring free VRAM for expert budget
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }

    // Reset cached VRAM state — the dense upload pass consumed an unknown amount
    // of VRAM (including CUDA driver overhead, page tables, alignment).
    // Expert upload must use real cudaMemGetInfo for accurate budget enforcement.
    g_cached_free_mem = 0;
    g_total_allocated = 0;

    // =========================================================================
    // --- Pass 2: Expert weight upload ---
    // Now that all non-expert weights are on GPU, measure actual free VRAM
    // and greedily upload expert layers until the budget is exhausted.
    // =========================================================================

    if (!upload_expert_weights(layers_, n_layers(), expert_reserve_bytes, ctx)) {
        return false;
    }

    // Upload NVFP4 pre-quantized scale tensors. Single scratch map keyed by
    // canonical slot name; replaced the per-layer NvFP4PreQuantWeight slots.
    if (config_.is_nvfp4_prequant) {
        int scale_count = 0;
        // Diagnostic: diagnostics.audit_nvfp4_scales (legacy
        // IMP_AUDIT_NVFP4_SCALES=1) dumps per-slot stats for weight_scale_2
        // (tensor-level FP32 scalar) BEFORE upload, so we can bisect
        // Mistral-3.2-NVFP4 long-form bugs by comparing scale ranges against
        // a known-good model (e.g. Gemma-4-NVFP4).
        const bool audit = imp::process_diag_audit_nvfp4_scales();
        float ws2_min = 1e30f, ws2_max = -1e30f, ws2_sum = 0.0f;
        int ws2_count = 0, ws2_zero = 0;
        std::vector<std::pair<std::string, float>> ws2_samples;
        // Same stats for input_scale (FP32 scalar per Linear, optional).
        float is_min = 1e30f, is_max = -1e30f, is_sum = 0.0f;
        int is_count = 0, is_zero = 0, is_present = 0;
        int is_scalar_count = 0, is_per_channel_count = 0;
        struct InputScaleSample {
            std::string name;
            int ndim;
            int64_t shape[4];
            size_t numel;
            float first_val;
        };
        std::vector<InputScaleSample> is_samples;
        if (audit) {
            ws2_samples.reserve(8);
            is_samples.reserve(8);
        }
        auto upload_scale = [&](Tensor& t) {
            if (!t.data || t.on_device || t.numel() == 0)
                return;
            size_t bytes = t.nbytes();
            void* d_ptr = nullptr;
            if (cudaMallocAsync(&d_ptr, bytes, stream) != cudaSuccess)
                return;
            cudaMemcpyAsync(d_ptr, t.data, bytes, cudaMemcpyHostToDevice, stream);
            gpu_allocations_.push_back(d_ptr);
            t.data = d_ptr;
            t.on_device = true;
            scale_count++;
        };
        for (auto& [name, sc] : nvfp4_scratch_) {
            // Audit weight_scale_2 BEFORE upload (it's still a host pointer).
            if (audit && sc.weight_scale_2.data && !sc.weight_scale_2.on_device) {
                size_t n = sc.weight_scale_2.numel();
                const float* p = static_cast<const float*>(sc.weight_scale_2.data);
                for (size_t i = 0; i < n; ++i) {
                    float v = p[i];
                    if (v == 0.0f)
                        ws2_zero++;
                    if (v < ws2_min)
                        ws2_min = v;
                    if (v > ws2_max)
                        ws2_max = v;
                    ws2_sum += v;
                    ws2_count++;
                }
                if (ws2_samples.size() < 8) {
                    ws2_samples.emplace_back(name, p[0]);
                }
            }
            if (audit && sc.input_scale.data && !sc.input_scale.on_device) {
                is_present++;
                size_t n = sc.input_scale.numel();
                if (n <= 1)
                    is_scalar_count++;
                else
                    is_per_channel_count++;
                const float* p = static_cast<const float*>(sc.input_scale.data);
                for (size_t i = 0; i < n; ++i) {
                    float v = p[i];
                    if (v == 0.0f)
                        is_zero++;
                    if (v < is_min)
                        is_min = v;
                    if (v > is_max)
                        is_max = v;
                    is_sum += v;
                    is_count++;
                }
                if (is_samples.size() < 8) {
                    InputScaleSample s;
                    s.name = name;
                    s.ndim = sc.input_scale.ndim;
                    for (int d = 0; d < s.ndim && d < 4; ++d)
                        s.shape[d] = sc.input_scale.shape[d];
                    s.numel = n;
                    s.first_val = p[0];
                    is_samples.push_back(std::move(s));
                }
            }
            upload_scale(sc.weight_scale);
            upload_scale(sc.weight_scale_2);
            // input_scale is loaded for diagnostics but never read by any
            // GEMM kernel (see executor_pre_dequant.cu Phase 0 comment + the
            // dead-end memory). Only upload when audit mode is on so we don't
            // burn VRAM on a tensor we'll never use in production.
            if (audit) {
                upload_scale(sc.input_scale);
            }
        }
        if (audit && ws2_count > 0) {
            IMP_LOG_INFO(
                "NVFP4 audit: weight_scale_2 stats — count=%d zeros=%d "
                "min=%.6g max=%.6g mean=%.6g",
                ws2_count, ws2_zero, ws2_min, ws2_max, ws2_sum / ws2_count);
            for (auto& [n, v] : ws2_samples) {
                IMP_LOG_INFO("  sample: %s = %.6g", n.c_str(), v);
            }
        }
        if (audit) {
            if (is_count > 0) {
                IMP_LOG_INFO(
                    "NVFP4 audit: input_scale present in %d/%zu Linears "
                    "(scalar=%d per_channel=%d), "
                    "stats — count=%d zeros=%d min=%.6g max=%.6g mean=%.6g",
                    is_present, nvfp4_scratch_.size(),
                    is_scalar_count, is_per_channel_count,
                    is_count, is_zero, is_min, is_max, is_sum / is_count);
                for (auto& s : is_samples) {
                    char shape_str[64];
                    int off = 0;
                    off += snprintf(shape_str + off, sizeof(shape_str) - off, "[");
                    for (int d = 0; d < s.ndim; ++d) {
                        off += snprintf(shape_str + off, sizeof(shape_str) - off,
                                        "%s%lld", d > 0 ? "," : "", (long long)s.shape[d]);
                    }
                    snprintf(shape_str + off, sizeof(shape_str) - off, "]");
                    IMP_LOG_INFO(
                        "  sample: %s.input_scale ndim=%d shape=%s numel=%zu first=%.6g",
                        s.name.c_str(), s.ndim, shape_str, s.numel, s.first_val);
                }
            } else {
                IMP_LOG_INFO(
                    "NVFP4 audit: no input_scale tensors found "
                    "(model uses purely dynamic input act-quant)");
            }
        }
        if (scale_count > 0)
            IMP_LOG_INFO("NVFP4 prequant: uploaded %d scale tensors to GPU", scale_count);
    }

    // --- MTP head weights (DeepSeek-V3 / Qwen3.6 family, optional sidecar) ---
    // Phase 2 of MTP wiring: upload the trained MTP head tensors. The forward
    // path that consumes them is Phase 3+. Loading them here gates VRAM-wise
    // — if the upload fails (no VRAM), we degrade by disabling MTP rather
    // than failing the entire model load.
    if (mtp_.has_value() && mtp_->loaded) {
        size_t allocs_before = gpu_allocations_.size();
        if (upload_mtp_weights(*mtp_, ctx)) {
            IMP_LOG_INFO("MTP head: uploaded to GPU (%zu allocations, %.2f GiB BF16→FP16)",
                         gpu_allocations_.size() - allocs_before,
                         static_cast<double>(mtp_->info.file_bytes) /
                             (1024.0 * 1024.0 * 1024.0));
        } else {
            IMP_LOG_WARN("MTP head: GPU upload failed — spec-decode disabled");
            mtp_->loaded = false;
        }
    }

    // Final sync
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }

    gpu_weights_ready_ = true;
    IMP_LOG_INFO("All model weights uploaded to GPU (%zu allocations)", gpu_allocations_.size());
    return true;
}

#undef UPLOAD_OR_FAIL
#undef UPLOAD_OR_FAIL_RAW
#undef UPLOAD_UNQUANT_OR_FAIL

}  // namespace imp
