#include "model/model.h"
#include "model/gguf_loader.h"
#include "quant/dequant_gpu.h"
#include "quant/dequant_gptq.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

static cudaError_t checked_cuda_malloc(void** ptr, size_t size) {
    size_t reserve = g_vram_reserve;
    // Use cached free memory (updated at start of each upload pass)
    if (g_cached_free_mem > 0) {
        if (g_total_allocated + size + reserve > g_cached_free_mem) {
            *ptr = nullptr;
            return cudaErrorMemoryAllocation;
        }
        cudaError_t err = cudaMalloc(ptr, size);
        if (err == cudaSuccess) g_total_allocated += size;
        return err;
    }
    // Fallback: per-tensor check (used outside upload passes)
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (size + reserve > free_mem) {
        *ptr = nullptr;
        return cudaErrorMemoryAllocation;
    }
    return cudaMalloc(ptr, size);
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
    static constexpr size_t kChunkSize = 64 << 20;  // 64 MiB per buffer
    void* buf[2] = {};
    cudaEvent_t done[2] = {};
    int idx = 0;

    bool init() {
        for (int i = 0; i < 2; i++) {
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
        for (size_t off = 0; off < n; ) {
            size_t chunk = std::min(n - off, kChunkSize);
            int b = idx & 1;
            cudaEventSynchronize(done[b]);
            memcpy(buf[b], static_cast<const char*>(src) + off, chunk);
            last = cudaMemcpyAsync(static_cast<char*>(dst) + off, buf[b],
                                   chunk, cudaMemcpyHostToDevice, s);
            cudaEventRecord(done[b], s);
            off += chunk;
            idx++;
        }
        return last;
    }

    void destroy() {
        for (int i = 0; i < 2; i++) {
            if (done[i]) { cudaEventSynchronize(done[i]); cudaEventDestroy(done[i]); done[i] = nullptr; }
            if (buf[i]) { cudaFreeHost(buf[i]); buf[i] = nullptr; }
        }
    }
};

// Active stager for current upload pass (nullptr = use plain cudaMemcpyAsync)
static PinnedStager* g_stager = nullptr;

// H2D copy that routes through pinned staging when available
static cudaError_t h2d_copy(void* dst, const void* src, size_t n, cudaStream_t s) {
    if (g_stager) return g_stager->copy(dst, src, n, s);
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
    if (cached >= 0) return cached;
    std::ifstream f("/proc/version");
    if (f) {
        std::string line;
        std::getline(f, line);
        cached = (line.find("microsoft") != std::string::npos ||
                  line.find("Microsoft") != std::string::npos ||
                  line.find("WSL") != std::string::npos) ? 1 : 0;
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
    uint16_t exp  = (h >> 10) & 0x1F;
    uint16_t man  = h & 0x3FF;

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
    int      f_exp  = static_cast<int>((fbits >> 23) & 0xFF) - 127;
    uint32_t f_man  = fbits & 0x7FFFFF;

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
    uint32_t round_bit = (f_man >> 12) & 1;   // bit 12 (first discarded bit)
    uint32_t sticky = f_man & 0xFFF;           // bits 11..0 (remaining discarded bits)
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

static bool upload_weight(Tensor& weight, GGMLQuantType qtype,
                          Tensor& scales_out,
                          DType compute_dtype,
                          cudaStream_t stream,
                          std::vector<void*>& gpu_allocs,
                          bool raw_quant = true) {
    if (weight.data == nullptr || weight.on_device) return true;
    if (weight.ndim < 1) return true;

    int64_t n_elements = weight.numel();
    if (n_elements == 0) return true;

    // ---- Q4_0 ----
    if (qtype == GGMLQuantType::Q4_0) {
        if (weight.ndim < 2) {
            IMP_LOG_WARN("Q4_0 weight has < 2 dims, skipping upload");
            return false;
        }

        int64_t N = weight.shape[0]; // out_features (rows)
        int64_t K = weight.shape[1]; // in_features (cols), logical

        // Raw upload: keep quantized bytes on GPU for dp4a GEMV decode path.
        // Prefill uses fp16_cache or on-the-fly dequant_gpu → cuBLAS GEMM.
        if (raw_quant) {
            size_t raw_bytes = static_cast<size_t>(N) * ggml_quant_row_bytes(qtype, K);
            void* d_data = nullptr;
            checked_cuda_malloc(&d_data, raw_bytes);
            if (!d_data) return false;
            h2d_copy(d_data, weight.data, raw_bytes, stream);
            gpu_allocs.push_back(d_data);

            // Logical shape [N, K] — qtype tells executor data is raw quantized
            int64_t new_shape[4] = {N, K, 0, 0};
            weight = Tensor(d_data, DType::FP16, 2, new_shape, true);
            return true;
        }

        // Split upload fallback: separate nibbles + scales for quant_gemm_int4.
        int blocks_per_row = static_cast<int>(K) / 32;
        int num_groups     = blocks_per_row;
        int half_K         = static_cast<int>(K) / 2;

        // GGML Q4_0 block format: 18 bytes per block (2 fp16 scale + 16 nibbles)
        static constexpr size_t Q4_0_BLOCK_SIZE = 18;

        size_t nibbles_bytes = static_cast<size_t>(N) * half_K;
        size_t scales_count  = static_cast<size_t>(N) * num_groups;

        std::vector<uint8_t>  h_nibbles(nibbles_bytes);
        std::vector<uint16_t> h_scales(scales_count); // raw FP16 bits

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
        checked_cuda_malloc(&d_nibbles, nibbles_bytes);
        if (!d_nibbles) return false;
        h2d_copy(d_nibbles, h_nibbles.data(), nibbles_bytes, stream);
        gpu_allocs.push_back(d_nibbles);

        // Upload scales to GPU
        void* d_scales = nullptr;
        size_t scales_bytes = scales_count * sizeof(uint16_t);
        checked_cuda_malloc(&d_scales, scales_bytes);
        if (!d_scales) { cudaFree(d_nibbles); return false; }
        h2d_copy(d_scales, h_scales.data(), scales_bytes, stream);
        gpu_allocs.push_back(d_scales);

        // Update weight tensor to point to packed nibbles on GPU
        int64_t new_shape[4] = {N, static_cast<int64_t>(half_K), 0, 0};
        weight = Tensor(d_nibbles, DType::INT4, 2, new_shape, true);

        // Set scales output
        int64_t scales_shape[4] = {N, static_cast<int64_t>(num_groups), 0, 0};
        scales_out = Tensor(d_scales, DType::FP16, 2, scales_shape, true);

        return true;
    }

    // ---- Q8_0 ----
    if (qtype == GGMLQuantType::Q8_0) {
        if (weight.ndim < 2) {
            IMP_LOG_WARN("Q8_0 weight has < 2 dims, skipping upload");
            return false;
        }

        int64_t N = weight.shape[0];
        int64_t K = weight.shape[1];

        // Raw upload: keep quantized bytes on GPU, dequant on-the-fly in executor
        if (raw_quant) {
            size_t raw_bytes = static_cast<size_t>(N) * ggml_quant_row_bytes(qtype, K);
            void* d_data = nullptr;
            checked_cuda_malloc(&d_data, raw_bytes);
            if (!d_data) return false;
            h2d_copy(d_data, weight.data, raw_bytes, stream);
            gpu_allocs.push_back(d_data);

            // Logical shape [N, K] — qtype tells executor data is raw quantized
            int64_t new_shape[4] = {N, K, 0, 0};
            weight = Tensor(d_data, DType::FP16, 2, new_shape, true);
            return true;
        }

        // CPU dequant fallback: decode to FP16 on host, upload
        int blocks_per_row = static_cast<int>(K) / 32;
        static constexpr size_t Q8_0_BLOCK_SIZE = 34; // 2 (fp16 scale) + 32 (int8 quants)

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
        checked_cuda_malloc(&d_data, bytes);
        if (!d_data) return false;
        h2d_copy(d_data, h_fp16.data(), bytes, stream);
        gpu_allocs.push_back(d_data);

        int64_t new_shape[4] = {N, K, 0, 0};
        weight = Tensor(d_data, DType::FP16, 2, new_shape, true);
        return true;
    }

    // ---- Q6_K ----
    if (qtype == GGMLQuantType::Q6_K) {
        if (weight.ndim < 2) {
            IMP_LOG_WARN("Q6_K weight has < 2 dims, skipping upload");
            return false;
        }

        int64_t N = weight.shape[0];
        int64_t K = weight.shape[1];

        // Raw upload: keep quantized bytes on GPU, dequant on-the-fly in executor
        if (raw_quant) {
            size_t raw_bytes = static_cast<size_t>(N) * ggml_quant_row_bytes(qtype, K);
            void* d_data = nullptr;
            checked_cuda_malloc(&d_data, raw_bytes);
            if (!d_data) return false;
            h2d_copy(d_data, weight.data, raw_bytes, stream);
            gpu_allocs.push_back(d_data);

            int64_t new_shape[4] = {N, K, 0, 0};
            weight = Tensor(d_data, DType::FP16, 2, new_shape, true);
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

                const uint8_t* ql     = block_ptr;
                const uint8_t* qh     = block_ptr + 128;
                const int8_t*  scales  = reinterpret_cast<const int8_t*>(block_ptr + 192);
                uint16_t d_bits;
                std::memcpy(&d_bits, block_ptr + 208, 2);
                float d = fp16_to_float(d_bits);

                for (int i = 0; i < 256; ++i) {
                    int group  = i / 128;
                    int within = i % 128;
                    int quad   = within / 32;
                    int l      = within % 32;

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
        checked_cuda_malloc(&d_data, bytes);
        if (!d_data) return false;
        h2d_copy(d_data, h_fp16.data(), bytes, stream);
        gpu_allocs.push_back(d_data);

        int64_t new_shape[4] = {N, K, 0, 0};
        weight = Tensor(d_data, DType::FP16, 2, new_shape, true);
        return true;
    }

    // ---- General quantized types (Q5_0, Q5_1, Q4_K, etc.) ----
    // Any type supported by dequant_gpu that wasn't handled above.
    if (dequant_gpu_supported(qtype) && weight.ndim >= 2) {
        int64_t N = weight.shape[0];
        int64_t K = weight.shape[1];

        if (raw_quant) {
            // Upload raw quantized bytes — executor dequants on-the-fly
            size_t raw_bytes = static_cast<size_t>(N) * ggml_quant_row_bytes(qtype, K);
            void* d_data = nullptr;
            checked_cuda_malloc(&d_data, raw_bytes);
            if (!d_data) return false;
            cudaError_t cpy_err = h2d_copy(d_data, weight.data, raw_bytes, stream);
            if (cpy_err != cudaSuccess) {
                IMP_LOG_ERROR("h2d_copy failed for qtype=%u [%ldx%ld] %zu bytes: %s",
                              (unsigned)qtype, (long)N, (long)K, raw_bytes,
                              cudaGetErrorString(cpy_err));
            }
            gpu_allocs.push_back(d_data);
            IMP_LOG_DEBUG("Upload raw qtype=%u [%ldx%ld] %zu bytes -> GPU %p",
                          (unsigned)qtype, (long)N, (long)K, raw_bytes, d_data);
            int64_t new_shape[4] = {N, K, 0, 0};
            weight = Tensor(d_data, DType::FP16, 2, new_shape, true);
            return true;
        } else {
            // Dequant on GPU: upload raw → dequant to FP16 → free raw
            size_t raw_bytes = static_cast<size_t>(N) * ggml_quant_row_bytes(qtype, K);
            void* d_raw = nullptr;
            checked_cuda_malloc(&d_raw, raw_bytes);
            if (!d_raw) return false;
            h2d_copy(d_raw, weight.data, raw_bytes, stream);

            size_t fp16_bytes = static_cast<size_t>(N) * K * sizeof(uint16_t);
            void* d_fp16 = nullptr;
            checked_cuda_malloc(&d_fp16, fp16_bytes);
            if (!d_fp16) { cudaFree(d_raw); return false; }

            dequant_gpu(d_raw, d_fp16, qtype, static_cast<int>(N),
                        static_cast<int>(K), stream);
            cudaStreamSynchronize(stream);
            cudaFree(d_raw);
            gpu_allocs.push_back(d_fp16);

            weight = Tensor(d_fp16, DType::FP16, weight.ndim, weight.shape, true);
            return true;
        }
    }

    // ---- F16 / BF16: direct upload ----
    if (qtype == GGMLQuantType::F16 || qtype == GGMLQuantType::BF16) {
        size_t bytes = weight.nbytes();
        void* d_data = nullptr;
        checked_cuda_malloc(&d_data, bytes);
        if (!d_data) return false;
        h2d_copy(d_data, weight.data, bytes, stream);
        gpu_allocs.push_back(d_data);

        weight.data = d_data;
        weight.on_device = true;
        return true;
    }

    // ---- F32: convert to FP16 on host, then upload ----
    if (qtype == GGMLQuantType::F32 || qtype == GGMLQuantType::NONE) {
        // NONE maps to F32 (both are enum value 0)
        if (weight.dtype != DType::FP32) {
            // If it's not actually FP32 data, just do a direct upload
            size_t bytes = weight.nbytes();
            void* d_data = nullptr;
            checked_cuda_malloc(&d_data, bytes);
            if (!d_data) return false;
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
        checked_cuda_malloc(&d_data, bytes);
        if (!d_data) return false;
        h2d_copy(d_data, h_fp16.data(), bytes, stream);
        gpu_allocs.push_back(d_data);

        weight = Tensor(d_data, DType::FP16, weight.ndim, weight.shape, true);
        return true;
    }

    IMP_LOG_WARN("Unsupported quant type %u for GPU upload, skipping",
                 static_cast<unsigned>(qtype));
    return false;
}

// ---------------------------------------------------------------------------
// Helper: upload a weight tensor that has no associated quant type
// (e.g., norm weights, embedding). We detect the dtype from the tensor.
// ---------------------------------------------------------------------------

static bool upload_unquantized_weight(Tensor& weight,
                                      GGMLQuantType qtype,
                                      DType compute_dtype,
                                      cudaStream_t stream,
                                      std::vector<void*>& gpu_allocs,
                                      bool raw_quant = true) {
    Tensor dummy_scales;
    return upload_weight(weight, qtype, dummy_scales, compute_dtype,
                         stream, gpu_allocs, raw_quant);
}

// ---------------------------------------------------------------------------
// Model::estimate_expert_bytes
// ---------------------------------------------------------------------------

size_t Model::estimate_expert_bytes() const {
    size_t total = 0;
    for (int i = 0; i < n_layers(); ++i) {
        const TransformerLayer& L = layers_[i];
        auto add_packed = [&](const Tensor& p, GGMLQuantType qt) {
            if (!p.data || p.ndim < 3 || !dequant_gpu_supported(qt)) return;
            size_t row_bytes = ggml_quant_row_bytes(qt, p.shape[2]);
            total += static_cast<size_t>(p.shape[0]) * p.shape[1] * row_bytes;
        };
        add_packed(L.expert_gate_packed, L.expert_gate_qtype);
        add_packed(L.expert_up_packed, L.expert_up_qtype);
        add_packed(L.expert_down_packed, L.expert_down_qtype);
    }
    return total;
}

// ---------------------------------------------------------------------------
// Upload context: bundles the repeated parameters needed by all upload helpers.
// Passed by reference to avoid >8 params on every helper call.
// ---------------------------------------------------------------------------
struct UploadCtx {
    DType compute_dtype;
    cudaStream_t stream;
    std::vector<void*>& gpu_allocs;
    std::vector<void*>& host_pinned;
    std::vector<void*>& host_pinned_allocs;
};

// ---------------------------------------------------------------------------
// UPLOAD_OR_FAIL / UPLOAD_UNQUANT_OR_FAIL: reduces the per-weight boilerplate
// of calling upload_weight() + error log + early return.
// ---------------------------------------------------------------------------
#define UPLOAD_OR_FAIL(tensor, qtype, scales, msg, layer_idx, ctx) \
    do { \
        if (!upload_weight((tensor), (qtype), (scales), (ctx).compute_dtype, \
                           (ctx).stream, (ctx).gpu_allocs)) { \
            IMP_LOG_ERROR("Failed to upload " msg " for layer %d", (layer_idx)); \
            return false; \
        } \
    } while (0)

#define UPLOAD_OR_FAIL_RAW(tensor, qtype, scales, raw, msg, layer_idx, ctx) \
    do { \
        if (!upload_weight((tensor), (qtype), (scales), (ctx).compute_dtype, \
                           (ctx).stream, (ctx).gpu_allocs, (raw))) { \
            IMP_LOG_ERROR("Failed to upload " msg " for layer %d", (layer_idx)); \
            return false; \
        } \
    } while (0)

#define UPLOAD_UNQUANT_OR_FAIL(tensor, msg, layer_idx, ctx) \
    do { \
        if ((tensor).data && !(tensor).on_device) { \
            if (!upload_unquantized_weight((tensor), GGMLQuantType::NONE, \
                                           (ctx).compute_dtype, (ctx).stream, \
                                           (ctx).gpu_allocs)) { \
                IMP_LOG_ERROR("Failed to upload " msg " for layer %d", (layer_idx)); \
                return false; \
            } \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// upload_embeddings_and_output: token embedding, output norm, output projection
// ---------------------------------------------------------------------------
static bool upload_embeddings_and_output(
        Tensor& tok_emb, GGMLQuantType& tok_emb_qtype,
        Tensor& out_norm, GGMLQuantType out_norm_qtype,
        Tensor& out_proj, GGMLQuantType out_proj_qtype,
        const UploadCtx& ctx) {
    // Upload token embedding
    // Embedding lookup only supports Q8_0/Q6_K natively; other quant types
    // need to be dequanted to FP16 (raw_quant=false) so the standard FP16
    // embedding gather works.
    const void* tok_emb_host_ptr = tok_emb.data;  // save for weight-tying check below
    if (tok_emb.data && !tok_emb.on_device) {
        bool emb_raw = (tok_emb_qtype == GGMLQuantType::Q8_0 ||
                        tok_emb_qtype == GGMLQuantType::Q6_K);
        if (!upload_unquantized_weight(tok_emb, tok_emb_qtype, ctx.compute_dtype,
                                       ctx.stream, ctx.gpu_allocs, emb_raw)) {
            IMP_LOG_ERROR("Failed to upload token embedding");
            return false;
        }
        // If we dequanted to FP16, update the qtype so embedding_lookup
        // uses the FP16 path.
        if (!emb_raw && tok_emb.dtype == DType::FP16) {
            tok_emb_qtype = GGMLQuantType::F16;
        }
    }

    // Upload output norm
    if (out_norm.data && !out_norm.on_device) {
        if (!upload_unquantized_weight(out_norm, out_norm_qtype, ctx.compute_dtype,
                                       ctx.stream, ctx.gpu_allocs)) {
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
        // Checking the host pointer prevents incorrectly sharing when a model has
        // separate output.weight and token_embd.weight tensors of the same qtype.
        bool actually_tied = (out_proj.data == tok_emb_host_ptr &&
                              out_proj_qtype == tok_emb_qtype);
        if (actually_tied && tok_emb.on_device) {
            out_proj = tok_emb;
            IMP_LOG_INFO("Output projection shares GPU data with token embedding (weight tying)");
        } else {
            bool raw_ok = (out_proj_qtype == GGMLQuantType::Q6_K ||
                           out_proj_qtype == GGMLQuantType::Q8_0 ||
                           out_proj_qtype == GGMLQuantType::Q4_0);
            if (!upload_unquantized_weight(out_proj, out_proj_qtype, ctx.compute_dtype,
                                           ctx.stream, ctx.gpu_allocs,
                                           /*raw_quant=*/raw_ok)) {
                IMP_LOG_ERROR("Failed to upload output projection");
                return false;
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// upload_gptq_weight: dequantize a GPTQ-packed weight to FP16 on GPU.
// Uploads qweight/qzeros/scales/g_idx to temporary GPU buffers, runs the
// dequant kernel, then frees the temporaries.  Sets output tensor to point
// to the resulting FP16 weight on GPU.
// ---------------------------------------------------------------------------
static bool upload_gptq_weight(const TransformerLayer::GPTQWeight& gptq,
                               Tensor& output,
                               cudaStream_t stream,
                               std::vector<void*>& gpu_allocs) {
    if (!gptq.qweight.data || !gptq.scales.data) return false;
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
    if (checked_cuda_malloc(reinterpret_cast<void**>(&d_qweight), qw_bytes) != cudaSuccess || !d_qweight) {
        IMP_LOG_ERROR("GPTQ: failed to allocate qweight (%zu bytes)", qw_bytes);
        return false;
    }
    h2d_copy(d_qweight, gptq.qweight.data, qw_bytes, stream);

    // 2. Upload qzeros to GPU
    int32_t* d_qzeros = nullptr;
    if (gptq.qzeros.data) {
        size_t qz_bytes = static_cast<size_t>(gptq.qzeros.shape[0]) * gptq.qzeros.shape[1] * sizeof(int32_t);
        if (checked_cuda_malloc(reinterpret_cast<void**>(&d_qzeros), qz_bytes) != cudaSuccess || !d_qzeros) {
            IMP_LOG_ERROR("GPTQ: failed to allocate qzeros");
            cudaFree(d_qweight);
            return false;
        }
        h2d_copy(d_qzeros, gptq.qzeros.data, qz_bytes, stream);
    }

    // 3. Upload scales to GPU
    size_t sc_bytes = static_cast<size_t>(gptq.scales.shape[0]) * gptq.scales.shape[1] * sizeof(half);
    half* d_scales = nullptr;
    if (checked_cuda_malloc(reinterpret_cast<void**>(&d_scales), sc_bytes) != cudaSuccess || !d_scales) {
        IMP_LOG_ERROR("GPTQ: failed to allocate scales");
        cudaFree(d_qweight);
        if (d_qzeros) cudaFree(d_qzeros);
        return false;
    }
    h2d_copy(d_scales, gptq.scales.data, sc_bytes, stream);

    // 4. Upload g_idx to GPU (optional, for desc_act reordering)
    int32_t* d_g_idx = nullptr;
    if (gptq.g_idx.data) {
        size_t gi_bytes = static_cast<size_t>(K) * sizeof(int32_t);
        if (checked_cuda_malloc(reinterpret_cast<void**>(&d_g_idx), gi_bytes) != cudaSuccess || !d_g_idx) {
            IMP_LOG_WARN("GPTQ: failed to allocate g_idx, falling back to sequential groups");
        } else {
            h2d_copy(d_g_idx, gptq.g_idx.data, gi_bytes, stream);
        }
    }

    // 5. Allocate FP16 output [N, K]
    size_t out_bytes = static_cast<size_t>(N) * K * sizeof(half);
    half* d_out = nullptr;
    if (checked_cuda_malloc(reinterpret_cast<void**>(&d_out), out_bytes) != cudaSuccess || !d_out) {
        IMP_LOG_ERROR("GPTQ: failed to allocate output (%zu bytes)", out_bytes);
        cudaFree(d_qweight);
        if (d_qzeros) cudaFree(d_qzeros);
        cudaFree(d_scales);
        if (d_g_idx) cudaFree(d_g_idx);
        return false;
    }

    // 6. Run dequantization kernel
    dequant_gptq4(d_out, d_qweight, d_qzeros, d_scales, d_g_idx,
                  N, K, gptq.group_size, stream);

    // 7. Sync and free temporary GPU buffers
    cudaStreamSynchronize(stream);
    cudaFree(d_qweight);
    if (d_qzeros) cudaFree(d_qzeros);
    cudaFree(d_scales);
    if (d_g_idx) cudaFree(d_g_idx);

    // 8. Set output tensor
    int64_t out_shape[4] = {N, K, 0, 0};
    output = Tensor(d_out, DType::FP16, 2, out_shape, true);
    gpu_allocs.push_back(d_out);

    return true;
}

// ---------------------------------------------------------------------------
// upload_layer_attention_weights: wq/wk/wv/wo + norms + biases for one layer
// ---------------------------------------------------------------------------
static bool upload_layer_attention_weights(TransformerLayer& L, int i,
                                           const UploadCtx& ctx) {
    // Attention weights — try regular upload first, fall back to GPTQ dequant
    UPLOAD_OR_FAIL(L.wq, L.wq_qtype, L.wq_scales, "wq", i, ctx);
    UPLOAD_OR_FAIL(L.wk, L.wk_qtype, L.wk_scales, "wk", i, ctx);
    UPLOAD_OR_FAIL(L.wv, L.wv_qtype, L.wv_scales, "wv", i, ctx);
    UPLOAD_OR_FAIL(L.wo, L.wo_qtype, L.wo_scales, "wo", i, ctx);

    // GPTQ fallback: if regular weight is missing but GPTQ tensors are present
    struct { Tensor& w; TransformerLayer::GPTQWeight& gptq; const char* name; } attn_gptq[] = {
        {L.wq, L.gptq_q, "q_proj"}, {L.wk, L.gptq_k, "k_proj"},
        {L.wv, L.gptq_v, "v_proj"}, {L.wo, L.gptq_o, "o_proj"},
    };
    for (auto& [w, gptq, name] : attn_gptq) {
        if (!w.on_device && gptq.qweight.data) {
            if (!upload_gptq_weight(gptq, w, ctx.stream, ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to dequant GPTQ %s for layer %d", name, i);
                return false;
            }
            IMP_LOG_DEBUG("GPTQ dequant %s layer %d -> [%lld, %lld] FP16",
                         name, i, w.shape[0], w.shape[1]);
        }
    }

    // Attention norm (typically F32/F16, no quant)
    UPLOAD_UNQUANT_OR_FAIL(L.attn_norm, "attn_norm", i, ctx);

    // QK-norm weights (Qwen3-style per-head RMSNorm, F32 [head_dim])
    UPLOAD_UNQUANT_OR_FAIL(L.attn_q_norm, "attn_q_norm", i, ctx);
    UPLOAD_UNQUANT_OR_FAIL(L.attn_k_norm, "attn_k_norm", i, ctx);

    // Attention biases (Qwen2-style Q/K/V biases, F32)
    for (auto* bias : {&L.q_bias, &L.k_bias, &L.v_bias}) {
        if (bias->data && !bias->on_device) {
            if (!upload_unquantized_weight(*bias, GGMLQuantType::NONE,
                                           ctx.compute_dtype, ctx.stream,
                                           ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload attention bias for layer %d", i);
                return false;
            }
        }
    }

    // Post-layer norms (Gemma-3 style)
    for (auto* norm : {&L.post_attn_norm, &L.post_ffn_norm}) {
        if (norm->data && !norm->on_device) {
            if (!upload_unquantized_weight(*norm, GGMLQuantType::NONE,
                                           ctx.compute_dtype, ctx.stream,
                                           ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload post-layer norm for layer %d", i);
                return false;
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// upload_layer_ffn_weights: w_gate/w_up/w_down + norms + MoE routing +
//                           shared experts for one layer
// ---------------------------------------------------------------------------
static bool upload_layer_ffn_weights(TransformerLayer& L, int i,
                                     const UploadCtx& ctx) {
    // FFN weights (dense path)
    UPLOAD_OR_FAIL(L.w_gate, L.w_gate_qtype, L.w_gate_scales, "w_gate", i, ctx);
    UPLOAD_OR_FAIL(L.w_up, L.w_up_qtype, L.w_up_scales, "w_up", i, ctx);
    UPLOAD_OR_FAIL(L.w_down, L.w_down_qtype, L.w_down_scales, "w_down", i, ctx);

    // GPTQ fallback for FFN weights
    struct { Tensor& w; TransformerLayer::GPTQWeight& gptq; const char* name; } ffn_gptq[] = {
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
            IMP_LOG_DEBUG("GPTQ dequant %s layer %d -> [%lld, %lld] FP16",
                         name, i, w.shape[0], w.shape[1]);
        }
    }

    // FFN norm (typically F32/F16, no quant)
    UPLOAD_UNQUANT_OR_FAIL(L.ffn_norm, "ffn_norm", i, ctx);

    // MoE gate (routing weights, typically F32/F16)
    UPLOAD_UNQUANT_OR_FAIL(L.moe_gate, "moe_gate", i, ctx);

    // Router bias (Nemotron MoE)
    UPLOAD_UNQUANT_OR_FAIL(L.moe_router_bias, "moe_router_bias", i, ctx);

    // Shared expert weights (Nemotron/DeepSeek style)
    if (L.w_up_shared.data && !L.w_up_shared.on_device) {
        Tensor dummy_scales;
        UPLOAD_OR_FAIL(L.w_up_shared, L.w_up_shared_qtype, dummy_scales,
                       "w_up_shared", i, ctx);
    }
    if (L.w_down_shared.data && !L.w_down_shared.on_device) {
        Tensor dummy_scales;
        UPLOAD_OR_FAIL(L.w_down_shared, L.w_down_shared_qtype, dummy_scales,
                       "w_down_shared", i, ctx);
    }
    if (L.w_gate_shared.data && !L.w_gate_shared.on_device) {
        Tensor dummy_scales;
        UPLOAD_OR_FAIL(L.w_gate_shared, L.w_gate_shared_qtype, dummy_scales,
                       "w_gate_shared", i, ctx);
    }

    return true;
}

// ---------------------------------------------------------------------------
// upload_layer_ssm_weights: SSM weights for one layer (Mamba2/Nemotron-H)
// ---------------------------------------------------------------------------
static bool upload_layer_ssm_weights(TransformerLayer& L, int i,
                                     const UploadCtx& ctx) {
    // SSM weights (Mamba2)
    if (L.ssm_in.data && !L.ssm_in.on_device) {
        UPLOAD_OR_FAIL(L.ssm_in, L.ssm_in_qtype, L.wq_scales, "ssm_in", i, ctx);
    }
    if (L.ssm_out.data && !L.ssm_out.on_device) {
        UPLOAD_OR_FAIL(L.ssm_out, L.ssm_out_qtype, L.wo_scales, "ssm_out", i, ctx);
    }
    // SSM tensors that convert to compute_dtype (FP16): conv1d weights, norm
    for (Tensor* t : {&L.ssm_conv1d_w, &L.ssm_conv1d_b, &L.ssm_norm_w}) {
        if (t->data && !t->on_device) {
            if (!upload_unquantized_weight(*t, GGMLQuantType::NONE, ctx.compute_dtype,
                                           ctx.stream, ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload SSM tensor for layer %d", i);
                return false;
            }
        }
    }
    // SSM tensors that MUST stay F32: A_log, D, dt_bias (scan kernel uses float*)
    for (Tensor* t : {&L.ssm_a, &L.ssm_d, &L.ssm_dt_b}) {
        if (t->data && !t->on_device) {
            size_t bytes = t->nbytes();
            void* d_data = nullptr;
            checked_cuda_malloc(&d_data, bytes);
            if (!d_data) {
                IMP_LOG_ERROR("Failed to allocate GPU memory for SSM F32 tensor in layer %d", i);
                return false;
            }
            h2d_copy(d_data, t->data, bytes, ctx.stream);
            ctx.gpu_allocs.push_back(d_data);
            t->data = d_data;
            t->on_device = true;
        }
    }

    // Gated DeltaNet (GDN) weights (Qwen3.5)
    if (L.gdn_gate.data && !L.gdn_gate.on_device) {
        Tensor dummy_scales;
        UPLOAD_OR_FAIL(L.gdn_gate, L.gdn_gate_qtype, dummy_scales,
                       "gdn_gate", i, ctx);
    }
    // GDN alpha/beta: used in direct gemm() for small projections.
    // Must be FP16 on device (NOT raw quantized) for cuBLAS GEMM.
    if (L.gdn_alpha.data && !L.gdn_alpha.on_device) {
        Tensor dummy_scales;
        UPLOAD_OR_FAIL_RAW(L.gdn_alpha, L.gdn_alpha_qtype, dummy_scales,
                           /*raw_quant=*/false, "gdn_alpha", i, ctx);
    }
    if (L.gdn_beta.data && !L.gdn_beta.on_device) {
        Tensor dummy_scales;
        UPLOAD_OR_FAIL_RAW(L.gdn_beta, L.gdn_beta_qtype, dummy_scales,
                           /*raw_quant=*/false, "gdn_beta", i, ctx);
    }

    return true;
}

// ---------------------------------------------------------------------------
// upload_expert_weights: MoE expert weight upload for all layers (Pass 2).
// Handles packed 3D tensors and per-expert 2D tensors.
// ---------------------------------------------------------------------------
static bool upload_expert_weights(
        std::vector<TransformerLayer>& layers, int n_layers,
        size_t expert_reserve_bytes,
        const UploadCtx& ctx) {

    // Compute per-layer expert weight costs
    size_t total_expert_bytes = 0;
    std::vector<size_t> layer_expert_bytes(n_layers, 0);
    for (int i = 0; i < n_layers; ++i) {
        const TransformerLayer& L = layers[i];
        auto add_packed = [&](const Tensor& p, GGMLQuantType qt) {
            if (!p.data || p.ndim < 3 || !dequant_gpu_supported(qt)) return;
            size_t row_bytes = ggml_quant_row_bytes(qt, p.shape[2]);
            size_t bytes = static_cast<size_t>(p.shape[0]) * p.shape[1] * row_bytes;
            layer_expert_bytes[i] += bytes;
            total_expert_bytes += bytes;
        };
        add_packed(L.expert_gate_packed, L.expert_gate_qtype);
        add_packed(L.expert_up_packed, L.expert_up_qtype);
        add_packed(L.expert_down_packed, L.expert_down_qtype);
    }

    // Decide which expert layers to upload based on actual remaining VRAM
    std::vector<bool> experts_upload_layer(n_layers, false);
    if (total_expert_bytes > 0) {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        // Reserve for KV cache + SSM state + activation workspace + FP16 cache.
        // Engine passes the exact reserve based on computed workspace sizes.
        // CUDA driver overhead on WSL2/WDDM: cudaMalloc alignment, page tables,
        // and WDDM shared memory management consume ~30% beyond the requested size.
        // Empirical on RTX 5090 WSL2: 22 GiB expert alloc leaves 0 MiB from 28.7 GiB free.
        size_t overhead = free_mem * 3 / 10;  // 30% of available VRAM
        size_t total_reserve = expert_reserve_bytes + overhead;
        size_t budget = (free_mem > total_reserve) ? (free_mem - total_reserve) : 0;

        if (budget >= total_expert_bytes) {
            // All experts fit
            for (int i = 0; i < n_layers; ++i) {
                if (layer_expert_bytes[i] > 0) experts_upload_layer[i] = true;
            }
            IMP_LOG_INFO("Expert weights: %.2f GiB -> uploading ALL to GPU "
                         "(%.2f GiB free, %.2f GiB reserve)",
                         total_expert_bytes / (1024.0*1024.0*1024.0),
                         free_mem / (1024.0*1024.0*1024.0),
                         expert_reserve_bytes / (1024.0*1024.0*1024.0));
        } else {
            // Partial upload: greedily upload layers until budget exhausted
            size_t uploaded = 0;
            int n_uploaded = 0, n_total_moe = 0;
            for (int i = 0; i < n_layers; ++i) {
                if (layer_expert_bytes[i] == 0) continue;
                n_total_moe++;
                if (uploaded + layer_expert_bytes[i] <= budget) {
                    experts_upload_layer[i] = true;
                    uploaded += layer_expert_bytes[i];
                    n_uploaded++;
                }
            }
            IMP_LOG_INFO("Expert weights: %.2f GiB total, uploading %d/%d MoE layers "
                         "(%.2f GiB on GPU, %.2f GiB on host, %.2f GiB free, "
                         "%.2f GiB reserve)",
                         total_expert_bytes / (1024.0*1024.0*1024.0),
                         n_uploaded, n_total_moe,
                         uploaded / (1024.0*1024.0*1024.0),
                         (total_expert_bytes - uploaded) / (1024.0*1024.0*1024.0),
                         free_mem / (1024.0*1024.0*1024.0),
                         expert_reserve_bytes / (1024.0*1024.0*1024.0));
        }
    }

    // Upload expert weights for each layer
    for (int i = 0; i < n_layers; ++i) {
        TransformerLayer& L = layers[i];

        // MoE expert weights -- two paths:
        // A) Packed 3D tensors (*_exps):
        //    - For quantized types (Q6_K, Q8_0, Q4_0): upload raw bytes to GPU,
        //      keep packed tensor. Dequant happens on-the-fly in run_moe_ffn.
        //    - For F16/BF16/F32: dequant/upload and slice into per-expert views.
        // B) Per-expert 2D tensors: upload individually (legacy per-expert GGUF format)

        auto upload_packed_experts = [&](Tensor& packed, GGMLQuantType qtype,
                                         std::vector<Tensor>& expert_vec,
                                         const char* name) -> bool {
            if (!packed.data || packed.ndim < 3) return true;  // nothing to do

            int n_experts = static_cast<int>(packed.shape[0]);
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];

            // Path A1: Quantized types -- upload raw bytes to GPU if they fit,
            // otherwise keep on host (mmap'd) with optional pinning for H2D.
            if (dequant_gpu_supported(qtype)) {
                size_t row_bytes = ggml_quant_row_bytes(qtype, cols);
                size_t expert_raw = static_cast<size_t>(rows) * row_bytes;
                size_t total_raw = static_cast<size_t>(n_experts) * expert_raw;

                if (experts_upload_layer[i]) {
                    // Upload raw quantized bytes to GPU (respects VRAM reserve)
                    void* gpu_ptr = nullptr;
                    cudaError_t err = checked_cuda_malloc(&gpu_ptr, total_raw);
                    if (err == cudaSuccess) {
                        cudaError_t cpy_err = h2d_copy(gpu_ptr, packed.data, total_raw, ctx.stream);
                        if (cpy_err != cudaSuccess) {
                            IMP_LOG_ERROR("  %s: h2d_copy failed: %s", name, cudaGetErrorString(cpy_err));
                            cudaFree(gpu_ptr);
                            return false;
                        }
                        packed.data = gpu_ptr;
                        packed.on_device = true;
                        ctx.gpu_allocs.push_back(gpu_ptr);
                        IMP_LOG_DEBUG("  %s: %d experts uploaded to GPU (%.2f MiB)",
                                      name, n_experts, total_raw / (1024.0 * 1024.0));
                        return true;
                    }
                    // cudaMalloc failed — fall through to host path
                    IMP_LOG_WARN("  %s: cudaMalloc failed for %.2f MiB, falling back to host",
                                 name, total_raw / (1024.0 * 1024.0));
                }

                // Host path: pin memory for fast async DMA H2D during decode.
                if (is_wsl2()) {
                    // WSL2: cudaHostRegister fails on mmap'd memory. Instead,
                    // allocate fresh pinned memory and copy mmap'd data there.
                    // This enables true async DMA H2D (no per-token CPU memcpy).
                    void* pinned_buf = nullptr;
                    cudaError_t pin_err = cudaHostAlloc(&pinned_buf, total_raw,
                                                         cudaHostAllocDefault);
                    if (pin_err == cudaSuccess) {
                        memcpy(pinned_buf, packed.data, total_raw);
                        packed.data = pinned_buf;
                        ctx.host_pinned_allocs.push_back(pinned_buf);
                        IMP_LOG_INFO("  %s: WSL2 pinned copy (%.2f MiB, DMA-ready)",
                                     name, total_raw / (1024.0 * 1024.0));
                    } else {
                        IMP_LOG_DEBUG("Cleared WSL2 cudaHostAlloc error: %s", cudaGetErrorString(pin_err));
                        cudaGetLastError();  // clear sticky CUDA error state
                        IMP_LOG_INFO("  %s: WSL2 cudaHostAlloc failed, falling back to "
                                     "unpinned mmap (%.2f MiB)", name,
                                     total_raw / (1024.0 * 1024.0));
                    }
                } else {
                    cudaError_t pin_err = cudaHostRegister(packed.data, total_raw,
                                                           cudaHostRegisterReadOnly);
                    if (pin_err == cudaSuccess) {
                        ctx.host_pinned.push_back(packed.data);
                        IMP_LOG_DEBUG("  %s: %d experts, raw %s pinned on host (%.2f MiB)",
                                      name, n_experts,
                                      qtype == GGMLQuantType::Q6_K ? "Q6_K" :
                                      qtype == GGMLQuantType::Q8_0 ? "Q8_0" : "Q4_0",
                                      total_raw / (1024.0 * 1024.0));
                    } else {
                        IMP_LOG_WARN("  %s: cudaHostRegister failed (%s), H2D will be slower",
                                     name, cudaGetErrorString(pin_err));
                    }
                }

                return true;
            }

            // Path A2: Unquantized (F16/BF16/F32) -- dequant to FP16, slice per-expert.
            int64_t flat_shape[4] = {static_cast<int64_t>(n_experts) * rows, cols, 0, 0};
            Tensor flat(packed.data, packed.dtype, 2, flat_shape, packed.on_device);

            Tensor dummy_scales;
            if (!upload_weight(flat, qtype, dummy_scales, ctx.compute_dtype,
                               ctx.stream, ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload packed %s for layer %d", name, i);
                return false;
            }

            expert_vec.resize(n_experts);
            size_t expert_bytes = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
            for (int e = 0; e < n_experts; e++) {
                char* ptr = static_cast<char*>(flat.data) + e * expert_bytes;
                int64_t eshape[4] = {rows, cols, 0, 0};
                expert_vec[e] = Tensor(ptr, DType::FP16, 2, eshape, true);
            }
            packed = Tensor();
            return true;
        };

        if (!upload_packed_experts(L.expert_gate_packed, L.expert_gate_qtype,
                                   L.expert_w_gate, "expert_gate_exps"))
            return false;
        if (!upload_packed_experts(L.expert_up_packed, L.expert_up_qtype,
                                   L.expert_w_up, "expert_up_exps"))
            return false;
        if (!upload_packed_experts(L.expert_down_packed, L.expert_down_qtype,
                                   L.expert_w_down, "expert_down_exps"))
            return false;

        // Path B: per-expert 2D tensors (from per-expert GGUF naming)
        for (size_t e = 0; e < L.expert_w_gate.size(); ++e) {
            if (!L.expert_w_gate[e].data || L.expert_w_gate[e].on_device) continue;
            Tensor dummy_scales;
            if (!upload_weight(L.expert_w_gate[e], L.expert_gate_qtype, dummy_scales,
                               ctx.compute_dtype, ctx.stream, ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload expert_w_gate[%zu] for layer %d", e, i);
                return false;
            }
        }
        for (size_t e = 0; e < L.expert_w_up.size(); ++e) {
            if (!L.expert_w_up[e].data || L.expert_w_up[e].on_device) continue;
            Tensor dummy_scales;
            if (!upload_weight(L.expert_w_up[e], L.expert_up_qtype, dummy_scales,
                               ctx.compute_dtype, ctx.stream, ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload expert_w_up[%zu] for layer %d", e, i);
                return false;
            }
        }
        for (size_t e = 0; e < L.expert_w_down.size(); ++e) {
            if (!L.expert_w_down[e].data || L.expert_w_down[e].on_device) continue;
            Tensor dummy_scales;
            if (!upload_weight(L.expert_w_down[e], L.expert_down_qtype, dummy_scales,
                               ctx.compute_dtype, ctx.stream, ctx.gpu_allocs)) {
                IMP_LOG_ERROR("Failed to upload expert_w_down[%zu] for layer %d", e, i);
                return false;
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Model::upload_weights_gpu
// ---------------------------------------------------------------------------

bool Model::upload_weights_gpu(DType compute_dtype, cudaStream_t stream,
                                size_t expert_reserve_bytes) {
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
        IMP_LOG_INFO("Pinned staging enabled (2x %.0f MiB buffers)",
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
                      free_mem / (1024.0 * 1024.0 * 1024.0),
                      total_mem / (1024.0 * 1024.0 * 1024.0));
    }

    UploadCtx ctx{compute_dtype, stream, gpu_allocations_,
                  host_pinned_, host_pinned_allocs_};

    // --- Embeddings, output norm, output projection ---
    if (!upload_embeddings_and_output(tok_emb_, tok_emb_qtype_,
                                      out_norm_, out_norm_qtype_,
                                      out_proj_, out_proj_qtype_, ctx)) {
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

        if (!upload_layer_attention_weights(L, i, ctx)) return false;
        if (!upload_layer_ffn_weights(L, i, ctx)) return false;

        // (Expert weights are uploaded in Pass 2 below)

        if (!upload_layer_ssm_weights(L, i, ctx)) return false;

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

    // Final sync
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }

    gpu_weights_ready_ = true;
    IMP_LOG_INFO("All model weights uploaded to GPU (%zu allocations)",
                 gpu_allocations_.size());
    return true;
}

#undef UPLOAD_OR_FAIL
#undef UPLOAD_OR_FAIL_RAW
#undef UPLOAD_UNQUANT_OR_FAIL

} // namespace imp
