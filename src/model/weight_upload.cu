#include "model/model.h"
#include "model/gguf_loader.h"
#include "quant/dequant_gpu.h"
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
static constexpr size_t kWeightReserveMiB = 256;  // reserve for KV cache, SSM state, misc

// Cached VRAM state — refreshed once per upload pass instead of per-tensor.
// Eliminates ~500+ cudaMemGetInfo roundtrips during weight upload.
static size_t g_cached_free_mem = 0;
static size_t g_total_allocated = 0;

static cudaError_t checked_cuda_malloc(void** ptr, size_t size) {
    size_t reserve = kWeightReserveMiB << 20;
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
        }
    } staging_guard;

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

    // Upload token embedding
    // Embedding lookup only supports Q8_0/Q6_K natively; other quant types
    // need to be dequanted to FP16 (raw_quant=false) so the standard FP16
    // embedding gather works.
    const void* tok_emb_host_ptr = tok_emb_.data;  // save for weight-tying check below
    if (tok_emb_.data && !tok_emb_.on_device) {
        bool emb_raw = (tok_emb_qtype_ == GGMLQuantType::Q8_0 ||
                        tok_emb_qtype_ == GGMLQuantType::Q6_K);
        if (!upload_unquantized_weight(tok_emb_, tok_emb_qtype_, compute_dtype,
                                       stream, gpu_allocations_, emb_raw)) {
            IMP_LOG_ERROR("Failed to upload token embedding");
            return false;
        }
        // If we dequanted to FP16, update the qtype so embedding_lookup
        // uses the FP16 path.
        if (!emb_raw && tok_emb_.dtype == DType::FP16) {
            tok_emb_qtype_ = GGMLQuantType::F16;
        }
    }

    // Upload output norm
    if (out_norm_.data && !out_norm_.on_device) {
        if (!upload_unquantized_weight(out_norm_, out_norm_qtype_, compute_dtype,
                                       stream, gpu_allocations_)) {
            IMP_LOG_ERROR("Failed to upload output norm");
            return false;
        }
    }

    // Upload output projection — raw Q6_K/Q8_0 for dp4a GEMV (saves ~60% VRAM).
    // Falls back to FP16 dequant for unsupported quant types.
    // For weight-tied models (out_proj == tok_emb), share the GPU data directly.
    if (out_proj_.data && !out_proj_.on_device) {
        // Weight tying: share GPU data only if both point to the same host tensor
        // (i.e. GGUF had no output.weight and the loader aliased out_proj = tok_emb).
        // Checking the host pointer prevents incorrectly sharing when a model has
        // separate output.weight and token_embd.weight tensors of the same qtype.
        bool actually_tied = (out_proj_.data == tok_emb_host_ptr &&
                              out_proj_qtype_ == tok_emb_qtype_);
        if (actually_tied && tok_emb_.on_device) {
            out_proj_ = tok_emb_;
            IMP_LOG_INFO("Output projection shares GPU data with token embedding (weight tying)");
        } else {
            bool raw_ok = (out_proj_qtype_ == GGMLQuantType::Q6_K ||
                           out_proj_qtype_ == GGMLQuantType::Q8_0 ||
                           out_proj_qtype_ == GGMLQuantType::Q4_0);
            if (!upload_unquantized_weight(out_proj_, out_proj_qtype_, compute_dtype,
                                           stream, gpu_allocations_,
                                           /*raw_quant=*/raw_ok)) {
                IMP_LOG_ERROR("Failed to upload output projection");
                return false;
            }
        }
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

        // Attention weights
        if (!upload_weight(L.wq, L.wq_qtype, L.wq_scales, compute_dtype,
                           stream, gpu_allocations_)) {
            IMP_LOG_ERROR("Failed to upload wq for layer %d", i);
            return false;
        }
        if (!upload_weight(L.wk, L.wk_qtype, L.wk_scales, compute_dtype,
                           stream, gpu_allocations_)) {
            IMP_LOG_ERROR("Failed to upload wk for layer %d", i);
            return false;
        }
        if (!upload_weight(L.wv, L.wv_qtype, L.wv_scales, compute_dtype,
                           stream, gpu_allocations_)) {
            IMP_LOG_ERROR("Failed to upload wv for layer %d", i);
            return false;
        }
        if (!upload_weight(L.wo, L.wo_qtype, L.wo_scales, compute_dtype,
                           stream, gpu_allocations_)) {
            IMP_LOG_ERROR("Failed to upload wo for layer %d", i);
            return false;
        }

        // Attention norm (typically F32/F16, no quant)
        if (L.attn_norm.data && !L.attn_norm.on_device) {
            if (!upload_unquantized_weight(L.attn_norm, GGMLQuantType::NONE,
                                           compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload attn_norm for layer %d", i);
                return false;
            }
        }

        // QK-norm weights (Qwen3-style per-head RMSNorm, F32 [head_dim])
        if (L.attn_q_norm.data && !L.attn_q_norm.on_device) {
            if (!upload_unquantized_weight(L.attn_q_norm, GGMLQuantType::NONE,
                                           compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload attn_q_norm for layer %d", i);
                return false;
            }
        }
        if (L.attn_k_norm.data && !L.attn_k_norm.on_device) {
            if (!upload_unquantized_weight(L.attn_k_norm, GGMLQuantType::NONE,
                                           compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload attn_k_norm for layer %d", i);
                return false;
            }
        }

        // Attention biases (Qwen2-style Q/K/V biases, F32)
        for (auto* bias : {&L.q_bias, &L.k_bias, &L.v_bias}) {
            if (bias->data && !bias->on_device) {
                if (!upload_unquantized_weight(*bias, GGMLQuantType::NONE,
                                               compute_dtype, stream, gpu_allocations_)) {
                    IMP_LOG_ERROR("Failed to upload attention bias for layer %d", i);
                    return false;
                }
            }
        }

        // Post-layer norms (Gemma-3 style)
        for (auto* norm : {&L.post_attn_norm, &L.post_ffn_norm}) {
            if (norm->data && !norm->on_device) {
                if (!upload_unquantized_weight(*norm, GGMLQuantType::NONE,
                                               compute_dtype, stream, gpu_allocations_)) {
                    IMP_LOG_ERROR("Failed to upload post-layer norm for layer %d", i);
                    return false;
                }
            }
        }

        // FFN weights (dense path)
        if (!upload_weight(L.w_gate, L.w_gate_qtype, L.w_gate_scales,
                           compute_dtype, stream, gpu_allocations_)) {
            IMP_LOG_ERROR("Failed to upload w_gate for layer %d", i);
            return false;
        }
        if (!upload_weight(L.w_up, L.w_up_qtype, L.w_up_scales,
                           compute_dtype, stream, gpu_allocations_)) {
            IMP_LOG_ERROR("Failed to upload w_up for layer %d", i);
            return false;
        }
        if (!upload_weight(L.w_down, L.w_down_qtype, L.w_down_scales,
                           compute_dtype, stream, gpu_allocations_)) {
            IMP_LOG_ERROR("Failed to upload w_down for layer %d", i);
            return false;
        }

        // FFN norm (typically F32/F16, no quant)
        if (L.ffn_norm.data && !L.ffn_norm.on_device) {
            if (!upload_unquantized_weight(L.ffn_norm, GGMLQuantType::NONE,
                                           compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload ffn_norm for layer %d", i);
                return false;
            }
        }

        // MoE gate (routing weights, typically F32/F16)
        if (L.moe_gate.data && !L.moe_gate.on_device) {
            if (!upload_unquantized_weight(L.moe_gate, GGMLQuantType::NONE,
                                           compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload moe_gate for layer %d", i);
                return false;
            }
        }

        // (Expert weights are uploaded in Pass 2 below)

        // SSM weights (Mamba2)
        if (L.ssm_in.data && !L.ssm_in.on_device) {
            if (!upload_weight(L.ssm_in, L.ssm_in_qtype, L.wq_scales, compute_dtype,
                               stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload ssm_in for layer %d", i);
                return false;
            }
        }
        if (L.ssm_out.data && !L.ssm_out.on_device) {
            if (!upload_weight(L.ssm_out, L.ssm_out_qtype, L.wo_scales, compute_dtype,
                               stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload ssm_out for layer %d", i);
                return false;
            }
        }
        // SSM tensors that convert to compute_dtype (FP16): conv1d weights, norm
        for (Tensor* t : {&L.ssm_conv1d_w, &L.ssm_conv1d_b, &L.ssm_norm_w}) {
            if (t->data && !t->on_device) {
                if (!upload_unquantized_weight(*t, GGMLQuantType::NONE, compute_dtype,
                                               stream, gpu_allocations_)) {
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
                h2d_copy(d_data, t->data, bytes, stream);
                gpu_allocations_.push_back(d_data);
                t->data = d_data;
                t->on_device = true;
            }
        }

        // Router bias (Nemotron MoE)
        if (L.moe_router_bias.data && !L.moe_router_bias.on_device) {
            if (!upload_unquantized_weight(L.moe_router_bias, GGMLQuantType::NONE, compute_dtype,
                                           stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload moe_router_bias for layer %d", i);
                return false;
            }
        }

        // Shared expert weights (Nemotron/DeepSeek style)
        if (L.w_up_shared.data && !L.w_up_shared.on_device) {
            Tensor dummy_scales;
            if (!upload_weight(L.w_up_shared, L.w_up_shared_qtype, dummy_scales,
                               compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload w_up_shared for layer %d", i);
                return false;
            }
        }
        if (L.w_down_shared.data && !L.w_down_shared.on_device) {
            Tensor dummy_scales;
            if (!upload_weight(L.w_down_shared, L.w_down_shared_qtype, dummy_scales,
                               compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload w_down_shared for layer %d", i);
                return false;
            }
        }
        if (L.w_gate_shared.data && !L.w_gate_shared.on_device) {
            Tensor dummy_scales;
            if (!upload_weight(L.w_gate_shared, L.w_gate_shared_qtype, dummy_scales,
                               compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload w_gate_shared for layer %d", i);
                return false;
            }
        }

        IMP_LOG_DEBUG("Layer %d/%d non-expert weights uploaded", i + 1, n_layers());
    }

    // Sync Pass 1 before measuring free VRAM for expert budget
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }

    // =========================================================================
    // --- Pass 2: Expert weight upload ---
    // Now that all non-expert weights are on GPU, measure actual free VRAM
    // and greedily upload expert layers until the budget is exhausted.
    // =========================================================================

    // Compute per-layer expert weight costs
    size_t total_expert_bytes = 0;
    std::vector<size_t> layer_expert_bytes(n_layers(), 0);
    for (int i = 0; i < n_layers(); ++i) {
        const TransformerLayer& L = layers_[i];
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
    std::vector<bool> experts_upload_layer(n_layers(), false);
    if (total_expert_bytes > 0) {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        // Reserve for KV cache + SSM state + activation workspace + FP16 cache.
        // Engine passes the exact reserve based on computed workspace sizes.
        size_t budget = (free_mem > expert_reserve_bytes) ? (free_mem - expert_reserve_bytes) : 0;

        if (budget >= total_expert_bytes) {
            // All experts fit
            for (int i = 0; i < n_layers(); ++i) {
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
            for (int i = 0; i < n_layers(); ++i) {
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
    for (int i = 0; i < n_layers(); ++i) {
        TransformerLayer& L = layers_[i];

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
                    // Upload raw quantized bytes to GPU
                    void* gpu_ptr = nullptr;
                    cudaError_t err = cudaMalloc(&gpu_ptr, total_raw);
                    if (err == cudaSuccess) {
                        cudaError_t cpy_err = h2d_copy(gpu_ptr, packed.data, total_raw, stream);
                        if (cpy_err != cudaSuccess) {
                            IMP_LOG_ERROR("  %s: h2d_copy failed: %s", name, cudaGetErrorString(cpy_err));
                            cudaFree(gpu_ptr);
                            return false;
                        }
                        packed.data = gpu_ptr;
                        packed.on_device = true;
                        gpu_allocations_.push_back(gpu_ptr);
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
                        host_pinned_allocs_.push_back(pinned_buf);
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
                        host_pinned_.push_back(packed.data);
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
            if (!upload_weight(flat, qtype, dummy_scales, compute_dtype,
                               stream, gpu_allocations_)) {
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
                               compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload expert_w_gate[%zu] for layer %d", e, i);
                return false;
            }
        }
        for (size_t e = 0; e < L.expert_w_up.size(); ++e) {
            if (!L.expert_w_up[e].data || L.expert_w_up[e].on_device) continue;
            Tensor dummy_scales;
            if (!upload_weight(L.expert_w_up[e], L.expert_up_qtype, dummy_scales,
                               compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload expert_w_up[%zu] for layer %d", e, i);
                return false;
            }
        }
        for (size_t e = 0; e < L.expert_w_down.size(); ++e) {
            if (!L.expert_w_down[e].data || L.expert_w_down[e].on_device) continue;
            Tensor dummy_scales;
            if (!upload_weight(L.expert_w_down[e], L.expert_down_qtype, dummy_scales,
                               compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload expert_w_down[%zu] for layer %d", e, i);
                return false;
            }
        }
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

} // namespace imp
