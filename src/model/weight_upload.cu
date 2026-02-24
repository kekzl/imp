#include "model/model.h"
#include "model/gguf_loader.h"
#include "quant/dequant_gpu.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstring>
#include <cmath>

namespace imp {

// ---------------------------------------------------------------------------
// Checked GPU allocation: prevents CUDA memory oversubscription by verifying
// enough free GPU memory exists before allocating.  Without this, cudaMalloc
// on Linux silently succeeds by backing with system RAM (unified memory),
// which causes cuBLASLt INTERNAL_ERROR on the resulting pointers.
// ---------------------------------------------------------------------------
static constexpr size_t kWeightReserveMiB = 256;  // reserve for KV cache, SSM state, misc

static cudaError_t checked_cuda_malloc(void** ptr, size_t size) {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t reserve = kWeightReserveMiB << 20;
    if (size + reserve > free_mem) {
        *ptr = nullptr;
        return cudaErrorMemoryAllocation;
    }
    return cudaMalloc(ptr, size);
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
        cudaMemcpyAsync(d_nibbles, h_nibbles.data(), nibbles_bytes,
                        cudaMemcpyHostToDevice, stream);
        gpu_allocs.push_back(d_nibbles);

        // Upload scales to GPU
        void* d_scales = nullptr;
        size_t scales_bytes = scales_count * sizeof(uint16_t);
        checked_cuda_malloc(&d_scales, scales_bytes);
        if (!d_scales) { cudaFree(d_nibbles); return false; }
        cudaMemcpyAsync(d_scales, h_scales.data(), scales_bytes,
                        cudaMemcpyHostToDevice, stream);
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
            cudaMemcpyAsync(d_data, weight.data, raw_bytes,
                            cudaMemcpyHostToDevice, stream);
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
        cudaMemcpyAsync(d_data, h_fp16.data(), bytes,
                        cudaMemcpyHostToDevice, stream);
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
            cudaMemcpyAsync(d_data, weight.data, raw_bytes,
                            cudaMemcpyHostToDevice, stream);
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
        cudaMemcpyAsync(d_data, h_fp16.data(), bytes,
                        cudaMemcpyHostToDevice, stream);
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
            cudaError_t cpy_err = cudaMemcpyAsync(d_data, weight.data, raw_bytes,
                            cudaMemcpyHostToDevice, stream);
            if (cpy_err != cudaSuccess) {
                IMP_LOG_ERROR("cudaMemcpyAsync failed for qtype=%u [%ldx%ld] %zu bytes: %s",
                              (unsigned)qtype, (long)N, (long)K, raw_bytes,
                              cudaGetErrorString(cpy_err));
            }
            gpu_allocs.push_back(d_data);
            IMP_LOG_DEBUG("Upload raw qtype=%u [%ldx%ld] %zu bytes -> GPU %p",
                          (unsigned)qtype, (long)N, (long)K, raw_bytes, d_data);
            weight.data = d_data;
            weight.on_device = true;
            return true;
        } else {
            // Dequant on GPU: upload raw → dequant to FP16 → free raw
            size_t raw_bytes = static_cast<size_t>(N) * ggml_quant_row_bytes(qtype, K);
            void* d_raw = nullptr;
            checked_cuda_malloc(&d_raw, raw_bytes);
            if (!d_raw) return false;
            cudaMemcpyAsync(d_raw, weight.data, raw_bytes,
                            cudaMemcpyHostToDevice, stream);

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
        cudaMemcpyAsync(d_data, weight.data, bytes,
                        cudaMemcpyHostToDevice, stream);
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
            cudaMemcpyAsync(d_data, weight.data, bytes,
                            cudaMemcpyHostToDevice, stream);
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
        cudaMemcpyAsync(d_data, h_fp16.data(), bytes,
                        cudaMemcpyHostToDevice, stream);
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
// Model::upload_weights_gpu
// ---------------------------------------------------------------------------

bool Model::upload_weights_gpu(DType compute_dtype, cudaStream_t stream) {
    if (gpu_weights_ready_) {
        IMP_LOG_WARN("Weights already uploaded to GPU");
        return true;
    }

    IMP_LOG_INFO("Uploading model weights to GPU (%d layers)...", n_layers());

    // Upload token embedding
    // Embedding lookup only supports Q8_0/Q6_K natively; other quant types
    // need to be dequanted to FP16 (raw_quant=false) so the standard FP16
    // embedding gather works.
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

    // Upload output projection — keep as CPU-dequanted FP16 (too large for
    // on-the-fly dequant scratch buffer)
    if (out_proj_.data && !out_proj_.on_device) {
        if (!upload_unquantized_weight(out_proj_, out_proj_qtype_, compute_dtype,
                                       stream, gpu_allocations_,
                                       /*raw_quant=*/false)) {
            IMP_LOG_ERROR("Failed to upload output projection");
            return false;
        }
    }

    // Upload per-layer weights
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

            // Path A1: Quantized types -- keep on host (mmap'd), pin for fast H2D.
            // Expert weights are offloaded: only the active expert's raw bytes are
            // copied to a small GPU staging buffer during the forward pass.
            // This saves ~31 GiB VRAM for models like Nemotron-H with 128 experts.
            if (dequant_gpu_supported(qtype)) {
                size_t row_bytes = ggml_quant_row_bytes(qtype, cols);
                size_t expert_raw = static_cast<size_t>(rows) * row_bytes;
                size_t total_raw = static_cast<size_t>(n_experts) * expert_raw;

                // Pin host memory for faster H2D transfers during inference.
                // cudaHostRegister can fail (e.g., mmap not pinnable), which is fine —
                // cudaMemcpy still works from unpinned host memory, just slower.
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

                // packed.data stays as host mmap pointer
                // packed.on_device stays false
                // No gpu_allocations_.push_back — nothing allocated on GPU
                // expert_vec stays empty -- executor uses packed + on-the-fly dequant
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

            // flat.data now points to GPU FP16 [n_experts*rows, cols]
            // Slice into per-expert views
            expert_vec.resize(n_experts);
            size_t expert_bytes = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
            for (int e = 0; e < n_experts; e++) {
                char* ptr = static_cast<char*>(flat.data) + e * expert_bytes;
                int64_t eshape[4] = {rows, cols, 0, 0};
                expert_vec[e] = Tensor(ptr, DType::FP16, 2, eshape, true);
            }
            packed = Tensor();  // clear packed (data owned by GPU allocation)
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
            if (!upload_weight(L.expert_w_gate[e], L.w_gate_qtype, dummy_scales,
                               compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload expert_w_gate[%zu] for layer %d",
                              e, i);
                return false;
            }
        }
        for (size_t e = 0; e < L.expert_w_up.size(); ++e) {
            if (!L.expert_w_up[e].data || L.expert_w_up[e].on_device) continue;
            Tensor dummy_scales;
            if (!upload_weight(L.expert_w_up[e], L.w_up_qtype, dummy_scales,
                               compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload expert_w_up[%zu] for layer %d",
                              e, i);
                return false;
            }
        }
        for (size_t e = 0; e < L.expert_w_down.size(); ++e) {
            if (!L.expert_w_down[e].data || L.expert_w_down[e].on_device) continue;
            Tensor dummy_scales;
            if (!upload_weight(L.expert_w_down[e], L.w_down_qtype, dummy_scales,
                               compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload expert_w_down[%zu] for layer %d",
                              e, i);
                return false;
            }
        }

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
                cudaMemcpyAsync(d_data, t->data, bytes,
                                cudaMemcpyHostToDevice, stream);
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

        IMP_LOG_DEBUG("Layer %d/%d uploaded", i + 1, n_layers());
    }

    // Synchronize to ensure all async copies are complete
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
