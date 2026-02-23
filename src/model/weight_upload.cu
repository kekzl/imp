#include "model/model.h"
#include "model/gguf_loader.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstring>
#include <cmath>

namespace imp {

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

    // Normal
    uint16_t h_exp = static_cast<uint16_t>(f_exp + 15);
    uint16_t h_man = static_cast<uint16_t>(f_man >> 13);
    return static_cast<uint16_t>((f_sign << 15) | (h_exp << 10) | h_man);
}

// ---------------------------------------------------------------------------
// upload_weight: upload a single weight tensor from host (mmap) to GPU.
//
// For Q4_0: splits into packed_nibbles [N, K/2] + scales [N, K/32] on GPU.
//           Updates weight tensor to point to packed_nibbles (dtype=INT4),
//           fills scales_out tensor.
// For Q8_0: dequants to FP16 on host, uploads as FP16. scales_out stays empty.
// For Q6_K: dequants to FP16 on host, uploads as FP16. scales_out stays empty.
// For F16/BF16: direct upload. scales_out stays empty.
// For F32: converts to FP16 on host, uploads. scales_out stays empty.
// ---------------------------------------------------------------------------

static bool upload_weight(Tensor& weight, GGMLQuantType qtype,
                          Tensor& scales_out,
                          DType compute_dtype,
                          cudaStream_t stream,
                          std::vector<void*>& gpu_allocs) {
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
        cudaMalloc(&d_nibbles, nibbles_bytes);
        cudaMemcpyAsync(d_nibbles, h_nibbles.data(), nibbles_bytes,
                        cudaMemcpyHostToDevice, stream);
        gpu_allocs.push_back(d_nibbles);

        // Upload scales to GPU
        void* d_scales = nullptr;
        size_t scales_bytes = scales_count * sizeof(uint16_t);
        cudaMalloc(&d_scales, scales_bytes);
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
        int blocks_per_row = static_cast<int>(K) / 32;

        static constexpr size_t Q8_0_BLOCK_SIZE = 34; // 2 (fp16 scale) + 32 (int8 quants)

        size_t fp16_count = static_cast<size_t>(N * K);
        std::vector<uint16_t> h_fp16(fp16_count);

        const uint8_t* raw = static_cast<const uint8_t*>(weight.data);

        for (int64_t n = 0; n < N; ++n) {
            for (int b = 0; b < blocks_per_row; ++b) {
                const uint8_t* block_ptr = raw + (n * blocks_per_row + b) * Q8_0_BLOCK_SIZE;

                // Scale: first 2 bytes (fp16) -- convert to float on host
                uint16_t scale_bits;
                std::memcpy(&scale_bits, block_ptr, 2);
                float scale_f = fp16_to_float(scale_bits);

                // Quants: next 32 bytes (int8)
                const int8_t* quants = reinterpret_cast<const int8_t*>(block_ptr + 2);
                for (int q = 0; q < 32; ++q) {
                    float val = static_cast<float>(quants[q]) * scale_f;
                    h_fp16[n * K + b * 32 + q] = float_to_fp16(val);
                }
            }
        }

        // Upload FP16 data to GPU
        size_t bytes = fp16_count * sizeof(uint16_t);
        void* d_data = nullptr;
        cudaMalloc(&d_data, bytes);
        cudaMemcpyAsync(d_data, h_fp16.data(), bytes,
                        cudaMemcpyHostToDevice, stream);
        gpu_allocs.push_back(d_data);

        // Update weight to FP16 on device — shape [N, K] (out_features, in_features)
        int64_t new_shape[4] = {N, K, 0, 0};
        weight = Tensor(d_data, DType::FP16, 2, new_shape, true);
        // scales_out stays empty (fully dequanted)

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
        int blocks_per_row = static_cast<int>(K) / 256;

        // Q6_K block: ql[128] + qh[64] + scales[16] + d(fp16) = 210 bytes
        static constexpr size_t Q6_K_BLOCK_SIZE = 210;

        size_t fp16_count = static_cast<size_t>(N * K);
        std::vector<uint16_t> h_fp16(fp16_count);

        const uint8_t* raw = static_cast<const uint8_t*>(weight.data);

        for (int64_t n = 0; n < N; ++n) {
            for (int b = 0; b < blocks_per_row; ++b) {
                const uint8_t* block_ptr = raw + (n * blocks_per_row + b) * Q6_K_BLOCK_SIZE;

                const uint8_t* ql     = block_ptr;          // 128 bytes
                const uint8_t* qh     = block_ptr + 128;    // 64 bytes
                const int8_t*  scales  = reinterpret_cast<const int8_t*>(block_ptr + 192); // 16 bytes
                uint16_t d_bits;
                std::memcpy(&d_bits, block_ptr + 208, 2);   // last 2 bytes
                float d = fp16_to_float(d_bits);

                for (int i = 0; i < 256; ++i) {
                    uint8_t low4  = (ql[i / 2] >> ((i % 2) * 4)) & 0xF;
                    uint8_t high2 = (qh[i / 4] >> ((i % 4) * 2)) & 0x3;
                    int q6 = static_cast<int>((high2 << 4) | low4) - 32;
                    float val = d * static_cast<float>(scales[i / 16]) * static_cast<float>(q6);
                    h_fp16[n * K + b * 256 + i] = float_to_fp16(val);
                }
            }
        }

        size_t bytes = fp16_count * sizeof(uint16_t);
        void* d_data = nullptr;
        cudaMalloc(&d_data, bytes);
        cudaMemcpyAsync(d_data, h_fp16.data(), bytes,
                        cudaMemcpyHostToDevice, stream);
        gpu_allocs.push_back(d_data);

        // Shape [N, K] (out_features, in_features)
        int64_t new_shape[4] = {N, K, 0, 0};
        weight = Tensor(d_data, DType::FP16, 2, new_shape, true);

        return true;
    }

    // ---- F16 / BF16: direct upload ----
    if (qtype == GGMLQuantType::F16 || qtype == GGMLQuantType::BF16) {
        size_t bytes = weight.nbytes();
        void* d_data = nullptr;
        cudaMalloc(&d_data, bytes);
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
            cudaMalloc(&d_data, bytes);
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
        cudaMalloc(&d_data, bytes);
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
                                      std::vector<void*>& gpu_allocs) {
    Tensor dummy_scales;
    return upload_weight(weight, qtype, dummy_scales, compute_dtype,
                         stream, gpu_allocs);
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
    if (tok_emb_.data && !tok_emb_.on_device) {
        if (!upload_unquantized_weight(tok_emb_, tok_emb_qtype_, compute_dtype,
                                       stream, gpu_allocations_)) {
            IMP_LOG_ERROR("Failed to upload token embedding");
            return false;
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

    // Upload output projection
    if (out_proj_.data && !out_proj_.on_device) {
        if (!upload_unquantized_weight(out_proj_, out_proj_qtype_, compute_dtype,
                                       stream, gpu_allocations_)) {
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

        // MoE expert weights
        for (size_t e = 0; e < L.expert_w_gate.size(); ++e) {
            Tensor dummy_scales;
            // Experts use the same qtype as the dense FFN counterpart
            if (!upload_weight(L.expert_w_gate[e], L.w_gate_qtype, dummy_scales,
                               compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload expert_w_gate[%zu] for layer %d",
                              e, i);
                return false;
            }
        }
        for (size_t e = 0; e < L.expert_w_up.size(); ++e) {
            Tensor dummy_scales;
            if (!upload_weight(L.expert_w_up[e], L.w_up_qtype, dummy_scales,
                               compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload expert_w_up[%zu] for layer %d",
                              e, i);
                return false;
            }
        }
        for (size_t e = 0; e < L.expert_w_down.size(); ++e) {
            Tensor dummy_scales;
            if (!upload_weight(L.expert_w_down[e], L.w_down_qtype, dummy_scales,
                               compute_dtype, stream, gpu_allocations_)) {
                IMP_LOG_ERROR("Failed to upload expert_w_down[%zu] for layer %d",
                              e, i);
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
