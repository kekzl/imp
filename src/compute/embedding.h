#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

void embedding_lookup(const Tensor& table, const int32_t* token_ids, int n_tokens, Tensor& out,
                      cudaStream_t stream = nullptr);

// Overload for raw quantized embedding tables (Q8_0/Q6_K).
// Dequantizes only the needed rows on the fly.
void embedding_lookup(const Tensor& table, const int32_t* token_ids, int n_tokens, Tensor& out, QType qtype,
                      cudaStream_t stream);

// Device-side embedding lookup: reads token ID from device memory (d_token_id[0]).
// Used for async sampling where the token ID stays on GPU between decode steps.
void embedding_lookup_from_device(const Tensor& table, const int32_t* d_token_id, Tensor& out,
                                  cudaStream_t stream = nullptr);

// Unpacked MXFP4 table (unpack_mxfp4_gguf): E2M1 [rows, d/2] + UE8M0 linear scales [rows, d/32].
void embedding_lookup_mxfp4(const void* packed, const void* linear_scales, const int32_t* token_ids,
                            int n_tokens, Tensor& out, cudaStream_t stream);

void embedding_lookup_from_device(const Tensor& table, const int32_t* d_token_id, Tensor& out, QType qtype,
                                  cudaStream_t stream);

}  // namespace imp
