#pragma once
// Gemma-4 E-variant per-layer inputs (PLE, HF Gemma4TextModel.project_per_layer_inputs and the
// decoder-layer per_layer_input branch). FP16 storage, FP32 math.

#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// In place over proj [n, n_layers * dim] (already * hidden_size^-0.5 by the GEMM alpha):
//   proj[t, l, :] = (rmsnorm(proj[t, l, :]) * norm_w + table[id_t, l, :] * tok_scale) * out_scale
// with id_t = ids[t], replaced by pad_id when it equals mm_token_id or is outside [0, vocab)
// (HF swaps multimodal placeholders for the pad token). table is FP16 [vocab, n_layers * dim]
// and may be mapped pinned host memory. dim <= 1024.
void pli_project_combine(void* proj, const void* table, const int32_t* ids, const void* norm_w, int n,
                         int n_layers, int dim, int vocab, int32_t mm_token_id, int32_t pad_id, float eps,
                         float tok_scale, float out_scale, cudaStream_t stream);

// In place over gate [n, dim]: gate = gelu_tanh(gate) * pli[t, layer * dim + j], pli row stride
// n_layers * dim (HF act_fn(per_layer_input_gate(h)) * per_layer_input).
void pli_gate_mul(void* gate, const void* pli, int n, int n_layers, int dim, int layer, cudaStream_t stream);

}  // namespace imp
