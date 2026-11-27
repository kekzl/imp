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
// Arguments in the old parameter order (audit D-11); all required.
struct PliProjectCombineArgs {
    void* proj = nullptr;
    const void* table = nullptr;
    const int32_t* ids = nullptr;
    const void* norm_w = nullptr;
    int n = 0;
    int n_layers = 0;
    int dim = 0;
    int vocab = 0;
    int32_t mm_token_id = 0;
    int32_t pad_id = 0;
    float eps = 0.0f;
    float tok_scale = 0.0f;
    float out_scale = 0.0f;
    cudaStream_t stream = nullptr;
};
void pli_project_combine(const PliProjectCombineArgs& a);

// In place over gate [n, dim]: gate = gelu_tanh(gate) * pli[t, layer * dim + j], pli row stride
// n_layers * dim (HF act_fn(per_layer_input_gate(h)) * per_layer_input).
void pli_gate_mul(void* gate, const void* pli, int n, int n_layers, int dim, int layer, cudaStream_t stream);

}  // namespace imp
