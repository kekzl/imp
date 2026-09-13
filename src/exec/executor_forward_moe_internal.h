#pragma once

#include "core/tensor.h"
#include "compute/activation.h"
#include "compute/ssm.h"
#include "quant/dequant_gpu.h"
#include "exec/expert_cache.h"
#include "exec/moe_workspace.h"
#include "exec/nvfp4_expert_offload.h"
#include "model/model_config.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace imp {

// Several MoE prefill paths read routing metadata on the host (D2H+sync)
// to size per-expert GEMMs. Under an active stream capture that sync fails
// SILENTLY, the host reads uninitialized offsets, and the recorded graph
// launches expert GEMMs with garbage geometry -> misaligned address at
// launch (#855 class, root-caused on Nemotron-H NVFP4 in #847). Only the
// CUTLASS 3.x device-args path records cleanly; every host-args path must fail the capture loudly (#858).
inline void moe_host_args_capture_guard(cudaStream_t stream) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &st) == cudaSuccess && st != cudaStreamCaptureStatusNone)
        throw std::runtime_error(
            "MoE host-args prefill path reads routing on the host — not graph-capturable");
}

__global__ void sanitize_fp16_kernel(__half* __restrict__ data, int64_t n);
__global__ void moe_apply_per_expert_scale_kernel(
    float* __restrict__ weights, const int32_t* __restrict__ indices,
    const __half* __restrict__ scales, int n_weights);

inline void sanitize_fp16(__half* data, int64_t n, cudaStream_t stream) {
    if (n <= 0)
        return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    sanitize_fp16_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

inline void apply_expert_activation(void* gate_data, void* up_data, void* swiglu_data, bool non_gated,
                                    int64_t rows, int64_t eff, QType compute_dtype, FFNActivation act_type,
                                    cudaStream_t stream) {
    int64_t act_shape[2] = {rows, eff};
    if (non_gated) {
        Tensor up_t(up_data, compute_dtype, 2, act_shape, true);
        relu_sqr_inplace(up_t, stream);
    } else {
        Tensor g(gate_data, compute_dtype, 2, act_shape, true);
        Tensor u(up_data, compute_dtype, 2, act_shape, true);
        Tensor a(swiglu_data, compute_dtype, 2, act_shape, true);
        if (act_type == FFNActivation::GEGLU)
            geglu(g, u, a, stream);
        else if (act_type == FFNActivation::GPT_OSS_GLU)
            gpt_oss_glu(g, u, a, stream);
        else
            swiglu(g, u, a, stream);
    }
}

inline size_t expert_stride(const Tensor& packed, QType qtype) {
    int64_t rows = packed.shape[1];
    int64_t cols = packed.shape[2];
    return static_cast<size_t>(rows) * qtype_row_bytes(qtype, cols);
}

// Can the fused MoE decode kernels address host-resident experts through
// the LRU cache's per-layer slot pool (base+idx*stride, idx=slot index)?
// Lives here because the DISPATCH predicate and run_moe_decode_fast must
// agree exactly, else a host pointer reaches a kernel. One definition, two
// call sites. The whole working set must fit the layer's pool (#1365), or one projection's loads evict
// another's.
inline bool host_expert_pool_ready(const Tensor& up_packed, const ExpertLRUCache& cache,
                                   const MoEWorkspace& moe, int top_k) {
    return (!up_packed.on_device && cache.n_slots_ > 0 && cache.pool_ != nullptr && cache.slot_size_ > 0 &&
            moe.d_slot_idx != nullptr && moe.d_slot_idx_count >= kExpertProjCount * top_k &&
            cache.slots_per_layer_ >= kExpertProjCount * top_k);
}

// Weaker form for the LEGACY (prefill) path: stages one expert at a time
// and consumes it before the next, so it needs neither the slot-index
// buffer nor 3*top_k slots (stream ordering already keeps a refill behind
// its reader). Just needs a pool that carries NVFP4 slots at all.
inline bool nvfp4_host_pool_ready_for_staging(const ExpertLRUCache& cache) {
    return (cache.nvfp4_slots_ && cache.d_slot_scales_ != nullptr && cache.pool_ != nullptr &&
            cache.slot_size_ > 0 && cache.slots_per_layer_ >= kExpertProjCount);
}

// Same question for NVFP4 experts, two extra requirements over GGUF: the
// pool must be initialised with NVFP4 slots (wider: packed weights AND
// micro-scales), and the per-slot tensor-scale mirror must exist (fused kernels index it by slot number).
inline bool nvfp4_host_pool_ready(const ExpertLRUCache& cache, const MoEWorkspace& moe, int top_k) {
    return (cache.nvfp4_slots_ && cache.d_slot_scales_ != nullptr && cache.n_slots_ > 0 &&
            cache.pool_ != nullptr && cache.slot_size_ > 0 && moe.d_slot_idx != nullptr &&
            moe.d_slot_idx_count >= kExpertProjCount * top_k &&
            cache.slots_per_layer_ >= kExpertProjCount * top_k);
}

// Does this layer's n==1 decode go to the host-resident NVFP4 slot path?
// ONE definition, three readers (can_decode_fast, the residual-fusion
// pre-check, run_moe_decode_fast) that must agree exactly. Deliberately
// does NOT look at expert_up_packed: that tensor is only stamped for
// device-resident experts, so checking it kept host-resident NVFP4 decode stuck in the slow serial path.
inline bool nvfp4_host_decode_ready(const TransformerLayer& ly, const ExpertLRUCache& cache,
                                    const MoEWorkspace& moe, int top_k) {
    if (!nvfp4_host_pool_ready(cache, moe, top_k))
        return false;
    const bool non_gated = (ly.expert_gate_packed.data == nullptr &&
                            (ly.expert_w_gate.empty() || ly.expert_w_gate[0].data == nullptr));
    return nvfp4_host_experts_servable(ly.expert_w_up) &&
           nvfp4_host_experts_servable(ly.expert_w_down) &&
           (non_gated || nvfp4_host_experts_servable(ly.expert_w_gate));
}

}  // namespace imp
