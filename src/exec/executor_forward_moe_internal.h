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

// Several MoE prefill paths read routing metadata on the host (D2H + stream
// sync) to size the per-expert GEMMs. Under an active stream capture that
// sync fails SILENTLY (the error was unchecked) and the host reads
// uninitialized offsets — the recorded graph then launches expert GEMMs with
// garbage geometry and dies with `misaligned address` at graph launch (the
// #855 census crash class; root-caused on Nemotron-H NVFP4 in #847). Only
// the CUTLASS 3.x device-args path records cleanly; every host-args path
// must fail the capture loudly instead (same lesson as #858).
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

// Can the fused MoE decode kernels address host-resident experts through the
// LRU cache's per-layer slot pool? They read `base + idx * stride`, and the
// pool is exactly that (fixed `slot_size_` stride), so `idx` becomes a slot
// index. Lives here because the DISPATCH predicate and run_moe_decode_fast
// must agree exactly: if the dispatch says yes and the function then declines,
// a host pointer reaches a kernel. One definition, two call sites.
//
// The whole working set must fit the layer's pool, or one projection's loads
// evict another's — the same threshold the cache gate in #1365 enforces.
inline bool host_expert_pool_ready(const Tensor& up_packed, const ExpertLRUCache& cache,
                                   const MoEWorkspace& moe, int top_k) {
    return (!up_packed.on_device && cache.n_slots_ > 0 && cache.pool_ != nullptr && cache.slot_size_ > 0 &&
            moe.d_slot_idx != nullptr && moe.d_slot_idx_count >= kExpertProjCount * top_k &&
            cache.slots_per_layer_ >= kExpertProjCount * top_k);
}

// The weaker form the LEGACY (prefill) path needs. It stages one expert at a
// time and consumes the result before touching the next, so it needs neither
// the slot-index buffer nor `3 * top_k` slots — stream ordering already keeps
// a refill behind the kernel that read the previous occupant. What it does
// need is a pool that carries NVFP4 slots at all.
inline bool nvfp4_host_pool_ready_for_staging(const ExpertLRUCache& cache) {
    return (cache.nvfp4_slots_ && cache.d_slot_scales_ != nullptr && cache.pool_ != nullptr &&
            cache.slot_size_ > 0 && cache.slots_per_layer_ >= kExpertProjCount);
}

// Same question for NVFP4 experts. Two extra requirements over the GGUF case:
// the pool must have been initialised with NVFP4 slots (they are wider — one
// slot holds packed weights AND micro-scales), and the per-slot tensor-scale
// mirror must exist, because the fused kernels index it with the slot number.
inline bool nvfp4_host_pool_ready(const ExpertLRUCache& cache, const MoEWorkspace& moe, int top_k) {
    return (cache.nvfp4_slots_ && cache.d_slot_scales_ != nullptr && cache.n_slots_ > 0 &&
            cache.pool_ != nullptr && cache.slot_size_ > 0 && moe.d_slot_idx != nullptr &&
            moe.d_slot_idx_count >= kExpertProjCount * top_k &&
            cache.slots_per_layer_ >= kExpertProjCount * top_k);
}

// Does this layer's n==1 decode go to the host-resident NVFP4 slot path?
//
// ONE definition, three readers: `can_decode_fast` (may I take the decode fast
// path at all), the residual-fusion pre-check, and run_moe_decode_fast's own
// branch. They must agree exactly — if the first says yes and the last
// declines, the layer falls through to a path that cannot serve it.
//
// This deliberately does NOT look at `expert_up_packed`: that tensor is only
// stamped for device-resident experts, so a host-resident NVFP4 layer arrives
// with it empty. Reading it there is what kept this decode in the serial
// legacy path at 35 tok/s while the slot path sat unused.
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
