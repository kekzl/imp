// Phase 4c (gemm.q8_drop_source): a dense Q8_0 weight with an NVFP4 decode overlay is read at
// M <= 32 through the overlay and at M >= 2 through its IMMA planes; the Q8_0 source only feeds
// the plane build and an IMMA decline (gemm_via_handle_ rebuilds FP16 from the planes then).
// Build the planes now and free the source: 1.0625 B/elem per weight.

#include "compute/mmq_q8_imma.h"
#include "core/logging.h"
#include "exec/quant_pipeline.h"
#include "exec/pre_dequant_internal.h"
#include "exec/quant_pipeline.h"
#include "exec/pre_dequant_internal.h"
#include "memory/mem_account.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <initializer_list>

namespace imp {

void QuantPipeline::pre_dequant_phase4c_q8_planes_only_(const ModelConfig& cfg, cudaStream_t stream) {
    const auto& g = dispatch_policy().gemm;
    if (!g.q8_imma_enabled || !g.q8_drop_source || wcache_->nvfp4_decode_mode <= 0)
        return;
    auto* m = const_cast<Model*>(model_);
    int freed = 0;
    const size_t pool_before = release_on_free_pool_reserved(ReleasePool::Q8Sources);
    // A group (q/k/v, gate/up) goes only when every member has an overlay: with one missing, the
    // fused n == 1 dp4a path reads the source of every member (dp4a_qkv_route_, fused_ffn_norm).
    auto release_group = [&](std::initializer_list<Tensor*> group) {
        for (const Tensor* t : group)
            if (t->data && !wcache_->nvfp4.contains(t->data))
                return;
        for (Tensor* t : group) {
            if (!t->data || t->dropped_source || t->qtype != QType::Q8_0 || t->ndim != 2 ||
                !m->is_base_gpu_allocation(t->data))
                continue;
            if (!mmq_q8_imma_build_planes(t->data, static_cast<int>(t->shape[0]),
                                          static_cast<int>(t->shape[1]), stream))
                continue;  // shape or plane budget: the source stays the prefill route
            wcache_->q8_source_freed_bytes += qtype_row_bytes(QType::Q8_0, t->shape[1]) *
                                              static_cast<size_t>(t->shape[0]);
            wcache_->q8_planes_only.insert(t->data);
            free_source_(*t, stream);
            ++freed;
        }
    };
    for (int i = 0; i < cfg.n_layers; ++i) {
        auto& L = m->layer(i);
        release_group({&L.wq, &L.wk, &L.wv});
        release_group({&L.wo});
        release_group({&L.w_gate, &L.w_up});
        release_group({&L.w_down});
        release_group({&L.w_gate_shared, &L.w_up_shared});
        release_group({&L.w_down_shared});
    }
    if (freed == 0)
        return;
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));  // the threshold-0 pool releases here
    const size_t pool_after = release_on_free_pool_reserved(ReleasePool::Q8Sources);
    const size_t back = pool_before - std::min(pool_before, pool_after) + trim_device_mempool();
    IMP_LOG_INFO(
        "Phase-4c: freed %d Q8_0 sources (%.1f MiB, %.1f MiB back to the driver); their IMMA "
        "planes serve M >= 2, the NVFP4 overlay M <= 32",
        freed, wcache_->q8_source_freed_bytes / (1024.0 * 1024.0), back / (1024.0 * 1024.0));
}

}  // namespace imp
