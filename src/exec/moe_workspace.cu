#include "exec/moe_workspace.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

// The batched-MoE pointer/scale arrays are T2 arena tenants since A7 step 4b.2:
// engine-lifetime, sized from n_experts, charged by exec_t2_demand as
// `moe_arrays`. The arena is closed by ~Engine after every executor teardown, so
// the frees that used to live here are gone — only the pointer nulling remains.

void MoEWorkspace::free(VRAMAllocator* alloc) {
    auto vfree = [alloc](void*& p) {
        if (!p)
            return;
        if (alloc)
            alloc->free(p);
        else
            IMP_CUDA_CHECK_LOG(cudaFree(p));
        p = nullptr;
    };

    routing_buffers.free();

    vfree(dequant_buf);
    dequant_buf_size = 0;

    vfree(batch_dequant_buf);
    batch_dequant_buf_size = 0;

    vfree(fp32_down_buf);
    fp32_down_buf_size = 0;

    if (d_work_ptrs) {
        d_work_ptrs = nullptr;
        d_work_ptrs_count = 0;
    }

    if (d_fp8_scales) {
        d_fp8_scales = nullptr;
    }

    if (d_M_per) {
        d_M_per = nullptr;
        d_M_per_count = 0;
    }

    if (d_alpha_compact) {
        d_alpha_compact = nullptr;
    }

    if (d_na) {
        d_na = nullptr;
    }

    if (d_sfa_offsets) {
        d_sfa_offsets = nullptr;
    }

    if (d_B_ptrs_cache) {
        d_B_ptrs_cache = nullptr;
    }
    if (d_SFB_ptrs_cache) {
        d_SFB_ptrs_cache = nullptr;
    }
    if (d_alpha_full) {
        d_alpha_full = nullptr;
    }

    // Phase 3c-full Step 3 per-layer caches.
    for (auto& c : per_layer_da_cache) {
        auto cfree = [](void* p) { if (p) IMP_CUDA_CHECK_LOG(cudaFree(p)); };
        cfree(c.d_gate_B_ptrs);   cfree(c.d_gate_SFB_ptrs);   cfree(c.d_gate_alpha);
        cfree(c.d_up_B_ptrs);     cfree(c.d_up_SFB_ptrs);     cfree(c.d_up_alpha);
        cfree(c.d_down_B_ptrs);   cfree(c.d_down_SFB_ptrs);   cfree(c.d_down_alpha);
        c = {};
    }
    per_layer_da_cache.clear();

    if (d_weight_ptrs) {
        d_weight_ptrs = nullptr;
        d_weight_ptrs_count = 0;
    }

    vfree(raw_staging_buf);
    raw_staging_size = 0;

    vfree(cutlass3x_packed);
    cutlass3x_packed_size = 0;
    vfree(cutlass3x_sf);
    cutlass3x_sf_size = 0;
    if (cutlass3x_sfa_ptrs) {
        IMP_CUDA_CHECK_LOG(cudaFree(cutlass3x_sfa_ptrs));
        cutlass3x_sfa_ptrs = nullptr;
        cutlass3x_sfa_ptrs_count = 0;
    }
}

}  // namespace imp
