#include "graph/moe_workspace.h"
#include <cuda_runtime.h>

namespace imp {

void MoEWorkspace::free(VRAMAllocator* alloc) {
    auto vfree = [alloc](void*& p) {
        if (!p) return;
        if (alloc) alloc->free(p);
        else cudaFree(p);
        p = nullptr;
    };

    routing_buffers.free();

    vfree(dequant_buf);
    dequant_buf_size = 0;

    vfree(batch_dequant_buf);
    batch_dequant_buf_size = 0;

    if (d_work_ptrs) {
        cudaFree(d_work_ptrs);
        d_work_ptrs = nullptr;
        d_work_ptrs_count = 0;
    }

    if (d_fp8_scales) {
        cudaFree(d_fp8_scales);
        d_fp8_scales = nullptr;
    }

    if (d_weight_ptrs) {
        cudaFree(d_weight_ptrs);
        d_weight_ptrs = nullptr;
        d_weight_ptrs_count = 0;
    }

    vfree(raw_staging_buf);
    raw_staging_size = 0;
}

} // namespace imp
