#include "graph/moe_workspace.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

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
        IMP_CUDA_CHECK_LOG(cudaFree(d_work_ptrs));
        d_work_ptrs = nullptr;
        d_work_ptrs_count = 0;
    }

    if (d_fp8_scales) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_fp8_scales));
        d_fp8_scales = nullptr;
    }

    if (d_M_per) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_M_per));
        d_M_per = nullptr;
        d_M_per_count = 0;
    }

    if (d_alpha_compact) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_alpha_compact));
        d_alpha_compact = nullptr;
    }

    if (d_na) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_na));
        d_na = nullptr;
    }

    if (d_sfa_offsets) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_sfa_offsets));
        d_sfa_offsets = nullptr;
    }

    if (d_B_ptrs_cache) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_B_ptrs_cache));
        d_B_ptrs_cache = nullptr;
    }
    if (d_SFB_ptrs_cache) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_SFB_ptrs_cache));
        d_SFB_ptrs_cache = nullptr;
    }
    if (d_alpha_full) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_alpha_full));
        d_alpha_full = nullptr;
    }

    if (d_weight_ptrs) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_weight_ptrs));
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
