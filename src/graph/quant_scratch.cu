#include "graph/quant_scratch.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

void QuantScratch::free(VRAMAllocator* alloc) {
    auto vfree = [alloc](void*& p) {
        if (!p) return;
        if (alloc) alloc->free(p);
        else IMP_CUDA_CHECK_LOG(cudaFree(p));
        p = nullptr;
    };

    vfree(dequant);
    dequant_size = 0;

    vfree(fp8_act);
    fp8_act_size = 0;
    if (d_act_scale) { IMP_CUDA_CHECK_LOG(cudaFree(d_act_scale)); d_act_scale = nullptr; }
    if (d_fp8_block_maxes) { IMP_CUDA_CHECK_LOG(cudaFree(d_fp8_block_maxes)); d_fp8_block_maxes = nullptr; }
    if (d_fp8_absmax) { IMP_CUDA_CHECK_LOG(cudaFree(d_fp8_absmax)); d_fp8_absmax = nullptr; }
    fp8_max_grid = 0;

    vfree(cutlass_act_data);
    cutlass_act_data_size = 0;
    vfree(cutlass_act_sf);
    cutlass_act_sf_size = 0;
    vfree(cutlass_workspace);
    cutlass_workspace_size = 0;

    vfree(mxfp4_act_sf);
    mxfp4_act_sf_size = 0;
    vfree(mxfp4_workspace);
    mxfp4_workspace_size = 0;

    if (q8_1_buf) { IMP_CUDA_CHECK_LOG(cudaFree(q8_1_buf)); q8_1_buf = nullptr; }
    if (d8_buf) { IMP_CUDA_CHECK_LOG(cudaFree(d8_buf)); d8_buf = nullptr; }
    q8_1_max_blocks = 0;

    if (splitk) { IMP_CUDA_CHECK_LOG(cudaFree(splitk)); splitk = nullptr; }
    splitk_size = 0;
}

} // namespace imp
