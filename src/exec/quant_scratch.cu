#include "exec/quant_scratch.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

// d_act_scale, d_fp8_block_maxes and d_fp8_absmax are T2 arena tenants since
// A7 step 4b.2 — engine-lifetime, charged by exec_t2_demand as `fp8_reduction`.
// ~Engine closes the arena after every executor teardown, so only the pointer
// nulling remains here.

void QuantScratch::free(VRAMAllocator* alloc) {
    auto vfree = [alloc](void*& p) {
        if (!p)
            return;
        if (alloc)
            alloc->free(p);
        else
            IMP_CUDA_CHECK_LOG(cudaFree(p));
        p = nullptr;
    };

    vfree(dequant);
    dequant_size = 0;

    vfree(fp8_act);
    fp8_act_size = 0;
    if (d_act_scale) {
        d_act_scale = nullptr;
    }
    if (d_fp8_block_maxes) {
        d_fp8_block_maxes = nullptr;
    }
    if (d_fp8_absmax) {
        d_fp8_absmax = nullptr;
    }
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

    if (q8_1_buf) {
        IMP_CUDA_CHECK_LOG(cudaFree(q8_1_buf));
        q8_1_buf = nullptr;
    }
    if (d8_buf) {
        IMP_CUDA_CHECK_LOG(cudaFree(d8_buf));
        d8_buf = nullptr;
    }
    q8_1_max_blocks = 0;

    if (q8_1_prefill_buf) {
        IMP_CUDA_CHECK_LOG(cudaFree(q8_1_prefill_buf));
        q8_1_prefill_buf = nullptr;
    }
    if (d8_prefill_buf) {
        IMP_CUDA_CHECK_LOG(cudaFree(d8_prefill_buf));
        d8_prefill_buf = nullptr;
    }
    q8_1_prefill_bytes = 0;
    d8_prefill_bytes = 0;

    if (ffn_block_mask) {
        IMP_CUDA_CHECK_LOG(cudaFree(ffn_block_mask));
        ffn_block_mask = nullptr;
    }
    ffn_block_mask_words = 0;

    if (splitk) {
        IMP_CUDA_CHECK_LOG(cudaFree(splitk));
        splitk = nullptr;
    }
    splitk_size = 0;
}

}  // namespace imp
