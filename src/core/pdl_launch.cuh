#pragma once

// pdl::launch: host side, but the <<<>>> fallback needs nvcc; include from .cu only (#2209).

#include "core/pdl.h"

namespace imp {
namespace pdl {

// PDL-aware kernel launch via cudaLaunchKernelEx with
// ProgrammaticStreamSerialization; falls back to plain <<<>>> when PDL is
// off/unavailable. Registration is the promise that a kernel calls
// pdl_wait() before its first global access and pdl_trigger() after its
// last input read (cuda_graph.cu only converts an edge when the CONSUMER is
// registered); a kernel without pdl_wait() must never be registered.
// Usage: pdl::launch(my_kernel, grid, block, smem, stream, arg1, arg2, ...);
template <typename KernelFunc, typename... Args>
void launch(KernelFunc func, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args... args) {
    const void* func_ptr = reinterpret_cast<const void*>(func);
    if (is_enabled(func_ptr)) {
        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = smem;
        config.stream = stream;

        cudaLaunchAttribute attr = {};
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;

        config.attrs = &attr;
        config.numAttrs = 1;

        // Report-only: the error also stays in cudaGetLastError, as with <<<>>>.
        IMP_CUDA_CHECK_LOG(cudaLaunchKernelEx(&config, func, args...));
    } else {
        func<<<grid, block, smem, stream>>>(args...);
    }
}

}  // namespace pdl
}  // namespace imp
