#pragma once
// Sticky CUDA error classification (AUDIT_arch_2026 D-1, #874): a device
// fault poisons the process-wide CUDA context, and hot-path IMP_CUDA_CHECK_*
// is log-only, so without an explicit signal the server could log a cleared
// error and still answer 200 with stale tokens.
// cuda_error_is_unrecoverable(e): the class table; recoverable/unknown never stops the server.
// cuda_clear_or_throw(where): pre-check; benign error cleared, sticky class throws.
// cuda_sync_or_throw(err, where): failed sync means the host buffer was never written; always throws.
// Throw lands in BatchingEngine::step()'s catch (or the C API boundary), which re-probes and decides.
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace imp {

inline bool cuda_error_is_unrecoverable(cudaError_t e) {
    switch (e) {
        case cudaErrorIllegalAddress:
        case cudaErrorMisalignedAddress:
        case cudaErrorLaunchFailure:
        case cudaErrorLaunchTimeout:
        case cudaErrorHardwareStackError:
        case cudaErrorIllegalInstruction:
        case cudaErrorECCUncorrectable:
        case cudaErrorExternalDevice:
            return true;
        default:
            return false;
    }
}

inline cudaError_t cuda_clear_or_throw(const char* where) {
    const cudaError_t e = cudaGetLastError();
    if (cuda_error_is_unrecoverable(e))
        throw std::runtime_error(std::string("CUDA context poisoned (") + cudaGetErrorString(e) +
                                 ") before " + where + ": the process must be restarted");
    return e;
}

inline void cuda_sync_or_throw(cudaError_t err, const char* where) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA sync failed (") + cudaGetErrorString(err) + ") in " +
                                 where + ": the sampled tokens were never written");
}

}  // namespace imp
