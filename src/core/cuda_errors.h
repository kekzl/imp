#pragma once
// Sticky CUDA error classification (AUDIT_arch_2026 D-1, #874): a device
// fault poisons the process-wide CUDA context, and hot-path IMP_CUDA_CHECK_*
// is log-only, so without an explicit signal the server could log a cleared
// error and still answer 200 with stale tokens.
// cuda_error_is_unrecoverable(e): the class table; recoverable/unknown never stops the server.
// cuda_clear_or_throw(where): pre-check; benign error cleared, sticky class throws.
// cuda_sync_or_throw(err, where): failed sync means the host buffer was never written; always throws.
// cuda_alloc_or_throw(err, where): failed allocation with no fallback; throws before a launch.
// Throw lands in BatchingEngine::step()'s catch (or the C API boundary), which re-probes and decides.
#include <cuda_runtime_api.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace imp {

[[nodiscard]] inline bool cuda_error_is_unrecoverable(cudaError_t e) {
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

// Device allocation with no fallback: failure throws before any launch on the buffer (#2446).
// An OOM is not sticky, so it is cleared and the next launch check does not report it again.
inline void cuda_alloc_or_throw(cudaError_t err, const char* where) {
    if (err == cudaSuccess)
        return;
    (void)cudaGetLastError();
    throw std::runtime_error(std::string("CUDA allocation failed (") + cudaGetErrorString(err) + ") in " +
                             where);
}

// Pinned sampler readback: *h_token after a successful sync, else throws; never a token the model did not sample.
[[nodiscard]] inline int32_t synced_token_or_throw(cudaError_t sync_err, const int32_t* h_token, const char* where) {
    cuda_sync_or_throw(sync_err, where);
    return *h_token;
}

// Sampler scratch alloc / readback enqueue: failure throws; no token 0 in place of a sample (#2307).
inline void cuda_call_or_throw(cudaError_t err, const char* where) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA call failed (") + cudaGetErrorString(err) + ") in " +
                                 where + ": no token was sampled");
}

// Sampler that enqueued nothing (CUB scratch unavailable): throws; its slot holds a stale token (#2307).
inline void sampler_enqueued_or_throw(bool enqueued, const char* where) {
    if (!enqueued)
        throw std::runtime_error(std::string("sampler not enqueued in ") + where + ": no token was sampled");
}

}  // namespace imp
