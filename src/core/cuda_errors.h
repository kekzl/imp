#pragma once
// Sticky CUDA error classification (AUDIT_arch_2026 D-1, #874).
//
// A device fault inside a kernel (illegal address, launch failure, ...)
// poisons the process-wide CUDA context: every later runtime call returns the
// same error and nothing clears it. Every IMP_CUDA_CHECK_* in the hot path is
// log-only, so without an explicit host signal the next forward() logged
// "Cleared stale error", the sampler returned the previous step's pinned
// tokens and the server answered 200 with /health ok. These three are the
// only places that turn a sticky error into a host signal:
//
//   cuda_error_is_unrecoverable(e)   the class table; a recoverable or unknown
//                                    class never stops the server
//   cuda_clear_or_throw(where)       forward() pre-check: a benign pending
//                                    error is cleared as before and returned
//                                    for the log line, a sticky class throws
//   cuda_sync_or_throw(err, where)   a failed stream/event sync means the host
//                                    buffer the caller is about to read was
//                                    never written; any failure throws
//
// The throw lands in BatchingEngine's step() catch (or the C API boundary),
// which re-probes the device and decides faulted-or-recover.
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
