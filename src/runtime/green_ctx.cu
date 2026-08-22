#include "runtime/green_ctx.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <algorithm>

namespace imp {

// Memory sync domains separate prefill and decode so concurrent operations
// on different domains don't need to honor implicit memory ordering. Explicit
// event/stream waits still work. Reduces cross-stream fence overhead on
// sm_90+ when prefill and decode overlap under a green context split.
// CUDA 13.2: cudaLaunchMemSyncDomainRemote is the non-default domain.
static constexpr cudaLaunchMemSyncDomain kPrefillSyncDomain = cudaLaunchMemSyncDomainDefault;
static constexpr cudaLaunchMemSyncDomain kDecodeSyncDomain = cudaLaunchMemSyncDomainRemote;

// Apply stream attributes (sync domain) best-effort. Failures are ignored —
// the feature requires CUDA 12.2+ and falls back silently on older drivers.
static void apply_stream_sync_domain(cudaStream_t stream, cudaLaunchMemSyncDomain domain) {
    cudaStreamAttrValue v = {};
    v.memSyncDomain = domain;
    cudaStreamSetAttribute(stream, cudaStreamAttributeMemSyncDomain, &v);
    // Clear any error — this attribute is advisory.
    cudaGetLastError();
}

GreenContextManager::~GreenContextManager() { destroy(); }

bool GreenContextManager::init(int device, float prefill_sm_ratio) {
    if (available_) {
        destroy();
    }

    device_ = device;
    prefill_ratio_ = prefill_sm_ratio;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("GreenContextManager: failed to set device %d: %s", device, cudaGetErrorString(err));
        return false;
    }

    // Query total SM count
    err = cudaDeviceGetAttribute(&total_sms_, cudaDevAttrMultiProcessorCount, device);
    if (err != cudaSuccess || total_sms_ <= 0) {
        IMP_LOG_ERROR("GreenContextManager: failed to query SM count");
        return false;
    }

    prefill_sms_ = static_cast<int>(total_sms_ * prefill_sm_ratio);
    prefill_sms_ = std::max(prefill_sms_, 1);
    decode_sms_ = total_sms_ - prefill_sms_;
    decode_sms_ = std::max(decode_sms_, 1);
    // Adjust if we over-allocated
    if (prefill_sms_ + decode_sms_ > total_sms_) {
        prefill_sms_ = total_sms_ - decode_sms_;
    }

    IMP_LOG_INFO(
        "GreenContextManager: device %d has %d SMs, "
        "prefill=%d (%.0f%%), decode=%d (%.0f%%)",
        device, total_sms_, prefill_sms_, 100.0f * prefill_sms_ / total_sms_, decode_sms_,
        100.0f * decode_sms_ / total_sms_);

    // Query stream priority range. `high` is numerically smaller than `low`.
    // Decode latency dominates user-visible TTFT so it gets the highest priority;
    // prefill runs at the lowest priority so it yields to decode work.
    int low_prio = 0, high_prio = 0;
    if (cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio) != cudaSuccess) {
        low_prio = 0;
        high_prio = 0;
    }

    // --- Green Contexts (Runtime API): true SM partitioning ---
    {
        // Get the full SM resource for this device
        cudaDevResource full_sm_resource;
        err = cudaDeviceGetDevResource(device, &full_sm_resource, cudaDevResourceTypeSm);
        if (err != cudaSuccess) {
            IMP_LOG_WARN(
                "GreenContextManager: cudaDeviceGetDevResource failed (%s), "
                "falling back to regular streams",
                cudaGetErrorString(err));
            goto fallback;
        }

        // Split SMs for prefill partition
        cudaDevResource prefill_dev_res;
        cudaDevResource decode_dev_res;
        unsigned int nb_groups = 1;
        unsigned int prefill_count = static_cast<unsigned int>(prefill_sms_);
        err = cudaDevSmResourceSplitByCount(&prefill_dev_res, &nb_groups, &full_sm_resource, &decode_dev_res,
                                            0, prefill_count);
        if (err != cudaSuccess) {
            IMP_LOG_WARN(
                "GreenContextManager: cudaDevSmResourceSplitByCount failed (%s), "
                "falling back to regular streams",
                cudaGetErrorString(err));
            goto fallback;
        }

        err = cudaDevResourceGenerateDesc(&prefill_resource_desc_, &prefill_dev_res, 1);
        if (err != cudaSuccess) {
            IMP_LOG_WARN(
                "GreenContextManager: cudaDevResourceGenerateDesc (prefill) "
                "failed (%s)",
                cudaGetErrorString(err));
            goto fallback;
        }

        err = cudaDevResourceGenerateDesc(&decode_resource_desc_, &decode_dev_res, 1);
        if (err != cudaSuccess) {
            IMP_LOG_WARN(
                "GreenContextManager: cudaDevResourceGenerateDesc (decode) "
                "failed (%s)",
                cudaGetErrorString(err));
            goto fallback;
        }

        // Create green contexts (flags=0 for Runtime API)
        err = cudaGreenCtxCreate(&prefill_green_ctx_, prefill_resource_desc_, device, 0);
        if (err != cudaSuccess) {
            IMP_LOG_WARN(
                "GreenContextManager: cudaGreenCtxCreate (prefill) "
                "failed (%s)",
                cudaGetErrorString(err));
            goto fallback;
        }

        err = cudaGreenCtxCreate(&decode_green_ctx_, decode_resource_desc_, device, 0);
        if (err != cudaSuccess) {
            IMP_LOG_WARN(
                "GreenContextManager: cudaGreenCtxCreate (decode) "
                "failed (%s)",
                cudaGetErrorString(err));
            cudaExecutionCtxDestroy(prefill_green_ctx_);
            prefill_green_ctx_ = nullptr;
            goto fallback;
        }

        // Create streams directly on green contexts. Priority: decode > prefill
        // so the scheduler yields to latency-critical decode work when both are
        // ready on different SM partitions.
        err = cudaExecutionCtxStreamCreate(&prefill_stream_, prefill_green_ctx_, cudaStreamNonBlocking,
                                           low_prio);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("GreenContextManager: failed to create prefill stream (%s)",
                         cudaGetErrorString(err));
            goto cleanup_green;
        }

        err = cudaExecutionCtxStreamCreate(&decode_stream_, decode_green_ctx_, cudaStreamNonBlocking,
                                           high_prio);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("GreenContextManager: failed to create decode stream (%s)", cudaGetErrorString(err));
            cudaStreamDestroy(prefill_stream_);
            prefill_stream_ = nullptr;
            goto cleanup_green;
        }

        // Assign distinct memory sync domains. Cross-stream ordering between
        // prefill and decode is handled explicitly via events, so the implicit
        // fences the scheduler otherwise inserts are unnecessary.
        apply_stream_sync_domain(prefill_stream_, kPrefillSyncDomain);
        apply_stream_sync_domain(decode_stream_, kDecodeSyncDomain);

        has_green_ctx_ = true;
        available_ = true;
        IMP_LOG_INFO(
            "GreenContextManager: initialized with CUDA 13.2 Green Contexts "
            "(prefill=%d SMs @ prio %d, decode=%d SMs @ prio %d, "
            "distinct memSyncDomains)",
            prefill_sms_, low_prio, decode_sms_, high_prio);
        return true;

    cleanup_green:
        if (prefill_green_ctx_) {
            cudaExecutionCtxDestroy(prefill_green_ctx_);
            prefill_green_ctx_ = nullptr;
        }
        if (decode_green_ctx_) {
            cudaExecutionCtxDestroy(decode_green_ctx_);
            decode_green_ctx_ = nullptr;
        }
        prefill_resource_desc_ = nullptr;
        decode_resource_desc_ = nullptr;
    }

fallback:
    // Clear any stale CUDA errors from failed green context operations.
    // Without this, the error propagates to subsequent cuBLAS calls
    // (cublasLtMatmul returns INVALID_VALUE) causing output corruption.
    cudaGetLastError();

    // Fallback: create regular CUDA streams (no SM partitioning) but keep
    // the priority + memory-sync-domain separation so that prefill and
    // decode can still overlap efficiently when run concurrently.
    IMP_LOG_INFO(
        "GreenContextManager: using regular CUDA streams with "
        "priority %d/%d and distinct memSyncDomains",
        low_prio, high_prio);

    err = cudaStreamCreateWithPriority(&prefill_stream_, cudaStreamNonBlocking, low_prio);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("GreenContextManager: failed to create prefill stream");
        return false;
    }

    err = cudaStreamCreateWithPriority(&decode_stream_, cudaStreamNonBlocking, high_prio);
    if (err != cudaSuccess) {
        cudaStreamDestroy(prefill_stream_);
        prefill_stream_ = nullptr;
        IMP_LOG_ERROR("GreenContextManager: failed to create decode stream");
        return false;
    }

    apply_stream_sync_domain(prefill_stream_, kPrefillSyncDomain);
    apply_stream_sync_domain(decode_stream_, kDecodeSyncDomain);

    has_green_ctx_ = false;
    available_ = true;
    return true;
}

bool GreenContextManager::reconfigure(float new_prefill_sm_ratio) {
    if (!available_) {
        return false;
    }

    int saved_device = device_;
    destroy();
    return init(saved_device, new_prefill_sm_ratio);
}

void GreenContextManager::destroy() {
    // #1656: reconfigure() calls this from inside step_schedule() whenever the
    // prefill/decode mix changes, i.e. with work in flight. cudaStreamDestroy
    // does not wait, the green context the work runs in is destroyed right
    // after, and the replacement streams carry no ordering against any of it.
    // Draining first costs the reconfigure its latency once; not draining
    // leaves the outcome to timing.
    if (prefill_stream_)
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(prefill_stream_));
    if (decode_stream_)
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(decode_stream_));

    if (prefill_stream_) {
        cudaStreamDestroy(prefill_stream_);
        prefill_stream_ = nullptr;
    }
    if (decode_stream_) {
        cudaStreamDestroy(decode_stream_);
        decode_stream_ = nullptr;
    }

    if (prefill_green_ctx_) {
        cudaExecutionCtxDestroy(prefill_green_ctx_);
        prefill_green_ctx_ = nullptr;
    }
    if (decode_green_ctx_) {
        cudaExecutionCtxDestroy(decode_green_ctx_);
        decode_green_ctx_ = nullptr;
    }
    prefill_resource_desc_ = nullptr;
    decode_resource_desc_ = nullptr;

    has_green_ctx_ = false;
    available_ = false;
    total_sms_ = 0;
    prefill_sms_ = 0;
    decode_sms_ = 0;
}

}  // namespace imp
