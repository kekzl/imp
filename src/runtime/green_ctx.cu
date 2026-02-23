#include "runtime/green_ctx.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <algorithm>

#if IMP_CUDA_13_1
#include <cuda.h>
#endif

namespace imp {

GreenContextManager::~GreenContextManager() {
    destroy();
}

bool GreenContextManager::init(int device, float prefill_sm_ratio) {
    if (available_) {
        destroy();
    }

    device_ = device;
    prefill_ratio_ = prefill_sm_ratio;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("GreenContextManager: failed to set device %d: %s",
                      device, cudaGetErrorString(err));
        return false;
    }

    // Query total SM count
    err = cudaDeviceGetAttribute(&total_sms_,
                                  cudaDevAttrMultiProcessorCount, device);
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

    IMP_LOG_INFO("GreenContextManager: device %d has %d SMs, "
                 "prefill=%d (%.0f%%), decode=%d (%.0f%%)",
                 device, total_sms_,
                 prefill_sms_, 100.0f * prefill_sms_ / total_sms_,
                 decode_sms_, 100.0f * decode_sms_ / total_sms_);

#if IMP_CUDA_13_1
    // --- CUDA 13.1 Green Contexts: true SM partitioning ---
    CUdevice cu_dev;
    CUresult cu_err = cuDeviceGet(&cu_dev, device);
    if (cu_err != CUDA_SUCCESS) {
        IMP_LOG_WARN("GreenContextManager: cuDeviceGet failed, falling back to regular streams");
        goto fallback;
    }

    {
        // Get the full SM resource for this device
        CUdevResource full_sm_resource;
        cu_err = cuDeviceGetDevResource(cu_dev, &full_sm_resource, CU_DEV_RESOURCE_TYPE_SM);
        if (cu_err != CUDA_SUCCESS) {
            IMP_LOG_WARN("GreenContextManager: cuDeviceGetDevResource failed (%d), "
                         "falling back to regular streams", cu_err);
            goto fallback;
        }

        // Split SMs for prefill partition
        // cuDevSmResourceSplitByCount(result, nbGroups, input, remainder, flags, minCount)
        CUdevResource prefill_dev_res;
        CUdevResource decode_dev_res;
        unsigned int nb_groups = 1;
        unsigned int prefill_count = static_cast<unsigned int>(prefill_sms_);
        cu_err = cuDevSmResourceSplitByCount(&prefill_dev_res, &nb_groups,
                                              &full_sm_resource, &decode_dev_res,
                                              0, prefill_count);
        if (cu_err != CUDA_SUCCESS) {
            IMP_LOG_WARN("GreenContextManager: cuDevSmResourceSplitByCount failed (%d), "
                         "falling back to regular streams", cu_err);
            goto fallback;
        }

        cu_err = cuDevResourceGenerateDesc(&prefill_resource_desc_, &prefill_dev_res, 1);
        if (cu_err != CUDA_SUCCESS) {
            IMP_LOG_WARN("GreenContextManager: cuDevResourceGenerateDesc (prefill) "
                         "failed (%d)", cu_err);
            goto fallback;
        }

        cu_err = cuDevResourceGenerateDesc(&decode_resource_desc_, &decode_dev_res, 1);
        if (cu_err != CUDA_SUCCESS) {
            IMP_LOG_WARN("GreenContextManager: cuDevResourceGenerateDesc (decode) "
                         "failed (%d)", cu_err);
            goto fallback;
        }

        // Create green contexts
        cu_err = cuGreenCtxCreate(&prefill_green_ctx_, prefill_resource_desc_,
                                   cu_dev, CU_GREEN_CTX_DEFAULT_STREAM);
        if (cu_err != CUDA_SUCCESS) {
            IMP_LOG_WARN("GreenContextManager: cuGreenCtxCreate (prefill) "
                         "failed (%d)", cu_err);
            goto fallback;
        }

        cu_err = cuGreenCtxCreate(&decode_green_ctx_, decode_resource_desc_,
                                   cu_dev, CU_GREEN_CTX_DEFAULT_STREAM);
        if (cu_err != CUDA_SUCCESS) {
            IMP_LOG_WARN("GreenContextManager: cuGreenCtxCreate (decode) "
                         "failed (%d)", cu_err);
            cuGreenCtxDestroy(prefill_green_ctx_);
            prefill_green_ctx_ = nullptr;
            goto fallback;
        }

        // Get CUcontext from green contexts
        cu_err = cuCtxFromGreenCtx(&prefill_ctx_, prefill_green_ctx_);
        if (cu_err != CUDA_SUCCESS) {
            IMP_LOG_WARN("GreenContextManager: cuCtxFromGreenCtx (prefill) "
                         "failed (%d)", cu_err);
            goto cleanup_green;
        }

        cu_err = cuCtxFromGreenCtx(&decode_ctx_, decode_green_ctx_);
        if (cu_err != CUDA_SUCCESS) {
            IMP_LOG_WARN("GreenContextManager: cuCtxFromGreenCtx (decode) "
                         "failed (%d)", cu_err);
            goto cleanup_green;
        }

        // Create streams on the green contexts
        // Push prefill context, create stream, pop
        cuCtxPushCurrent(prefill_ctx_);
        CUstream cu_prefill_stream;
        cu_err = cuStreamCreate(&cu_prefill_stream, CU_STREAM_NON_BLOCKING);
        cuCtxPopCurrent(nullptr);
        if (cu_err != CUDA_SUCCESS) {
            IMP_LOG_WARN("GreenContextManager: failed to create prefill stream");
            goto cleanup_green;
        }
        prefill_stream_ = reinterpret_cast<cudaStream_t>(cu_prefill_stream);

        // Push decode context, create stream, pop
        cuCtxPushCurrent(decode_ctx_);
        CUstream cu_decode_stream;
        cu_err = cuStreamCreate(&cu_decode_stream, CU_STREAM_NON_BLOCKING);
        cuCtxPopCurrent(nullptr);
        if (cu_err != CUDA_SUCCESS) {
            IMP_LOG_WARN("GreenContextManager: failed to create decode stream");
            cudaStreamDestroy(prefill_stream_);
            prefill_stream_ = nullptr;
            goto cleanup_green;
        }
        decode_stream_ = reinterpret_cast<cudaStream_t>(cu_decode_stream);

        has_green_ctx_ = true;
        available_ = true;
        IMP_LOG_INFO("GreenContextManager: initialized with CUDA 13.1 Green Contexts "
                     "(prefill=%d SMs, decode=%d SMs)", prefill_sms_, decode_sms_);
        return true;

    cleanup_green:
        if (prefill_green_ctx_) {
            cuGreenCtxDestroy(prefill_green_ctx_);
            prefill_green_ctx_ = nullptr;
        }
        if (decode_green_ctx_) {
            cuGreenCtxDestroy(decode_green_ctx_);
            decode_green_ctx_ = nullptr;
        }
        prefill_resource_desc_ = nullptr;
        decode_resource_desc_ = nullptr;
        prefill_ctx_ = nullptr;
        decode_ctx_ = nullptr;
    }

fallback:
#endif  // IMP_CUDA_13_1

    // Fallback: create regular CUDA streams (no SM partitioning)
    IMP_LOG_INFO("GreenContextManager: using regular CUDA streams (no SM partitioning)");

    err = cudaStreamCreate(&prefill_stream_);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("GreenContextManager: failed to create prefill stream");
        return false;
    }

    err = cudaStreamCreate(&decode_stream_);
    if (err != cudaSuccess) {
        cudaStreamDestroy(prefill_stream_);
        prefill_stream_ = nullptr;
        IMP_LOG_ERROR("GreenContextManager: failed to create decode stream");
        return false;
    }

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
    if (prefill_stream_) {
        cudaStreamDestroy(prefill_stream_);
        prefill_stream_ = nullptr;
    }
    if (decode_stream_) {
        cudaStreamDestroy(decode_stream_);
        decode_stream_ = nullptr;
    }

#if IMP_CUDA_13_1
    if (prefill_green_ctx_) {
        cuGreenCtxDestroy(prefill_green_ctx_);
        prefill_green_ctx_ = nullptr;
    }
    if (decode_green_ctx_) {
        cuGreenCtxDestroy(decode_green_ctx_);
        decode_green_ctx_ = nullptr;
    }
    prefill_resource_desc_ = nullptr;
    decode_resource_desc_ = nullptr;
    prefill_ctx_ = nullptr;
    decode_ctx_ = nullptr;
#endif

    has_green_ctx_ = false;
    available_ = false;
    total_sms_ = 0;
    prefill_sms_ = 0;
    decode_sms_ = 0;
}

} // namespace imp
