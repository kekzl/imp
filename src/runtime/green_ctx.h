#pragma once

#include <cuda_runtime.h>

#if IMP_CUDA_13_1
#include <cuda.h>
#endif

namespace imp {

class GreenContextManager {
public:
    GreenContextManager() = default;
    ~GreenContextManager();

    // Initialize with SM partitioning: prefill gets prefill_sm_ratio of SMs,
    // decode gets the remainder. device is the CUDA device ordinal.
    bool init(int device, float prefill_sm_ratio = 0.8f);
    void destroy();

    // Reconfigure SM split at runtime (requires destroy + reinit under the hood)
    bool reconfigure(float new_prefill_sm_ratio);

    // Streams bound to SM partitions
    cudaStream_t prefill_stream() const { return prefill_stream_; }
    cudaStream_t decode_stream() const { return decode_stream_; }

    bool is_available() const { return available_; }
    bool has_green_contexts() const { return has_green_ctx_; }

    // Query SM allocation
    int total_sms() const { return total_sms_; }
    int prefill_sms() const { return prefill_sms_; }
    int decode_sms() const { return decode_sms_; }
    float prefill_ratio() const { return prefill_ratio_; }

private:
    cudaStream_t prefill_stream_ = nullptr;
    cudaStream_t decode_stream_ = nullptr;
    bool available_ = false;
    bool has_green_ctx_ = false;

    int device_ = 0;
    int total_sms_ = 0;
    int prefill_sms_ = 0;
    int decode_sms_ = 0;
    float prefill_ratio_ = 0.8f;

#if IMP_CUDA_13_1
    CUgreenCtx prefill_green_ctx_ = nullptr;
    CUgreenCtx decode_green_ctx_ = nullptr;
    CUdevResourceDesc prefill_resource_desc_ = nullptr;
    CUdevResourceDesc decode_resource_desc_ = nullptr;
    CUcontext prefill_ctx_ = nullptr;
    CUcontext decode_ctx_ = nullptr;
#endif
};

} // namespace imp
