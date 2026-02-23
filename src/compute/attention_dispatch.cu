#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

// Forward declaration for Blackwell kernel (only available when compiled for sm_120)
void flash_attention_prefill_blackwell(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, cudaStream_t stream);

static int cached_sm_version = -1;

int get_device_sm_version() {
    if (cached_sm_version >= 0) return cached_sm_version;
    int device = 0;
    cudaGetDevice(&device);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    cached_sm_version = major * 10 + minor;
    IMP_LOG_INFO("Device SM version: %d.%d (sm_%d)", major, minor, cached_sm_version);
    return cached_sm_version;
}

void attention_prefill_dispatch(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, cudaStream_t stream) {
    int sm = get_device_sm_version();
    if (sm >= 120) {
        flash_attention_prefill_blackwell(Q, K, V, O, scale, causal, stream);
    } else if (sm >= 90) {
        flash_attention_prefill_tc(Q, K, V, O, scale, causal, stream);
    } else {
        flash_attention_prefill(Q, K, V, O, scale, causal, stream);
    }
}

} // namespace imp
