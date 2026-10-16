#include "core/device_info.h"

#include "core/logging.h"
#include "core/sm120_limits.h"

#include <cuda_runtime.h>

namespace imp {
namespace {

struct DeviceInfo {
    int sm_count = 0;  // 0: query failed
    int cc = 0;        // major * 10 + minor; 0: query failed
};

DeviceInfo query_device_info() {
    DeviceInfo info;
    int dev = 0;
    if (const cudaError_t e = cudaGetDevice(&dev); e != cudaSuccess) {
        IMP_LOG_WARN("device query failed: %s; SM-count fallbacks in use, sm_120 kernels off",
                     cudaGetErrorString(e));
        return info;
    }
    int n = 0;
    if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || n <= 0)
        IMP_LOG_WARN("device %d: SM count query failed; SM-count fallbacks in use", dev);
    else
        info.sm_count = n;
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess)
        IMP_LOG_WARN("device %d: compute capability query failed; sm_120 kernels off", dev);
    else
        info.cc = major * 10 + minor;
    return info;
}

const DeviceInfo& device_info() {
    static const DeviceInfo info = query_device_info();  // thread-safe, once per process
    return info;
}

}  // namespace

int device_sm_count_or_0() { return device_info().sm_count; }

int device_sm_count() {
    const int n = device_info().sm_count;
    return n > 0 ? n : kSm120SmCount;
}

int device_cc() { return device_info().cc; }

bool device_is_sm120() { return device_info().cc >= 120; }

}  // namespace imp
