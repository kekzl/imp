// core/device_info.h: cached current-device SM count and compute capability. CPU lane has no
// device: SM count falls back to kSm120SmCount (170), device_is_sm120() is false.
#include <gtest/gtest.h>
#include "core/device_info.h"
#include "core/pdl.h"
#include "core/sm120_limits.h"

#include <cuda_runtime.h>

namespace imp {
namespace {

struct DirectQuery {
    int sm_count = 0;  // 0: failed
    int cc = 0;        // 0: failed
};

DirectQuery query_directly() {
    DirectQuery q;
    int dev = 0, n = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        (void)cudaGetLastError();
        return q;
    }
    if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && n > 0)
        q.sm_count = n;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess)
        q.cc = major * 10 + minor;
    (void)cudaGetLastError();
    return q;
}

TEST(DeviceInfoTest, SmCountMatchesDirectQueryOrFallsBackTo170) {
    const DirectQuery q = query_directly();
    EXPECT_EQ(device_sm_count_or_0(), q.sm_count);
    EXPECT_EQ(device_sm_count(), q.sm_count > 0 ? q.sm_count : kSm120SmCount);
    if (q.sm_count == 0) {
        EXPECT_EQ(device_sm_count(), 170);
    }
}

TEST(DeviceInfoTest, Sm120FalseOnFailedQueryAndBelowCc120) {
    const DirectQuery q = query_directly();
    EXPECT_EQ(device_cc(), q.cc);
    EXPECT_EQ(device_is_sm120(), q.cc >= 120);
    if (q.cc == 0) {
        EXPECT_FALSE(device_is_sm120());
    }
}

// pdl keeps its own threshold: cc >= 9.0, false on a failed query.
TEST(DeviceInfoTest, PdlAvailableIsCc90OrAbove) { EXPECT_EQ(pdl::is_available(), query_directly().cc >= 90); }

TEST(DeviceInfoTest, CachedValuesAreStable) {
    const int sm = device_sm_count();
    const int cc = device_cc();
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(device_sm_count(), sm);
        EXPECT_EQ(device_cc(), cc);
    }
}

}  // namespace
}  // namespace imp
