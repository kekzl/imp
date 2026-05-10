// src/compute/gemm_grouped_nvfp4_smallM.cu
#include "compute/gemm_grouped_nvfp4_smallM.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace imp {

namespace detail {

int pick_m_tile(int M_e) {
    if (M_e <= 16) return 16;
    if (M_e <= 32) return 32;
    if (M_e <= 64) return 64;
    return 128;
}

std::vector<WorkItem> build_work_queue(int n_experts, const int* M_per, int N) {
    std::vector<WorkItem> q;
    q.reserve((size_t)n_experts * (size_t)((N + 127) / 128) + 8);
    for (int e = 0; e < n_experts; ++e) {
        if (M_per[e] <= 0) continue;
        int tm = pick_m_tile(M_per[e]);
        int nm = (M_per[e] + tm - 1) / tm;
        int nn = (N + 127) / 128;
        for (int mi = 0; mi < nm; ++mi)
            for (int ni = 0; ni < nn; ++ni)
                q.push_back({e, mi, ni, (uint8_t)tm});
    }
    std::stable_sort(q.begin(), q.end(),
        [](const WorkItem& a, const WorkItem& b) {
            return a.m_tile_size > b.m_tile_size;
        });
    return q;
}

}  // namespace detail

static int s_smallM_available = -1;

bool gemm_grouped_nvfp4_smallM_available() {
    if (s_smallM_available >= 0) return s_smallM_available;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    s_smallM_available = (prop.major * 10 + prop.minor >= 120) ? 1 : 0;
    return s_smallM_available;
}

void gemm_grouped_nvfp4_smallM_cleanup() {}

bool gemm_grouped_nvfp4_smallM(
    int /*n_experts*/, const int* /*host_M*/, int /*N*/, int /*K*/,
    const void* const* /*host_ptr_A*/, const void* const* /*host_ptr_SFA*/,
    const void* const* /*host_ptr_B*/, const void* const* /*host_ptr_SFB*/,
    void* const* /*host_ptr_D*/, const float* /*host_alpha*/,
    cudaStream_t /*stream*/) {
    return false;  // skeleton: caller falls back to CUTLASS path
}

}  // namespace imp
