// src/compute/gemm_grouped_nvfp4_smallM.cu
#include "compute/gemm_grouped_nvfp4_smallM.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace imp {

namespace {

// Inline-PTX wrapper for the block-scaled MMA on SM120.
// Issues 1 mma.sync that consumes:
//   A: 16x64 FP4 (4 b32 registers per warp)
//   B: 8x64 FP4 (2 b32 registers per warp)
//   SFA: 4 UE4M3 scales per group (packed in 1 b32)
//   SFB: 4 UE4M3 scales per group (packed in 1 b32)
//   D: accumulator FP32, 4 floats per thread (16x8 owned by warp)
//
// Validated 268 TOPS via tests/test_mxf4nvf4_mma_variants_bench.cu.
// bid/tid for scale addressing are zero (all threads in the same tile see the
// same scale register — matches the load pattern in the full kernel).
__device__ __forceinline__ void mma_sync_mxf4nvf4_m16n8k64(
    float* d,           // 4 floats in/out (FP32 accumulator, C→D)
    const uint32_t* a,  // 4 uint32 (A fragment for the warp)
    const uint32_t* b,  // 2 uint32 (B fragment for the warp)
    uint32_t sfa,       // 1 uint32 = 4 UE4M3 scales for A
    uint32_t sfb) {     // 1 uint32 = 4 UE4M3 scales for B
#if (__CUDA_ARCH__ >= 1200)
    constexpr uint16_t bid = 0, tid = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32."
        "ue4m3 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%10,%11,%12,%13},"
        "{%14},{%15,%16},"
        "{%17},{%18,%19};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]),
          "r"(sfa),  "h"(bid),  "h"(tid),
          "r"(sfb),  "h"(bid),  "h"(tid));
#else
    (void)d; (void)a; (void)b; (void)sfa; (void)sfb;
#endif
}

}  // anonymous namespace

#ifdef SMALLM_TEST_HOOKS
namespace imp_test {

__global__ void smallM_smoke_single_mma_kernel(
    float* d_out, const uint32_t* a, const uint32_t* b,
    uint32_t sfa, uint32_t sfb) {
    if (threadIdx.x < 32) {  // single warp
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        mma_sync_mxf4nvf4_m16n8k64(acc, a, b, sfa, sfb);
        if (threadIdx.x == 0) {
            d_out[0] = acc[0]; d_out[1] = acc[1];
            d_out[2] = acc[2]; d_out[3] = acc[3];
        }
    }
}

}  // namespace imp_test

extern "C" void smallM_smoke_single_mma(
    float* d_out, const uint32_t* a, const uint32_t* b,
    uint32_t sfa, uint32_t sfb, cudaStream_t stream) {
    imp_test::smallM_smoke_single_mma_kernel<<<1, 32, 0, stream>>>(d_out, a, b, sfa, sfb);
}
#endif  // SMALLM_TEST_HOOKS

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
