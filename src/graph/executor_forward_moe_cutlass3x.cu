#include "graph/executor_forward_moe_cutlass3x.h"
#include "compute/gemm_cutlass_grouped_3x.h"
#include "core/logging.h"
#include <vector>

namespace imp {

__attribute__((noinline))
bool dispatch_gate_up_grouped_fused(
    int ne_active,
    const int* host_M,
    int N, int K,
    const void* const* host_ptr_A,
    const void* const* host_ptr_SFA,
    const void* const* host_ptr_B_gate,
    const void* const* host_ptr_SFB_gate,
    const void* const* host_ptr_B_up,
    const void* const* host_ptr_SFB_up,
    void* const* host_ptr_D_gate,
    void* const* host_ptr_D_up,
    const float* host_alpha_gate,
    const float* host_alpha_up,
    cudaStream_t stream) {
    if (ne_active <= 0) return true;

    std::vector<int> M_2x(2 * ne_active);
    std::vector<const void*> A_2x(2 * ne_active), SFA_2x(2 * ne_active);
    std::vector<const void*> B_2x(2 * ne_active), SFB_2x(2 * ne_active);
    std::vector<void*> D_2x(2 * ne_active);
    std::vector<float> alpha_2x(2 * ne_active);

    for (int e = 0; e < ne_active; ++e) {
        // gate at index e
        M_2x[e] = host_M[e];
        A_2x[e] = host_ptr_A[e];
        SFA_2x[e] = host_ptr_SFA[e];
        B_2x[e] = host_ptr_B_gate[e];
        SFB_2x[e] = host_ptr_SFB_gate[e];
        D_2x[e] = host_ptr_D_gate[e];
        alpha_2x[e] = host_alpha_gate[e];
        // up at index ne_active + e
        M_2x[ne_active + e] = host_M[e];
        A_2x[ne_active + e] = host_ptr_A[e];
        SFA_2x[ne_active + e] = host_ptr_SFA[e];
        B_2x[ne_active + e] = host_ptr_B_up[e];
        SFB_2x[ne_active + e] = host_ptr_SFB_up[e];
        D_2x[ne_active + e] = host_ptr_D_up[e];
        alpha_2x[ne_active + e] = host_alpha_up[e];
    }

    return gemm_grouped_cutlass_3x_nvfp4(
        2 * ne_active, M_2x.data(), N, K,
        A_2x.data(), SFA_2x.data(),
        B_2x.data(), SFB_2x.data(),
        D_2x.data(), alpha_2x.data(),
        stream);
}

}  // namespace imp
