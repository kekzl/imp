// CUTLASS grouped GEMM for MoE expert dispatch.
//
// Uses CUTLASS 2.x GemmGrouped with kDeviceOnly scheduling — a single
// persistent kernel launch processes all active experts. Each expert has
// independent M (token count) but shared N, K dimensions.
//
// The SM80 cp.async pipeline is forward-compatible and runs efficiently
// on SM90 (Hopper) and SM120 (Blackwell) GPUs.
//
// This is a thin wrapper around the existing CUTLASS 2.x GemmGrouped
// pattern, matching the gemm_cutlass.cu implementation but with a
// different API for integration into gemm_moe_batched().

#include "compute/gemm_cutlass_grouped_sm120.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace imp {

// ============================================================================
// CUTLASS 2.x kernel type configuration (SM80 forward-compat to SM120)
// ============================================================================

using GrpElemA     = cutlass::half_t;
using GrpElemB     = cutlass::half_t;
using GrpElemC     = cutlass::half_t;
using GrpElemAccum = float;

// A [M, K] RowMajor: lda = K
// B [N, K] RowMajor in memory = [K, N] ColumnMajor: ldb = K
// D [M, N] RowMajor: ldd = N
using GrpLayA = cutlass::layout::RowMajor;
using GrpLayB = cutlass::layout::ColumnMajor;
using GrpLayC = cutlass::layout::RowMajor;

static constexpr int kGrpAlign = 128 / cutlass::sizeof_bits<GrpElemA>::value;  // 8

using GrpEpiOp = cutlass::epilogue::thread::LinearCombination<
    GrpElemC, kGrpAlign, GrpElemAccum, GrpElemAccum>;

using GrpSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;

// Tile 128x128x32, 4 cp.async stages, SM80 (compatible with SM90/SM120)
using GrpGemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    GrpElemA, GrpLayA, cutlass::ComplexTransform::kNone, kGrpAlign,
    GrpElemB, GrpLayB, cutlass::ComplexTransform::kNone, kGrpAlign,
    GrpElemC, GrpLayC, GrpElemAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    GrpEpiOp, GrpSwizzle,
    4,  // stages
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly
>::GemmKernel;

using GrpGDev = cutlass::gemm::device::GemmGrouped<GrpGemmKernel>;

// ============================================================================
// Persistent device buffers (grow-only, process lifetime)
// ============================================================================

static void*  s_staging    = nullptr;
static size_t s_staging_sz = 0;
static void*  s_workspace  = nullptr;
static size_t s_workspace_sz = 0;

static size_t align8(size_t x) { return (x + 7) & ~size_t(7); }

// Cache the availability check.
static int s_available = -1;

bool cutlass_grouped_gemm_sm120_available() {
    if (s_available >= 0) return s_available;

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    // SM80+ for CUTLASS 2.x cp.async (Ampere, Hopper, Blackwell)
    s_available = (prop.major >= 8) ? 1 : 0;
    return s_available;
}

bool gemm_grouped_cutlass_sm120(
    const void* const* d_A_ptrs,
    const void* const* d_B_ptrs,
    void* const* d_C_ptrs,
    const int* d_problem_m,
    int N, int K,
    int n_problems,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream)
{
    (void)workspace;
    (void)workspace_size;
    if (n_problems == 0) return true;

    // D2H copy for per-expert M values (prefill path).
    std::vector<int> h_M(n_problems);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_M.data(), d_problem_m, n_problems * sizeof(int),
                    cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Also need host-side data pointers (d_A_ptrs etc are device arrays)
    std::vector<const void*> h_A(n_problems);
    std::vector<const void*> h_B(n_problems);
    std::vector<void*>       h_C(n_problems);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_A.data(), d_A_ptrs, n_problems * sizeof(void*),
                    cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_B.data(), d_B_ptrs, n_problems * sizeof(void*),
                    cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_C.data(), d_C_ptrs, n_problems * sizeof(void*),
                    cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Build CUTLASS 2.x GemmGrouped host arrays
    using GemmCoord = cutlass::gemm::GemmCoord;

    std::vector<GemmCoord> h_sizes(n_problems);
    std::vector<GrpElemA*> h_a(n_problems);
    std::vector<GrpElemB*> h_b(n_problems);
    std::vector<GrpElemC*> h_d(n_problems);
    std::vector<int64_t>   h_lda(n_problems);
    std::vector<int64_t>   h_ldb(n_problems);
    std::vector<int64_t>   h_ldd(n_problems);

    for (int i = 0; i < n_problems; i++) {
        h_sizes[i] = GemmCoord(h_M[i], N, K);
        h_a[i] = reinterpret_cast<GrpElemA*>(const_cast<void*>(h_A[i]));
        h_b[i] = reinterpret_cast<GrpElemB*>(const_cast<void*>(h_B[i]));
        h_d[i] = reinterpret_cast<GrpElemC*>(h_C[i]);
        h_lda[i] = K;  // A [M, K] RowMajor
        h_ldb[i] = K;  // B [K, N] ColumnMajor (= [N, K] RowMajor in memory)
        h_ldd[i] = N;  // D [M, N] RowMajor
    }

    // Device staging: [sizes | a_ptrs | b_ptrs | d_ptrs | lda | ldb | ldd]
    size_t o0 = 0;
    size_t o1 = align8(o0 + n_problems * sizeof(GemmCoord));
    size_t o2 = align8(o1 + n_problems * sizeof(GrpElemA*));
    size_t o3 = align8(o2 + n_problems * sizeof(GrpElemB*));
    size_t o4 = align8(o3 + n_problems * sizeof(GrpElemC*));
    size_t o5 = align8(o4 + n_problems * sizeof(int64_t));
    size_t o6 = align8(o5 + n_problems * sizeof(int64_t));
    size_t total = align8(o6 + n_problems * sizeof(int64_t));

    if (total > s_staging_sz) {
        if (s_staging) IMP_CUDA_CHECK_LOG(cudaFree(s_staging));
        IMP_CUDA_CHECK_LOG(cudaMalloc(&s_staging, total));
        s_staging_sz = total;
    }

    char* dv = static_cast<char*>(s_staging);

    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dv+o0, h_sizes.data(), n_problems*sizeof(GemmCoord), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dv+o1, h_a.data(),     n_problems*sizeof(GrpElemA*), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dv+o2, h_b.data(),     n_problems*sizeof(GrpElemB*), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dv+o3, h_d.data(),     n_problems*sizeof(GrpElemC*), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dv+o4, h_lda.data(),   n_problems*sizeof(int64_t),   cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dv+o5, h_ldb.data(),   n_problems*sizeof(int64_t),   cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dv+o6, h_ldd.data(),   n_problems*sizeof(int64_t),   cudaMemcpyHostToDevice, stream));

    // Compute threadblock count
    int tb_count = GrpGDev::sufficient(h_sizes.data(), n_problems);
    if (tb_count <= 0) return false;

    // Build CUTLASS arguments
    typename GrpGDev::Arguments args(
        reinterpret_cast<GemmCoord*>(dv + o0),
        n_problems,
        tb_count,
        typename GrpEpiOp::Params{1.0f, 0.0f},
        reinterpret_cast<GrpElemA**>(dv + o1),
        reinterpret_cast<GrpElemB**>(dv + o2),
        reinterpret_cast<GrpElemC**>(dv + o3),   // ptr_C = ptr_D (beta=0)
        reinterpret_cast<GrpElemC**>(dv + o3),   // ptr_D
        reinterpret_cast<int64_t*>(dv + o4),
        reinterpret_cast<int64_t*>(dv + o5),
        reinterpret_cast<int64_t*>(dv + o6),     // ldc = ldd
        reinterpret_cast<int64_t*>(dv + o6),     // ldd
        h_sizes.data()
    );

    // Check can_implement
    {
        cutlass::Status ci = GrpGDev::can_implement(args);
        if (ci != cutlass::Status::kSuccess) {
            IMP_LOG_WARN("CUTLASS grouped: can_implement failed (%d) n=%d K=%d N=%d",
                         (int)ci, n_problems, K, N);
            return false;
        }
    }

    // Workspace
    size_t ws = GrpGDev::get_workspace_size(args);
    if (ws > s_workspace_sz) {
        if (s_workspace) IMP_CUDA_CHECK_LOG(cudaFree(s_workspace));
        IMP_CUDA_CHECK_LOG(cudaMalloc(&s_workspace, ws));
        s_workspace_sz = ws;
    }

    // Initialize + launch
    GrpGDev gemm;
    cutlass::Status st = gemm.initialize(args, s_workspace, stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS grouped: init failed (%d)", (int)st);
        return false;
    }

    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS grouped: run failed (%d)", (int)st);
        return false;
    }

    return true;
}

size_t gemm_grouped_cutlass_sm120_workspace(int max_problems, int max_M, int N, int K) {
    (void)max_problems; (void)max_M; (void)N; (void)K;
    return 32ULL << 20;  // 32 MiB default
}

} // namespace imp
