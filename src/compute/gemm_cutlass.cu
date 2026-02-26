// CUTLASS 2.x FP16 grouped GEMM for MoE expert parallelism.
// Drop-in replacement for gemm_moe_batched() with ~3µs launch overhead
// (vs ~27µs cuBLAS), enabling L2-chunked dequant+GEMM processing.
//
// Uses CUTLASS 2.x GemmGrouped with cp.async (not TMA), which natively
// supports all layout combinations including RowMajor-A × ColumnMajor-B.
// Our B weights [N,K] RowMajor are reinterpreted as [K,N] ColumnMajor.

#include "compute/gemm_cutlass.h"
#include "core/tensor.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>

namespace imp {

// ============================================================================
// CUTLASS 2.x kernel type configuration
// ============================================================================

using ElemA     = cutlass::half_t;
using ElemB     = cutlass::half_t;
using ElemC     = cutlass::half_t;
using ElemAccum = float;

// A [M, K] RowMajor: lda = K
// B [N, K] RowMajor in memory = [K, N] ColumnMajor: ldb = K
// D [M, N] RowMajor: ldd = N
using LayA = cutlass::layout::RowMajor;
using LayB = cutlass::layout::ColumnMajor;
using LayC = cutlass::layout::RowMajor;

static constexpr int kAlign = 128 / cutlass::sizeof_bits<ElemA>::value;  // 8

using EpiOp = cutlass::epilogue::thread::LinearCombination<
    ElemC, kAlign, ElemAccum, ElemAccum>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;

// Tile 128x128x32, 4 cp.async stages, SM80 (compatible with SM90/SM120)
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElemA, LayA, cutlass::ComplexTransform::kNone, kAlign,
    ElemB, LayB, cutlass::ComplexTransform::kNone, kAlign,
    ElemC, LayC, ElemAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpiOp, Swizzle,
    4,  // stages
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly
>::GemmKernel;

using GDev = cutlass::gemm::device::GemmGrouped<GemmKernel>;

// ============================================================================
// Persistent device buffers (grow-only, process lifetime)
// ============================================================================

static void*  s_staging    = nullptr;
static size_t s_staging_sz = 0;
static void*  s_workspace  = nullptr;
static size_t s_workspace_sz = 0;

static size_t align8(size_t x) { return (x + 7) & ~size_t(7); }

// ============================================================================
// gemm_moe_cutlass
// ============================================================================

void gemm_moe_cutlass(const void* a_base, void* c_base,
                      const int32_t* offsets,
                      const void* const* b_ptrs,
                      int K, int N, DType dtype, int n_experts,
                      cudaStream_t stream,
                      void** /*d_work_ptrs*/)
{
    if (n_experts == 0) return;
    if (dtype != DType::FP16) {
        fprintf(stderr, "imp::gemm_moe_cutlass: only FP16 supported\n");
        return;
    }
    constexpr size_t esz = sizeof(half);

    // 1. Count active experts
    int n_active = 0;
    for (int e = 0; e < n_experts; e++) {
        if (offsets[e + 1] > offsets[e]) n_active++;
    }
    if (n_active == 0) return;

    // 2. Build host-side arrays
    using GemmCoord = cutlass::gemm::GemmCoord;

    std::vector<GemmCoord> h_sizes(n_active);
    std::vector<ElemA*>    h_a(n_active);
    std::vector<ElemB*>    h_b(n_active);
    std::vector<ElemC*>    h_d(n_active);
    std::vector<int64_t>   h_lda(n_active);
    std::vector<int64_t>   h_ldb(n_active);
    std::vector<int64_t>   h_ldd(n_active);

    const char* ab = static_cast<const char*>(a_base);
    char*       cb = static_cast<char*>(c_base);

    int gi = 0;
    for (int e = 0; e < n_experts; e++) {
        int M = offsets[e + 1] - offsets[e];
        if (M <= 0) continue;

        h_sizes[gi] = GemmCoord(M, N, K);
        // const_cast needed: CUTLASS 2.x Arguments use non-const pointers
        h_a[gi] = reinterpret_cast<ElemA*>(const_cast<char*>(ab + (size_t)offsets[e] * K * esz));
        h_b[gi] = reinterpret_cast<ElemB*>(const_cast<void*>(b_ptrs[e]));
        h_d[gi] = reinterpret_cast<ElemC*>(cb + (size_t)offsets[e] * N * esz);
        h_lda[gi] = K;   // A [M, K] RowMajor stride
        h_ldb[gi] = K;   // B [K, N] ColumnMajor stride (= [N, K] RowMajor in memory)
        h_ldd[gi] = N;   // D [M, N] RowMajor stride
        gi++;
    }

    // 3. Device staging: [sizes | a_ptrs | b_ptrs | d_ptrs | lda | ldb | ldd]
    size_t o0 = 0;
    size_t o1 = align8(o0 + n_active * sizeof(GemmCoord));
    size_t o2 = align8(o1 + n_active * sizeof(ElemA*));
    size_t o3 = align8(o2 + n_active * sizeof(ElemB*));
    size_t o4 = align8(o3 + n_active * sizeof(ElemC*));
    size_t o5 = align8(o4 + n_active * sizeof(int64_t));
    size_t o6 = align8(o5 + n_active * sizeof(int64_t));
    size_t total = align8(o6 + n_active * sizeof(int64_t));

    if (total > s_staging_sz) {
        if (s_staging) cudaFree(s_staging);
        cudaMalloc(&s_staging, total);
        s_staging_sz = total;
    }

    char* dv = static_cast<char*>(s_staging);

    // H2D copies (pageable → synchronous from host, ordered on stream)
    cudaMemcpyAsync(dv+o0, h_sizes.data(), n_active*sizeof(GemmCoord), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dv+o1, h_a.data(),     n_active*sizeof(ElemA*),    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dv+o2, h_b.data(),     n_active*sizeof(ElemB*),    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dv+o3, h_d.data(),     n_active*sizeof(ElemC*),    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dv+o4, h_lda.data(),   n_active*sizeof(int64_t),   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dv+o5, h_ldb.data(),   n_active*sizeof(int64_t),   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dv+o6, h_ldd.data(),   n_active*sizeof(int64_t),   cudaMemcpyHostToDevice, stream);

    // 4. Compute threadblock count
    int tb_count = GDev::sufficient(h_sizes.data(), n_active);
    if (tb_count <= 0) return;

    // 5. Arguments
    typename GDev::Arguments args(
        reinterpret_cast<GemmCoord*>(dv + o0),     // device problem_sizes
        n_active,
        tb_count,
        typename EpiOp::Params{1.0f, 0.0f},        // alpha=1, beta=0
        reinterpret_cast<ElemA**>(dv + o1),         // ptr_A (device)
        reinterpret_cast<ElemB**>(dv + o2),         // ptr_B (device)
        reinterpret_cast<ElemC**>(dv + o3),         // ptr_C = ptr_D (beta=0)
        reinterpret_cast<ElemC**>(dv + o3),         // ptr_D (device)
        reinterpret_cast<int64_t*>(dv + o4),        // lda (device)
        reinterpret_cast<int64_t*>(dv + o5),        // ldb (device)
        reinterpret_cast<int64_t*>(dv + o6),        // ldc = ldd (beta=0)
        reinterpret_cast<int64_t*>(dv + o6),        // ldd (device)
        h_sizes.data()                               // host_problem_sizes
    );

    // 6. Check can_implement
    {
        cutlass::Status ci = GDev::can_implement(args);
        if (ci != cutlass::Status::kSuccess) {
            fprintf(stderr, "imp::gemm_moe_cutlass: can_implement failed (%d), n_active=%d K=%d N=%d\n",
                    (int)ci, n_active, K, N);
            return;
        }
    }

    // 7. Workspace
    size_t ws = GDev::get_workspace_size(args);
    if (ws > s_workspace_sz) {
        if (s_workspace) cudaFree(s_workspace);
        cudaMalloc(&s_workspace, ws);
        s_workspace_sz = ws;
    }

    // 8. Initialize + launch (~3µs total host overhead)
    GDev gemm;
    cutlass::Status st = gemm.initialize(args, s_workspace, stream);
    if (st != cutlass::Status::kSuccess) {
        fprintf(stderr, "imp::gemm_moe_cutlass: init failed (%d)\n", (int)st);
        return;
    }

    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) {
        fprintf(stderr, "imp::gemm_moe_cutlass: run failed (%d)\n", (int)st);
    }
}

} // namespace imp
