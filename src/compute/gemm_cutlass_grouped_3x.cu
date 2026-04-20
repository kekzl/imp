// CUTLASS 3.x NVFP4 BlockScaled Grouped GEMM for MoE (SM120).
// Based on Example 79d: Blackwell GeForce NVFP4 Grouped GEMM.
//
// Zero D2H sync during execution — per-expert device pointer arrays are
// pre-built once per call on the host (no problem-shape lookup on GPU),
// which CUTLASS then indexes directly for its GroupProblemShape scheduler.

#include "compute/gemm_cutlass_grouped_3x.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ---------------------------------------------------------------------------
// CUTLASS 3.x kernel config: NVFP4 × NVFP4 → FP16 Grouped (SM120)
// ---------------------------------------------------------------------------

using GrpProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

using GrpElementInput = cutlass::float_e2m1_t;
using GrpElementA     = cutlass::nv_float4_t<GrpElementInput>;
using GrpLayoutATag   = cutlass::layout::RowMajor;
constexpr int GrpAlignA = 32;

using GrpElementB     = cutlass::nv_float4_t<GrpElementInput>;
using GrpLayoutBTag   = cutlass::layout::ColumnMajor;
constexpr int GrpAlignB = 32;

using GrpElementD     = cutlass::half_t;
using GrpElementC     = cutlass::half_t;
using GrpLayoutCTag   = cutlass::layout::RowMajor;
using GrpLayoutDTag   = cutlass::layout::RowMajor;
constexpr int GrpAlignC = 128 / cutlass::sizeof_bits<GrpElementC>::value;
constexpr int GrpAlignD = 128 / cutlass::sizeof_bits<GrpElementD>::value;

using GrpElementAccum = float;
using GrpElementCompute = float;
using GrpArchTag      = cutlass::arch::Sm120;
using GrpOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using GrpTileShape    = Shape<_128, _128, _128>;
using GrpClusterShape = Shape<_1, _1, _1>;

using GrpCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    GrpArchTag, GrpOperatorClass,
    GrpTileShape, GrpClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    GrpElementAccum, GrpElementCompute,
    GrpElementC, GrpLayoutCTag *, GrpAlignC,
    GrpElementD, GrpLayoutDTag *, GrpAlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using GrpCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    GrpArchTag, GrpOperatorClass,
    GrpElementA, GrpLayoutATag *, GrpAlignA,
    GrpElementB, GrpLayoutBTag *, GrpAlignB,
    GrpElementAccum,
    GrpTileShape, GrpClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename GrpCollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GrpGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    GrpProblemShape,
    GrpCollectiveMainloop,
    GrpCollectiveEpilogue
>;

using GrpGemm = cutlass::gemm::device::GemmUniversalAdapter<GrpGemmKernel>;

using GrpStrideA   = typename GrpGemm::GemmKernel::InternalStrideA;
using GrpStrideB   = typename GrpGemm::GemmKernel::InternalStrideB;
using GrpStrideC   = typename GrpGemm::GemmKernel::InternalStrideC;
using GrpStrideD   = typename GrpGemm::GemmKernel::InternalStrideD;
using GrpLayoutSFA = typename GrpGemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using GrpLayoutSFB = typename GrpGemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using GrpSm1xxBlkScaledConfig = typename GrpGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using GrpElementSF = typename GrpGemm::GemmKernel::CollectiveMainloop::ElementSF;

using GrpUnderlyingShape = typename GrpProblemShape::UnderlyingProblemShape;

namespace imp {

static int s_grp3x_available = -1;

bool cutlass_grouped_3x_nvfp4_available() {
    if (s_grp3x_available >= 0) return s_grp3x_available;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    s_grp3x_available = (prop.major * 10 + prop.minor >= 120) ? 1 : 0;
    return s_grp3x_available;
}

// Persistent staging + workspace (grow-only, process lifetime).
static void*  s_staging    = nullptr;
static size_t s_staging_sz = 0;
static void*  s_workspace  = nullptr;
static size_t s_workspace_sz = 0;

static size_t align128(size_t x) { return (x + 127) & ~size_t(127); }

static void ensure_staging(size_t need) {
    if (need <= s_staging_sz) return;
    if (s_staging) IMP_CUDA_CHECK_LOG(cudaFree(s_staging));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&s_staging, need));
    s_staging_sz = need;
}

static void ensure_workspace(size_t need) {
    if (need <= s_workspace_sz) return;
    if (s_workspace) IMP_CUDA_CHECK_LOG(cudaFree(s_workspace));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&s_workspace, need));
    s_workspace_sz = need;
}

bool gemm_grouped_cutlass_3x_nvfp4(
    int n_experts,
    const int* host_M,
    int N, int K,
    const void* const* host_ptr_A,
    const void* const* host_ptr_SFA,
    const void* const* host_ptr_B,
    const void* const* host_ptr_SFB,
    void*       const* host_ptr_D,
    const float*       host_alpha,
    cudaStream_t stream)
{
    if (n_experts <= 0) return true;

    // Flush sticky CUDA errors so CUTLASS TMA setup does not see stale state.
    {
        cudaError_t prior = cudaGetLastError();
        if (prior != cudaSuccess) {
            IMP_LOG_ERROR("CUTLASS 3x grouped: prior CUDA error: %s", cudaGetErrorString(prior));
            return false;
        }
    }

    using ElemA  = typename GrpGemm::ElementA;
    using ElemB  = typename GrpGemm::ElementB;
    using ElemC  = typename GrpGemm::ElementC;
    using ElemD  = typename GrpGemm::EpilogueOutputOp::ElementOutput;

    // Single-allocation device staging: all per-expert arrays packed with 128B
    // alignment between sections. A matching host buffer is built, then copied
    // in ONE cudaMemcpyAsync (instead of 15 separate calls — saves launch overhead
    // on every prefill chunk).
    const size_t n = static_cast<size_t>(n_experts);
    struct Offs {
        size_t shape, stA, stB, stC, stD, lSFA, lSFB;
        size_t ptrA, ptrB, ptrSFA, ptrSFB, ptrC, ptrD;
        size_t alpha, aPtr, total;
    } o;
    o.shape  = 0;
    o.stA    = align128(o.shape  + n * sizeof(GrpUnderlyingShape));
    o.stB    = align128(o.stA    + n * sizeof(GrpStrideA));
    o.stC    = align128(o.stB    + n * sizeof(GrpStrideB));
    o.stD    = align128(o.stC    + n * sizeof(GrpStrideC));
    o.lSFA   = align128(o.stD    + n * sizeof(GrpStrideD));
    o.lSFB   = align128(o.lSFA   + n * sizeof(GrpLayoutSFA));
    o.ptrA   = align128(o.lSFB   + n * sizeof(GrpLayoutSFB));
    o.ptrB   = align128(o.ptrA   + n * sizeof(void*));
    o.ptrSFA = align128(o.ptrB   + n * sizeof(void*));
    o.ptrSFB = align128(o.ptrSFA + n * sizeof(void*));
    o.ptrC   = align128(o.ptrSFB + n * sizeof(void*));
    o.ptrD   = align128(o.ptrC   + n * sizeof(void*));
    o.alpha  = align128(o.ptrD   + n * sizeof(void*));
    o.aPtr   = align128(o.alpha  + n * sizeof(float));
    o.total  = align128(o.aPtr   + n * sizeof(float*));

    ensure_staging(o.total);
    char* d_base = static_cast<char*>(s_staging);

    // Fill the matching host buffer in-place (avoids many small std::vectors).
    std::vector<char> host_buf(o.total);
    char* h_base = host_buf.data();
    auto at = [&](size_t off) { return h_base + off; };

    // Per-expert strides + layouts + shapes + pointers.
    for (int i = 0; i < n_experts; ++i) {
        int M_i = host_M[i];
        reinterpret_cast<GrpUnderlyingShape*>(at(o.shape))[i] = {M_i, N, K};
        reinterpret_cast<GrpStrideA*>(at(o.stA))[i] = cutlass::make_cute_packed_stride(GrpStrideA{}, {M_i, K, 1});
        reinterpret_cast<GrpStrideB*>(at(o.stB))[i] = cutlass::make_cute_packed_stride(GrpStrideB{}, {N,   K, 1});
        reinterpret_cast<GrpStrideC*>(at(o.stC))[i] = cutlass::make_cute_packed_stride(GrpStrideC{}, {M_i, N, 1});
        reinterpret_cast<GrpStrideD*>(at(o.stD))[i] = cutlass::make_cute_packed_stride(GrpStrideD{}, {M_i, N, 1});
        reinterpret_cast<GrpLayoutSFA*>(at(o.lSFA))[i] = GrpSm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M_i, N, K, 1));
        reinterpret_cast<GrpLayoutSFB*>(at(o.lSFB))[i] = GrpSm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M_i, N, K, 1));
        reinterpret_cast<const ElemA**>(at(o.ptrA))[i]     = reinterpret_cast<const ElemA*>(host_ptr_A[i]);
        reinterpret_cast<const ElemB**>(at(o.ptrB))[i]     = reinterpret_cast<const ElemB*>(host_ptr_B[i]);
        reinterpret_cast<const GrpElementSF**>(at(o.ptrSFA))[i] = reinterpret_cast<const GrpElementSF*>(host_ptr_SFA[i]);
        reinterpret_cast<const GrpElementSF**>(at(o.ptrSFB))[i] = reinterpret_cast<const GrpElementSF*>(host_ptr_SFB[i]);
        // beta=0 but CUTLASS still TMA-maps C — reuse D pointer (never read).
        reinterpret_cast<const ElemC**>(at(o.ptrC))[i] = reinterpret_cast<const ElemC*>(host_ptr_D[i]);
        reinterpret_cast<ElemD**>(at(o.ptrD))[i]       = reinterpret_cast<ElemD*>(host_ptr_D[i]);
    }

    // Alpha block + per-group alpha pointer array (points into device alpha block).
    std::memcpy(at(o.alpha), host_alpha, n * sizeof(float));
    float* d_alpha_block = reinterpret_cast<float*>(d_base + o.alpha);
    for (int i = 0; i < n_experts; ++i) {
        reinterpret_cast<float**>(at(o.aPtr))[i] = d_alpha_block + i;
    }

    // Single memcpy for the whole staging region.
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_base, h_base, o.total, cudaMemcpyHostToDevice, stream));

    auto d_shapes   = reinterpret_cast<GrpUnderlyingShape*>(d_base + o.shape);
    auto d_stA      = reinterpret_cast<GrpStrideA*>(d_base + o.stA);
    auto d_stB      = reinterpret_cast<GrpStrideB*>(d_base + o.stB);
    auto d_stC      = reinterpret_cast<GrpStrideC*>(d_base + o.stC);
    auto d_stD      = reinterpret_cast<GrpStrideD*>(d_base + o.stD);
    auto d_lSFA     = reinterpret_cast<GrpLayoutSFA*>(d_base + o.lSFA);
    auto d_lSFB     = reinterpret_cast<GrpLayoutSFB*>(d_base + o.lSFB);
    auto d_ptrA     = reinterpret_cast<const ElemA**>(d_base + o.ptrA);
    auto d_ptrB     = reinterpret_cast<const ElemB**>(d_base + o.ptrB);
    auto d_ptrSFA   = reinterpret_cast<const GrpElementSF**>(d_base + o.ptrSFA);
    auto d_ptrSFB   = reinterpret_cast<const GrpElementSF**>(d_base + o.ptrSFB);
    auto d_ptrC     = reinterpret_cast<const ElemC**>(d_base + o.ptrC);
    auto d_ptrD     = reinterpret_cast<ElemD**>(d_base + o.ptrD);
    auto d_aPtrArr  = reinterpret_cast<float**>(d_base + o.aPtr);

    // Host problem-shape array for CUTLASS can_implement validation.
    // Lives inside host_buf; its lifetime extends through the call below.
    GrpUnderlyingShape* h_shapes = reinterpret_cast<GrpUnderlyingShape*>(at(o.shape));

    // ----- Build Arguments -----
    typename GrpGemm::Arguments arguments;
    {
        // Epilogue fusion args: per-group alpha via pointer array.
        decltype(arguments.epilogue.thread) fusion_args;
        fusion_args.alpha = 0.f;
        fusion_args.beta  = 0.f;
        fusion_args.alpha_ptr = nullptr;
        fusion_args.beta_ptr  = nullptr;
        fusion_args.alpha_ptr_array = d_aPtrArr;          // per-group alpha
        fusion_args.beta_ptr_array  = nullptr;
        fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1}; // one alpha per group
        fusion_args.dBeta  = {cute::_0{}, cute::_0{}, 0};

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

        typename GrpGemm::GemmKernel::TileSchedulerArguments scheduler;

        arguments = typename GrpGemm::Arguments {
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {n_experts, d_shapes, h_shapes},
            {d_ptrA, d_stA, d_ptrB, d_stB, d_ptrSFA, d_lSFA, d_ptrSFB, d_lSFB},
            {fusion_args, d_ptrC, d_stC, d_ptrD, d_stD},
            hw_info, scheduler
        };
    }

    GrpGemm gemm;
    cutlass::Status st = gemm.can_implement(arguments);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_WARN("CUTLASS 3x grouped NVFP4: can_implement failed (%d) ne=%d N=%d K=%d",
                     (int)st, n_experts, N, K);
        return false;
    }

    size_t needed = GrpGemm::get_workspace_size(arguments);
    ensure_workspace(needed);

    st = gemm.initialize(arguments, s_workspace, stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS 3x grouped NVFP4: initialize failed (%d)", (int)st);
        return false;
    }

    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS 3x grouped NVFP4: run failed (%d)", (int)st);
        return false;
    }

    return true;
}

void gemm_grouped_3x_nvfp4_cleanup() {
    if (s_staging)   { IMP_CUDA_CHECK_LOG(cudaFree(s_staging));   s_staging   = nullptr; s_staging_sz   = 0; }
    if (s_workspace) { IMP_CUDA_CHECK_LOG(cudaFree(s_workspace)); s_workspace = nullptr; s_workspace_sz = 0; }
}

// Compile-time verification: kernel type instantiates on SM120.
static_assert(sizeof(GrpGemm) > 0, "GrpGemm type must instantiate");

} // namespace imp
