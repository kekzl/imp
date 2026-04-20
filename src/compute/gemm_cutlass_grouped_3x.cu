// CUTLASS 3.x NVFP4 BlockScaled Grouped GEMM for MoE (SM120).
// Based on Example 79d: Blackwell GeForce NVFP4 Grouped GEMM.
// Zero D2H sync — uses GroupProblemShape with device-side problem shapes.

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
// Mirrors gemm_cutlass_sm120.cu but with GroupProblemShape + PtrArray schedule
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
using GrpArchTag      = cutlass::arch::Sm120;
using GrpOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using GrpTileShape    = Shape<_128, _128, _128>;
using GrpClusterShape = Shape<_1, _1, _1>;

// Use Auto schedule — CUTLASS selects the best SM120 PtrArray schedule
using GrpCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    GrpArchTag, GrpOperatorClass,
    GrpTileShape, GrpClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    GrpElementAccum, GrpElementAccum,
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

namespace imp {

static int s_grp3x_available = -1;

bool cutlass_grouped_3x_nvfp4_available() {
    if (s_grp3x_available >= 0) return s_grp3x_available;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    s_grp3x_available = (prop.major * 10 + prop.minor >= 120) ? 1 : 0;
    return s_grp3x_available;
}

// Stub — full MoE integration requires per-expert pointer arrays for
// data, scale_factors, and activation scales. To be wired into
// executor_forward_moe.cu when NVFP4 MoE models are available for testing.
bool gemm_grouped_cutlass_3x_nvfp4(
    const void* /*a_packed*/, const void* /*a_sf*/,
    void* /*d_fp16*/,
    const int32_t* /*d_offsets*/,
    const CutlassNvFP4Weight* const* /*d_weight_ptrs*/,
    int /*K*/, int /*N*/, int /*n_experts*/,
    float /*tensor_scale*/,
    cudaStream_t /*stream*/)
{
    // TODO: Build per-expert pointer arrays on GPU and launch GrpGemm.
    // Requires: d_B_data_ptrs[ne], d_SFB_ptrs[ne], d_shapes[ne], d_D_ptrs[ne]
    // The kernel type (GrpGemm) compiles — verified via static_assert below.
    IMP_LOG_WARN("CUTLASS 3.x NVFP4 grouped GEMM: not yet wired (kernel compiles, dispatch pending)");
    return false;
}

void gemm_grouped_3x_nvfp4_cleanup() {}

// Compile-time verification: the kernel type instantiates without errors.
// This proves SM120 BlockScaled PtrArray grouped GEMM is supported.
static_assert(sizeof(GrpGemm) > 0, "GrpGemm type must instantiate");

} // namespace imp
