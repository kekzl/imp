// CUTLASS 3.x NVFP4 BlockScaled Grouped GEMM for MoE (SM120).
// Based on Example 79d: Blackwell GeForce NVFP4 Grouped GEMM.
//
// Zero D2H sync during execution — per-expert device pointer arrays are
// pre-built once per call on the host (no problem-shape lookup on GPU),
// which CUTLASS then indexes directly for its GroupProblemShape scheduler.

#include "compute/gemm_cutlass_grouped_3x.h"
#include "core/cuda_static_reset.h"
#include "core/logging.h"
#include "memory/engine_arena.h"

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

using GrpProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

using GrpElementInput = cutlass::float_e2m1_t;
using GrpElementA = cutlass::nv_float4_t<GrpElementInput>;
using GrpLayoutATag = cutlass::layout::RowMajor;
constexpr int GrpAlignA = 32;

using GrpElementB = cutlass::nv_float4_t<GrpElementInput>;
using GrpLayoutBTag = cutlass::layout::ColumnMajor;
constexpr int GrpAlignB = 32;

using GrpElementD = cutlass::half_t;
using GrpElementC = cutlass::half_t;
using GrpLayoutCTag = cutlass::layout::RowMajor;
using GrpLayoutDTag = cutlass::layout::RowMajor;
constexpr int GrpAlignC = 128 / cutlass::sizeof_bits<GrpElementC>::value;
constexpr int GrpAlignD = 128 / cutlass::sizeof_bits<GrpElementD>::value;

using GrpElementAccum = float;
using GrpElementCompute = float;
using GrpArchTag = cutlass::arch::Sm120;
using GrpOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using GrpTileShape = Shape<_128, _128, _128>;
using GrpClusterShape = Shape<_1, _1, _1>;

using GrpCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    GrpArchTag, GrpOperatorClass, GrpTileShape, GrpClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, GrpElementAccum, GrpElementCompute, GrpElementC,
    GrpLayoutCTag*, GrpAlignC, GrpElementD, GrpLayoutDTag*, GrpAlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using GrpCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    GrpArchTag, GrpOperatorClass, GrpElementA, GrpLayoutATag*, GrpAlignA, GrpElementB, GrpLayoutBTag*,
    GrpAlignB, GrpElementAccum, GrpTileShape, GrpClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename GrpCollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GrpGemmKernel =
    cutlass::gemm::kernel::GemmUniversal<GrpProblemShape, GrpCollectiveMainloop, GrpCollectiveEpilogue>;

using GrpGemm = cutlass::gemm::device::GemmUniversalAdapter<GrpGemmKernel>;

using GrpStrideA = typename GrpGemm::GemmKernel::InternalStrideA;
using GrpStrideB = typename GrpGemm::GemmKernel::InternalStrideB;
using GrpStrideC = typename GrpGemm::GemmKernel::InternalStrideC;
using GrpStrideD = typename GrpGemm::GemmKernel::InternalStrideD;
using GrpLayoutSFA = typename GrpGemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using GrpLayoutSFB = typename GrpGemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using GrpSm1xxBlkScaledConfig = typename GrpGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using GrpElementSF = typename GrpGemm::GemmKernel::CollectiveMainloop::ElementSF;

using GrpUnderlyingShape = typename GrpProblemShape::UnderlyingProblemShape;

namespace imp {

static int s_grp3x_available = -1;

bool cutlass_grouped_3x_nvfp4_available() {
    if (s_grp3x_available >= 0)
        return s_grp3x_available;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    s_grp3x_available = (prop.major * 10 + prop.minor >= 120) ? 1 : 0;
    return s_grp3x_available;
}

// Staging + workspace, both engine-persistent (T2) since A7 step 8.
static void* s_staging = nullptr;
static size_t s_staging_sz = 0;
static void* s_workspace = nullptr;
static size_t s_workspace_sz = 0;

// Persistent CUTLASS adapter — first call uses initialize() to do the
// (sticky) cudaFuncSetAttribute(MaxDynamicSharedMemorySize) + workspace init;
// every subsequent call uses the lightweight update() path which only
// recomputes params_ from new args (per-group pointers/strides), skipping
// the CUDA driver roundtrip and re-init kernels. Plus a (N,K)-keyed
// can_implement memo — alignment checks don't depend on per-group M.
static GrpGemm* s_gemm = nullptr;
static bool s_gemm_initialized = false;
static int s_can_impl_N = -1;
static int s_can_impl_K = -1;

static size_t align128(size_t x) { return (x + 127) & ~size_t(127); }

// Both buffers come from the engine-persistent (T2) arena (A7 step 8). The
// growth path stays — unlike the cudaMalloc it replaces, a bump-arena take is
// pointer arithmetic and not a CUDA call, so it is legal under stream capture,
// which is the property the 512 MiB pre-reservation existed to buy. Growing
// still strands the previous slab (a bump arena has no free), so the prewarm
// takes the measured requirement up front and growth should never happen.
static void* take_t2(size_t need, const char* what) {
    auto slab = engine_arena().take_bytes(need);
    if (slab.empty()) {
        IMP_LOG_ERROR(
            "CUTLASS 3x grouped: %s (%.2f MiB) unavailable from the T2 arena "
            "(%.1f MiB free of %.1f MiB) — the grouped MoE GEMM will fail",
            what, need / (1024.0 * 1024.0), engine_arena().remaining() / (1024.0 * 1024.0),
            engine_arena().capacity() / (1024.0 * 1024.0));
        return nullptr;
    }
    return slab.data();
}

static void ensure_staging(size_t need) {
    if (need <= s_staging_sz)
        return;
    if (void* p = take_t2(need, "staging")) {
        s_staging = p;
        s_staging_sz = need;
    }
}

static void ensure_workspace(size_t need) {
    if (need <= s_workspace_sz)
        return;
    if (s_workspace_sz > 0) {
        // The prewarm's measured reserve went stale — a CUTLASS bump, or a
        // shape class the 2026-07-31 sweep did not cover. Recoverable (the
        // take below serves it out of the arena's slack) but it strands the
        // old slab, so it must not pass silently.
        IMP_LOG_WARN(
            "CUTLASS 3x grouped: workspace grew %zu -> %zu B past the measured reserve "
            "— re-measure kGrouped3xWorkspaceBytes",
            s_workspace_sz, need);
    }
    if (void* p = take_t2(need, "GEMM workspace")) {
        s_workspace = p;
        s_workspace_sz = need;
    }
}

void gemm_grouped_3x_nvfp4_prewarm() {
    if (!cutlass_grouped_3x_nvfp4_available()) return;
    // Staging: the per-group struct-of-arrays block, ~200 B per expert against
    // a hard n_experts <= 256 — 1 MiB is 20x the worst case.
    //
    // Workspace: MEASURED, not guessed. This used to reserve 512 MiB "to cover
    // CUTLASS scratch even for very large grouped problems"; instrumenting
    // GrpGemm::get_workspace_size() across three MoE geometries
    // (ne=128 N=768 K=2048, ne=256 N=512 K=2048, ne=128 N=1856 K=2688) and
    // prefills from 130 to 2800 tokens returned the same 152 320 B every time
    // — 170 SMs x 896 B of persistent-scheduler state, which is a property of
    // the CHIP and not of the problem. The old number was 3500x the real one
    // and it was resident for the life of every MoE process (AUDIT B73).
    // 1 MiB keeps ~7x headroom for a CUTLASS bump; more than that and
    // ensure_workspace() grows into the arena's alignment slack and says so.
    ensure_staging(kGrouped3xStagingBytes);
    ensure_workspace(kGrouped3xWorkspaceBytes);
}

bool gemm_grouped_cutlass_3x_nvfp4(int n_experts, const int* host_M, int N, int K,
                                   const void* const* host_ptr_A, const void* const* host_ptr_SFA,
                                   const void* const* host_ptr_B, const void* const* host_ptr_SFB,
                                   void* const* host_ptr_D, const float* host_alpha, cudaStream_t stream) {
    if (n_experts <= 0)
        return true;

    // Flush sticky CUDA errors so CUTLASS TMA setup does not see stale state.
    {
        cudaError_t prior = cudaGetLastError();
        if (prior != cudaSuccess) {
            IMP_LOG_ERROR("CUTLASS 3x grouped: prior CUDA error: %s", cudaGetErrorString(prior));
            return false;
        }
    }

    using ElemA = typename GrpGemm::ElementA;
    using ElemB = typename GrpGemm::ElementB;
    using ElemC = typename GrpGemm::ElementC;
    using ElemD = typename GrpGemm::EpilogueOutputOp::ElementOutput;

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
    o.shape = 0;
    o.stA = align128(o.shape + n * sizeof(GrpUnderlyingShape));
    o.stB = align128(o.stA + n * sizeof(GrpStrideA));
    o.stC = align128(o.stB + n * sizeof(GrpStrideB));
    o.stD = align128(o.stC + n * sizeof(GrpStrideC));
    o.lSFA = align128(o.stD + n * sizeof(GrpStrideD));
    o.lSFB = align128(o.lSFA + n * sizeof(GrpLayoutSFA));
    o.ptrA = align128(o.lSFB + n * sizeof(GrpLayoutSFB));
    o.ptrB = align128(o.ptrA + n * sizeof(void*));
    o.ptrSFA = align128(o.ptrB + n * sizeof(void*));
    o.ptrSFB = align128(o.ptrSFA + n * sizeof(void*));
    o.ptrC = align128(o.ptrSFB + n * sizeof(void*));
    o.ptrD = align128(o.ptrC + n * sizeof(void*));
    o.alpha = align128(o.ptrD + n * sizeof(void*));
    o.aPtr = align128(o.alpha + n * sizeof(float));
    o.total = align128(o.aPtr + n * sizeof(float*));

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
        reinterpret_cast<GrpStrideA*>(at(o.stA))[i] = cutlass::make_cute_packed_stride(GrpStrideA{},
                                                                                       {M_i, K, 1});
        reinterpret_cast<GrpStrideB*>(at(o.stB))[i] = cutlass::make_cute_packed_stride(GrpStrideB{},
                                                                                       {N, K, 1});
        reinterpret_cast<GrpStrideC*>(at(o.stC))[i] = cutlass::make_cute_packed_stride(GrpStrideC{},
                                                                                       {M_i, N, 1});
        reinterpret_cast<GrpStrideD*>(at(o.stD))[i] = cutlass::make_cute_packed_stride(GrpStrideD{},
                                                                                       {M_i, N, 1});
        reinterpret_cast<GrpLayoutSFA*>(at(o.lSFA))[i] = GrpSm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
            cute::make_shape(M_i, N, K, 1));
        reinterpret_cast<GrpLayoutSFB*>(at(o.lSFB))[i] = GrpSm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
            cute::make_shape(M_i, N, K, 1));
        reinterpret_cast<const ElemA**>(at(o.ptrA))[i] = reinterpret_cast<const ElemA*>(host_ptr_A[i]);
        reinterpret_cast<const ElemB**>(at(o.ptrB))[i] = reinterpret_cast<const ElemB*>(host_ptr_B[i]);
        reinterpret_cast<const GrpElementSF**>(at(o.ptrSFA))[i] = reinterpret_cast<const GrpElementSF*>(
            host_ptr_SFA[i]);
        reinterpret_cast<const GrpElementSF**>(at(o.ptrSFB))[i] = reinterpret_cast<const GrpElementSF*>(
            host_ptr_SFB[i]);
        // beta=0 but CUTLASS still TMA-maps C — reuse D pointer (never read).
        reinterpret_cast<const ElemC**>(at(o.ptrC))[i] = reinterpret_cast<const ElemC*>(host_ptr_D[i]);
        reinterpret_cast<ElemD**>(at(o.ptrD))[i] = reinterpret_cast<ElemD*>(host_ptr_D[i]);
    }

    // Alpha block + per-group alpha pointer array (points into device alpha block).
    std::memcpy(at(o.alpha), host_alpha, n * sizeof(float));
    float* d_alpha_block = reinterpret_cast<float*>(d_base + o.alpha);
    for (int i = 0; i < n_experts; ++i) {
        reinterpret_cast<float**>(at(o.aPtr))[i] = d_alpha_block + i;
    }

    // Single memcpy for the whole staging region.
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_base, h_base, o.total, cudaMemcpyHostToDevice, stream));

    auto d_shapes = reinterpret_cast<GrpUnderlyingShape*>(d_base + o.shape);
    auto d_stA = reinterpret_cast<GrpStrideA*>(d_base + o.stA);
    auto d_stB = reinterpret_cast<GrpStrideB*>(d_base + o.stB);
    auto d_stC = reinterpret_cast<GrpStrideC*>(d_base + o.stC);
    auto d_stD = reinterpret_cast<GrpStrideD*>(d_base + o.stD);
    auto d_lSFA = reinterpret_cast<GrpLayoutSFA*>(d_base + o.lSFA);
    auto d_lSFB = reinterpret_cast<GrpLayoutSFB*>(d_base + o.lSFB);
    auto d_ptrA = reinterpret_cast<const ElemA**>(d_base + o.ptrA);
    auto d_ptrB = reinterpret_cast<const ElemB**>(d_base + o.ptrB);
    auto d_ptrSFA = reinterpret_cast<const GrpElementSF**>(d_base + o.ptrSFA);
    auto d_ptrSFB = reinterpret_cast<const GrpElementSF**>(d_base + o.ptrSFB);
    auto d_ptrC = reinterpret_cast<const ElemC**>(d_base + o.ptrC);
    auto d_ptrD = reinterpret_cast<ElemD**>(d_base + o.ptrD);
    auto d_aPtrArr = reinterpret_cast<float**>(d_base + o.aPtr);

    // Host problem-shape array for CUTLASS can_implement validation.
    // Lives inside host_buf; its lifetime extends through the call below.
    GrpUnderlyingShape* h_shapes = reinterpret_cast<GrpUnderlyingShape*>(at(o.shape));

    // ----- Build Arguments -----
    typename GrpGemm::Arguments arguments;
    {
        // Epilogue fusion args: per-group alpha via pointer array.
        decltype(arguments.epilogue.thread) fusion_args;
        fusion_args.alpha = 0.f;
        fusion_args.beta = 0.f;
        fusion_args.alpha_ptr = nullptr;
        fusion_args.beta_ptr = nullptr;
        fusion_args.alpha_ptr_array = d_aPtrArr;  // per-group alpha
        fusion_args.beta_ptr_array = nullptr;
        fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};  // one alpha per group
        fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

        typename GrpGemm::GemmKernel::TileSchedulerArguments scheduler;

        arguments = typename GrpGemm::Arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                                {n_experts, d_shapes, h_shapes},
                                                {d_ptrA, d_stA, d_ptrB, d_stB, d_ptrSFA, d_lSFA, d_ptrSFB,
                                                 d_lSFB},
                                                {fusion_args, d_ptrC, d_stC, d_ptrD, d_stD},
                                                hw_info,
                                                scheduler};
    }

    if (s_gemm == nullptr) {
        s_gemm = new GrpGemm();
    }

    // can_implement only validates host-side alignment, which depends on N, K,
    // and the static layout/element types — NOT on per-group M values. Memoize
    // the result so we pay it once per (N,K) seen in this process.
    if (N != s_can_impl_N || K != s_can_impl_K) {
        cutlass::Status st = s_gemm->can_implement(arguments);
        if (st != cutlass::Status::kSuccess) {
            IMP_LOG_WARN("CUTLASS 3x grouped NVFP4: can_implement failed (%d) ne=%d N=%d K=%d",
                         (int)st, n_experts, N, K);
            return false;
        }
        s_can_impl_N = N;
        s_can_impl_K = K;
    }

    size_t needed = GrpGemm::get_workspace_size(arguments);
    ensure_workspace(needed);

    // First call: full initialize() — sticky cudaFuncSetAttribute on the kernel
    //             function symbol + (no-op) workspace init + params_ build.
    // Subsequent: update() — just rebuilds params_ from new args. Skips the
    //             CUDA driver roundtrip, which is the per-call CPU cost.
    cutlass::Status st;
    if (!s_gemm_initialized) {
        st = s_gemm->initialize(arguments, s_workspace, stream);
        if (st != cutlass::Status::kSuccess) {
            IMP_LOG_ERROR("CUTLASS 3x grouped NVFP4: initialize failed (%d)", (int)st);
            return false;
        }
        s_gemm_initialized = true;
    } else {
        st = s_gemm->update(arguments, s_workspace);
        if (st != cutlass::Status::kSuccess) {
            IMP_LOG_ERROR("CUTLASS 3x grouped NVFP4: update failed (%d)", (int)st);
            return false;
        }
    }

    st = s_gemm->run(stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS 3x grouped NVFP4: run failed (%d)", (int)st);
        return false;
    }

    return true;
}

void gemm_grouped_3x_nvfp4_cleanup() {
    // Arena-owned since A7 step 8 — the T2 arena owns the region and ~Engine
    // closes it. Re-arm the guards so the next prewarm takes a fresh slice.
    s_staging = nullptr;
    s_staging_sz = 0;
    s_workspace = nullptr;
    s_workspace_sz = 0;
    if (s_gemm) {
        delete s_gemm;
        s_gemm = nullptr;
    }
    s_gemm_initialized = false;
    s_can_impl_N = -1;
    s_can_impl_K = -1;
}

// Registered as a pre-cudaDeviceReset hook (#1207); see core/cuda_static_reset.h.
namespace {
IMP_REGISTER_CUDA_STATIC_RESET(gemm_grouped_3x_nvfp4_cleanup);
}  // namespace

// Compile-time verification: kernel type instantiates on SM120.
static_assert(sizeof(GrpGemm) > 0, "GrpGemm type must instantiate");

}  // namespace imp

// ===========================================================================
// Phase 3b: device-args wrapper. Same kernel call (s_gemm->run) as the host-
// args variant, but the staging buffer is filled by a device kernel reading
// d_M_per / d_expert_offsets / d_sfa_offsets / d_alpha — no host loop, no
// D2H or H2D copies on the dispatch path. Graph-capturable.
// ===========================================================================

namespace imp {

using DeviceArgsElemA = typename GrpGemm::ElementA;
using DeviceArgsElemB = typename GrpGemm::ElementB;
using DeviceArgsElemC = typename GrpGemm::ElementC;
using DeviceArgsElemD = typename GrpGemm::EpilogueOutputOp::ElementOutput;

}  // namespace imp

// One thread per expert. n_experts <= 256, single block with 256 threads.
__global__ void build_grouped_3x_staging_kernel(
    // Staging-buffer outputs (typed pointers; host wrapper computes offsets).
    GrpUnderlyingShape* __restrict__ shapes,
    GrpStrideA*  __restrict__ stA,
    GrpStrideB*  __restrict__ stB,
    GrpStrideC*  __restrict__ stC,
    GrpStrideD*  __restrict__ stD,
    GrpLayoutSFA* __restrict__ lSFA,
    GrpLayoutSFB* __restrict__ lSFB,
    const imp::DeviceArgsElemA**  __restrict__ ptrA,
    const imp::DeviceArgsElemB**  __restrict__ ptrB,
    const GrpElementSF**          __restrict__ ptrSFA,
    const GrpElementSF**          __restrict__ ptrSFB,
    const imp::DeviceArgsElemC**  __restrict__ ptrC,
    imp::DeviceArgsElemD**        __restrict__ ptrD,
    float* __restrict__ alpha_block,
    float** __restrict__ aPtr,
    // Device-resident inputs.
    const int32_t* __restrict__ d_M_per,
    const int32_t* __restrict__ d_expert_offsets,
    const int64_t* __restrict__ d_sfa_offsets,
    const float*   __restrict__ d_alpha,
    // Base pointers + per-expert strides (host scalars, passed by value).
    const void* base_A_packed,
    const void* base_A_sf,
    const void* base_B_packed,
    int64_t     b_expert_stride_packed,
    const void* base_B_sf,
    int64_t     b_expert_stride_sf,
    // Optional per-expert B/SFB pointer arrays (mode (b)). When non-null,
    // these override base_B_*/b_expert_stride_* — used by registry-handle
    // MoE weight layout (CUTLASS 3.x prefill in executor_forward_moe.cu).
    const void* const* d_B_ptrs,
    const void* const* d_SFB_ptrs,
    void*       base_D,
    // Shared shape dims.
    int N, int K, int n_experts)
{
    int e = threadIdx.x;
    if (e >= n_experts) return;

    int     M_e          = d_M_per[e];
    int64_t row_offset_e = static_cast<int64_t>(d_expert_offsets[e]);
    int64_t sfa_offset_e = d_sfa_offsets[e];

    // 1. Per-expert problem shape {M_e, N, K}.
    shapes[e] = GrpUnderlyingShape{M_e, N, K};

    // 2. Packed strides via CUTLASS helper (CUTLASS_HOST_DEVICE).
    //    The relevant Stride types ignore M_i in the body and only set
    //    get<0|1>(stride) = get<0|1>(shape), so the result is constant
    //    across experts in practice. Writing them inline keeps the kernel
    //    self-contained (no separate pre-bake step).
    stA[e] = cutlass::make_cute_packed_stride(GrpStrideA{}, {M_e, K, 1});
    stB[e] = cutlass::make_cute_packed_stride(GrpStrideB{}, {N,   K, 1});
    stC[e] = cutlass::make_cute_packed_stride(GrpStrideC{}, {M_e, N, 1});
    stD[e] = cutlass::make_cute_packed_stride(GrpStrideD{}, {M_e, N, 1});

    // 3. SFA / SFB CUTE layouts via SfAtom helper (CUTE_HOST_DEVICE).
    lSFA[e] = GrpSm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M_e, N, K, 1));
    lSFB[e] = GrpSm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M_e, N, K, 1));

    // 4. Per-expert pointer arrays.
    //   A activations: contiguous, K/2 bytes per row, offset_e rows.
    int64_t a_row_bytes = static_cast<int64_t>(K) / 2;
    ptrA[e] = reinterpret_cast<const imp::DeviceArgsElemA*>(
        static_cast<const char*>(base_A_packed) + row_offset_e * a_row_bytes);
    //   SFA: SfAtom-padded slab, byte offset from prefix sum.
    ptrSFA[e] = reinterpret_cast<const GrpElementSF*>(
        static_cast<const char*>(base_A_sf) + sfa_offset_e);
    //   B weights: mode (b) per-expert pointer array if provided, else mode
    //   (a) base + per-expert byte stride.
    if (d_B_ptrs != nullptr) {
        ptrB[e]   = reinterpret_cast<const imp::DeviceArgsElemB*>(d_B_ptrs[e]);
        ptrSFB[e] = reinterpret_cast<const GrpElementSF*>(d_SFB_ptrs[e]);
    } else {
        ptrB[e]   = reinterpret_cast<const imp::DeviceArgsElemB*>(
            static_cast<const char*>(base_B_packed) + static_cast<int64_t>(e) * b_expert_stride_packed);
        ptrSFB[e] = reinterpret_cast<const GrpElementSF*>(
            static_cast<const char*>(base_B_sf) + static_cast<int64_t>(e) * b_expert_stride_sf);
    }
    //   C/D outputs alias the FP16 result buffer (beta=0).
    void* dst_e =
        static_cast<char*>(base_D) +
        row_offset_e * static_cast<int64_t>(N) * static_cast<int64_t>(sizeof(half));
    ptrC[e] = reinterpret_cast<const imp::DeviceArgsElemC*>(dst_e);
    ptrD[e] = reinterpret_cast<imp::DeviceArgsElemD*>(dst_e);

    // 5. Per-expert alpha value + alpha-pointer-array slot.
    alpha_block[e] = d_alpha[e];
    aPtr[e] = alpha_block + e;
}

namespace imp {

// Persistent host shapes (max-M dummy) for can_implement validation.
// can_implement only checks alignment (which depends on N, K and element
// types — not on per-group M values), so any safely-aligned M works.
static std::vector<GrpUnderlyingShape>* s_host_shapes_max = nullptr;

bool gemm_grouped_cutlass_3x_nvfp4_device_args(
    int n_experts, int N, int K,
    const GroupedNvfp4DeviceArgs& args,
    cudaStream_t stream) {
    if (n_experts <= 0)
        return true;
    if (n_experts > 256) {
        IMP_LOG_ERROR("CUTLASS 3x grouped device-args: n_experts=%d > 256 not supported",
                      n_experts);
        return false;
    }

    // Flush sticky CUDA errors so CUTLASS TMA setup does not see stale state.
    {
        cudaError_t prior = cudaGetLastError();
        if (prior != cudaSuccess) {
            IMP_LOG_ERROR("CUTLASS 3x grouped device-args: prior CUDA error: %s",
                          cudaGetErrorString(prior));
            return false;
        }
    }

    // Same staging-buffer layout as the host-args variant — re-use s_staging.
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

    auto d_shapes = reinterpret_cast<GrpUnderlyingShape*>(d_base + o.shape);
    auto d_stA    = reinterpret_cast<GrpStrideA*>(d_base + o.stA);
    auto d_stB    = reinterpret_cast<GrpStrideB*>(d_base + o.stB);
    auto d_stC    = reinterpret_cast<GrpStrideC*>(d_base + o.stC);
    auto d_stD    = reinterpret_cast<GrpStrideD*>(d_base + o.stD);
    auto d_lSFA   = reinterpret_cast<GrpLayoutSFA*>(d_base + o.lSFA);
    auto d_lSFB   = reinterpret_cast<GrpLayoutSFB*>(d_base + o.lSFB);
    auto d_ptrA   = reinterpret_cast<const DeviceArgsElemA**>(d_base + o.ptrA);
    auto d_ptrB   = reinterpret_cast<const DeviceArgsElemB**>(d_base + o.ptrB);
    auto d_ptrSFA = reinterpret_cast<const GrpElementSF**>(d_base + o.ptrSFA);
    auto d_ptrSFB = reinterpret_cast<const GrpElementSF**>(d_base + o.ptrSFB);
    auto d_ptrC   = reinterpret_cast<const DeviceArgsElemC**>(d_base + o.ptrC);
    auto d_ptrD   = reinterpret_cast<DeviceArgsElemD**>(d_base + o.ptrD);
    auto d_alpha_block = reinterpret_cast<float*>(d_base + o.alpha);
    auto d_aPtrArr     = reinterpret_cast<float**>(d_base + o.aPtr);

    // Build/refresh persistent host_shapes for can_implement on first (N,K).
    if (!s_host_shapes_max)
        s_host_shapes_max = new std::vector<GrpUnderlyingShape>();
    if ((int)s_host_shapes_max->size() < n_experts) {
        // Dummy alignment-safe M (multiple of 128 covers all TMA alignments
        // we use). Actual per-expert M is on device via d_M_per.
        s_host_shapes_max->assign(n_experts, GrpUnderlyingShape{128, N, K});
    } else {
        // Refresh N, K (could change across calls — alignment depends on them).
        for (int i = 0; i < n_experts; ++i) {
            (*s_host_shapes_max)[i] = GrpUnderlyingShape{128, N, K};
        }
    }
    GrpUnderlyingShape* h_shapes = s_host_shapes_max->data();

    // Launch the device-side staging-build kernel.
    build_grouped_3x_staging_kernel<<<1, 256, 0, stream>>>(
        d_shapes, d_stA, d_stB, d_stC, d_stD, d_lSFA, d_lSFB,
        d_ptrA, d_ptrB, d_ptrSFA, d_ptrSFB, d_ptrC, d_ptrD,
        d_alpha_block, d_aPtrArr,
        args.d_M_per, args.d_expert_offsets, args.d_sfa_offsets, args.d_alpha,
        args.base_A_packed, args.base_A_sf,
        args.base_B_packed, args.b_expert_stride_packed,
        args.base_B_sf,     args.b_expert_stride_sf,
        args.d_B_ptrs, args.d_SFB_ptrs,
        args.base_D,
        N, K, n_experts);
    IMP_CUDA_CHECK_LAUNCH();

    // ----- Build Arguments -----
    typename GrpGemm::Arguments arguments;
    {
        decltype(arguments.epilogue.thread) fusion_args;
        fusion_args.alpha = 0.f;
        fusion_args.beta  = 0.f;
        fusion_args.alpha_ptr = nullptr;
        fusion_args.beta_ptr  = nullptr;
        fusion_args.alpha_ptr_array = d_aPtrArr;
        fusion_args.beta_ptr_array  = nullptr;
        fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
        fusion_args.dBeta  = {cute::_0{}, cute::_0{}, 0};

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

        typename GrpGemm::GemmKernel::TileSchedulerArguments scheduler;

        arguments = typename GrpGemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {n_experts, d_shapes, h_shapes},
            {d_ptrA, d_stA, d_ptrB, d_stB, d_ptrSFA, d_lSFA, d_ptrSFB, d_lSFB},
            {fusion_args, d_ptrC, d_stC, d_ptrD, d_stD},
            hw_info,
            scheduler};
    }

    if (s_gemm == nullptr)
        s_gemm = new GrpGemm();

    // can_implement memoized per (N, K). Uses the M=128 dummy host shapes —
    // can_implement only checks alignment, which depends on N/K, not M.
    if (N != s_can_impl_N || K != s_can_impl_K) {
        cutlass::Status st = s_gemm->can_implement(arguments);
        if (st != cutlass::Status::kSuccess) {
            IMP_LOG_WARN(
                "CUTLASS 3x grouped device-args: can_implement failed (%d) ne=%d N=%d K=%d",
                (int)st, n_experts, N, K);
            return false;
        }
        s_can_impl_N = N;
        s_can_impl_K = K;
    }

    // Withhold the host shapes from initialize/run: real per-expert M lives
    // ONLY in d_shapes (device). When host shapes are present the group tile
    // scheduler sizes its grid from them — the M=128 dummy UNDERSIZED the
    // tile count, so experts with M_e above the tile height silently lost
    // their upper row tiles (MoE long-prefill corruption at n ≳ 900, where
    // hot experts cross the tile boundary; found 2026-06-11). With
    // host_problem_shapes == nullptr CUTLASS launches the fully persistent
    // grid (sm_count) and walks the device-side group shapes dynamically —
    // correct for any routing distribution.
    arguments.problem_shape.host_problem_shapes = nullptr;

    size_t needed = GrpGemm::get_workspace_size(arguments);
    ensure_workspace(needed);

    cutlass::Status st;
    if (!s_gemm_initialized) {
        st = s_gemm->initialize(arguments, s_workspace, stream);
        if (st != cutlass::Status::kSuccess) {
            IMP_LOG_ERROR("CUTLASS 3x grouped device-args: initialize failed (%d)", (int)st);
            return false;
        }
        s_gemm_initialized = true;
    } else {
        st = s_gemm->update(arguments, s_workspace);
        if (st != cutlass::Status::kSuccess) {
            IMP_LOG_ERROR("CUTLASS 3x grouped device-args: update failed (%d)", (int)st);
            return false;
        }
    }

    st = s_gemm->run(stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS 3x grouped device-args: run failed (%d)", (int)st);
        return false;
    }
    return true;
}

}  // namespace imp
