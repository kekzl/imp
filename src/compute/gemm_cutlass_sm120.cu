// CUTLASS sm_120 block-scaled NVFP4xNVFP4 GEMM for prefill (CUTLASS 4.4.1 Example 79a
// pattern, warp-specialized persistent kernel, mma.sync.aligned.block_scale on GeForce
// Blackwell). A and B use nv_float4_t<float_e2m1_t> with float_ue4m3_t scales in
// SfAtom interleaved layout; output D is FP16.
// Weight conversion and activation quantizers (same SfAtom bytes): nvfp4_sfatom_quant.cu.

#include "compute/gemm_cutlass_sm120.h"
#include "core/cuda_static_reset.h"
#include "core/cuda_errors.h"
#include "core/logging.h"
#include "core/device_info.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <type_traits>

#include "core/process_diag.h"

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// CUTLASS GEMM type config: NVFP4 x NVFP4 -> FP16, based on Example 79a but half_t
// output instead of bfloat16_t.

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD = cutlass::half_t;  // FP16 output
using ElementC = cutlass::half_t;  // C matrix type (unused, beta=0)
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;  // GeForce = no multicast

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag,
    AlignmentD, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                        CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Stream-K variant of the cooperative 128x128 tile (gemm.nvfp4_cutlass_streamk).
// Data-parallel tiling quantizes to waves: at M=512 the N=5120 projections are 4x40=
// 160 CTAs on 170 SMs (0.94 waves, 1 CTA/SM, no second CTA to hide latency) - this caps
// pp512 roofline efficiency below pp4096's. Stream-K gives every SM an equal share of
// MAC iterations and reduces K-split partials through the scheduler workspace; same
// mainloop/epilogue, only the tile scheduler tag differs (CUTLASS maps
// StreamKScheduler on Sm120 to the Sm100 stream-K scheduler inside the sm90
// cooperative kernel).
using GemmKernelStreamK = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
using GemmStreamK = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelStreamK>;

// FP32-output variant (large-N cooperative tile only): the LM head writes FP32 logits
// (samplers read const float*), so the batched-decode LM head GEMM (N=vocab >>
// kSmallNThreshold) needs a float epilogue. ElementC=ElementD=float keeps the unused
// beta=0 C/D pointers the same type so the shared impl reinterprets one buffer for both.
using ElementDFp32 = float;
using ElementCFp32 = float;
constexpr int AlignmentDFp32 = 128 / cutlass::sizeof_bits<ElementDFp32>::value;  // 4
constexpr int AlignmentCFp32 = 128 / cutlass::sizeof_bits<ElementCFp32>::value;  // 4

using CollectiveEpilogueFp32 = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator, ElementCFp32, LayoutCTag, AlignmentCFp32, ElementDFp32,
    LayoutDTag, AlignmentDFp32, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloopFp32 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogueFp32::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernelFp32 = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                           CollectiveMainloopFp32, CollectiveEpilogueFp32,
                                                           void>;
using GemmFp32 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelFp32>;

// Strides / SF layouts are derived per-variant inside the templated impl
// below (typename GemmT::GemmKernel::StrideA etc.) — both variants share the
// same SfAtom data layout, only the CTA tiling differs.

// Verify SFVecSize matches our constant (kSFVecSize = 16)
static_assert(Gemm::GemmKernel::CollectiveMainloop::TiledMma::Traits::SFVecSize == 16,
              "CUTLASS SFVecSize mismatch — expected 16 for nv_float4_t");

// Small-N variant: pingpong schedule + 128x64x128 tile. The default cooperative
// 128x128 tile starves the GPU on small-N GEMMs (e.g. kv_proj N=1024 at M=512: 32 CTAs
// on 170 SMs); pingpong's 64-wide N-tile doubles CTA count and overlaps two consumer
// warpgroups. Loses on large-N (hence the dispatch threshold). SfAtom scale layout is
// tile-shape-independent (128-row x 4-group atoms, SFVecSize=16), so both variants
// share identical A/B scale buffers.
using ThreadBlockShapeSmallN = Shape<_128, _64, _128>;

using CollectiveEpilogueSmallN = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShapeSmallN, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator, ElementC,
    LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloopSmallN = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShapeSmallN, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogueSmallN::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using GemmKernelSmallN = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                              CollectiveMainloopSmallN,
                                                              CollectiveEpilogueSmallN, void>;
using GemmSmallN = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelSmallN>;

static_assert(GemmSmallN::GemmKernel::CollectiveMainloop::TiledMma::Traits::SFVecSize == 16,
              "CUTLASS SFVecSize mismatch (small-N variant)");

// N at/below this routes to the small-N pingpong kernel. Measured: N=1024
// is 2.1x faster, N=5120 is ~25% slower — the crossover lies between;
// 2048 is the conservative cut.
static constexpr int kSmallNThreshold = 2048;

// Stream-K (mode 1) is handed to the scheduler's heuristic only where the
// data-parallel 128x128 grid leaves a short tail wave: at least one full wave and a
// last wave at most half full. Mode 2 forces stream-K at every shape (A/B testing);
// 0 disables.

namespace imp {

// ---------------------------------------------------------------------------
// CUTLASS GEMM execution
// ---------------------------------------------------------------------------

template <class GemmT>
static size_t cutlass_workspace_for(int M, int N, int K, bool force_streamk = false) {
    using GK = typename GemmT::GemmKernel;
    auto stride_A = cutlass::make_cute_packed_stride(typename GK::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GK::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename GK::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename GK::StrideD{}, {M, N, 1});

    using BlkCfg = typename GK::CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto layout_SFA = BlkCfg::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = BlkCfg::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    typename GemmT::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                   {M, N, K, 1},
                                   {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr,
                                    layout_SFB},
                                   {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}};
    if constexpr (std::is_same_v<GemmT, GemmStreamK>) {
        // The forced stream-K decomposition needs the largest reduction
        // workspace; size for it so the forced arm (mode 2, the A/B) fits.
        using DM = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
        args.scheduler.decomposition_mode = force_streamk ? DM::StreamK : DM::Heuristic;
    } else {
        (void)force_streamk;
    }

    return GemmT::get_workspace_size(args);
}

// Host-side probe of the stream-K decision for a shape: the number of
// stream-K units the scheduler would launch (0 = it decided data-parallel).
// Same argument construction as the workspace sizing, no launch, no memory.
int gemm_nvfp4_cutlass_sm120_streamk_units(int M, int N, int K, bool force) {
    using GK = GemmStreamK::GemmKernel;
    auto stride_A = cutlass::make_cute_packed_stride(typename GK::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GK::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename GK::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename GK::StrideD{}, {M, N, 1});
    using BlkCfg = typename GK::CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto layout_SFA = BlkCfg::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = BlkCfg::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    typename GemmStreamK::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                         {M, N, K, 1},
                                         {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr,
                                          layout_SFB},
                                         {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}};
    using DM = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
    args.scheduler.decomposition_mode = force ? DM::StreamK : DM::Heuristic;
    auto params = GK::to_underlying_arguments(args, nullptr);
    return static_cast<int>(params.scheduler.sk_params_.sk_units_);
}

size_t gemm_nvfp4_cutlass_sm120_streamk_workspace(int M, int N, int K) {
    return std::max(cutlass_workspace_for<GemmStreamK>(M, N, K, false),
                    cutlass_workspace_for<GemmStreamK>(M, N, K, true));
}

size_t gemm_nvfp4_cutlass_sm120_workspace(int M, int N, int K) {
    size_t ws = (N <= kSmallNThreshold) ? cutlass_workspace_for<GemmSmallN>(M, N, K)
                                        : cutlass_workspace_for<Gemm>(M, N, K);
    // Engine sizes one workspace at its MAX (M,N,K); the GEMM falls back to the dequant path when
    // a launch needs more. Stream-K's workspace is NOT monotone in shape (the heuristic at the max
    // shape picks data-parallel = 0B, while a forced decomposition needs MORE at a smaller M than
    // at the largest). Sweep every 128-tile grid up to (M,N) and keep the largest forced-stream-K
    // workspace; host-side arithmetic, once at init.
    if (N > kSmallNThreshold && imp::process_diag_nvfp4_cutlass_streamk() != 0) {
        const int m_tiles = (M + 127) / 128, n_tiles = (N + 127) / 128;
        for (int a = 1; a <= m_tiles; ++a)
            for (int b = 1; b <= n_tiles; ++b)
                if (128 * b > kSmallNThreshold)
                    ws = std::max(ws, cutlass_workspace_for<GemmStreamK>(128 * a, 128 * b, K,
                                                                          /*force_streamk=*/true));
        ws = std::max(ws, cutlass_workspace_for<GemmStreamK>(M, N, K, /*force_streamk=*/true));
    }
    return ws;
}

// Stream-K decision for the cooperative 128x128 tile (large N only).
static bool use_streamk(int M, int N) {
    const int mode = imp::process_diag_nvfp4_cutlass_streamk();
    if (mode == 0 || N <= kSmallNThreshold)
        return false;
    if (mode == 2)
        return true;
    const long tiles = (long)((M + 127) / 128) * ((N + 127) / 128);
    const long sms = device_sm_count();
    const long tail = tiles % sms;
    return tiles >= sms && tail > 0 && tail <= sms / 2;
}

template <class GemmT, class ElemD = ElementD>
static bool gemm_nvfp4_cutlass_sm120_impl(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b,
                                          void* d_fp16, int M, int N, int K, void* workspace,
                                          size_t workspace_size, cudaStream_t stream,
                                          bool force_streamk = false) {
    // Flush any prior async errors — a sticky CUDA error will make
    // cuTensorMapEncodeTiled return 719 (LAUNCH_FAILED) instead of the real code.
    {
        cudaError_t prior = cudaGetLastError();
        if (prior != cudaSuccess) {
            IMP_LOG_ERROR("CUTLASS sm120: prior CUDA error before GEMM: %s", cudaGetErrorString(prior));
            return false;
        }
    }

    using GK = typename GemmT::GemmKernel;
    auto stride_A = cutlass::make_cute_packed_stride(typename GK::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GK::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename GK::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename GK::StrideD{}, {M, N, 1});

    using BlkCfg = typename GK::CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto layout_SFA = BlkCfg::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = BlkCfg::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    auto* a_ptr = reinterpret_cast<const ElementA::DataType*>(a_data);
    auto* b_ptr = reinterpret_cast<const ElementB::DataType*>(b.data);
    auto* sfa_ptr = reinterpret_cast<const ElementA::ScaleFactorType*>(a_sf);
    auto* sfb_ptr = reinterpret_cast<const ElementB::ScaleFactorType*>(b.scale_factors);

    // C pointer must be valid even with beta=0 — CUTLASS creates a TMA
    // descriptor for C during initialize() and cuTensorMapEncodeTiled
    // fails on nullptr.  Re-use the D buffer since it's never read.
    auto* d_ptr = reinterpret_cast<ElemD*>(d_fp16);

    // Use tensor_scale as alpha: compensates for not absorbing it into SFB.
    // D = tensor_scale * (A_fp4 * SFA * B_fp4 * micro_scale_only) = correct result.
    float alpha = b.tensor_scale;

    typename GemmT::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                   {M, N, K, 1},
                                   {a_ptr, stride_A, b_ptr, stride_B, sfa_ptr, layout_SFA, sfb_ptr,
                                    layout_SFB},
                                   {{alpha, 0.0f},
                                    d_ptr,
                                    stride_C,  // C = D buffer (beta=0, never read)
                                    d_ptr,
                                    stride_D}};

    if constexpr (std::is_same_v<GemmT, GemmStreamK>) {
        using DM = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
        // Heuristic lets the scheduler fall back to data-parallel where the
        // split would not pay; mode 2 forces the stream-K decomposition.
        args.scheduler.decomposition_mode = force_streamk ? DM::StreamK : DM::Heuristic;
    } else {
        (void)force_streamk;
    }

    GemmT gemm;
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_WARN("CUTLASS sm120 NVFP4 GEMM: can_implement failed (%d) for M=%d N=%d K=%d", (int)st, M, N,
                     K);
        return false;
    }

    // The caller's workspace is the whole workspace: A7 step 8 deleted the cudaFree+cudaMalloc grow
    // path here, which ran at GEMM time on a path reachable under CUDA-graph capture (illegal
    // there) to serve a case every in-tree caller already sizes against. Refusing lets the
    // dispatch fall back to the dequant path with correct output; allocating here could not
    // (docs/internals/MEMORY.md A5.3).
    size_t needed = GemmT::get_workspace_size(args);
    if (needed > workspace_size) {
        IMP_LOG_WARN(
            "CUTLASS sm120 NVFP4 GEMM: workspace %zu B < %zu B needed for M=%d N=%d K=%d "
            "— refusing, the caller falls back",
            workspace_size, needed, M, N, K);
        return false;
    }

    st = gemm.initialize(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS sm120 NVFP4 GEMM: initialize failed (%d) M=%d N=%d K=%d", (int)st, M, N, K);
        return false;
    }

    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS sm120 NVFP4 GEMM: run failed (%d)", (int)st);
        return false;
    }

    return true;
}

bool gemm_nvfp4_cutlass_sm120(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b, void* d_fp16,
                              int M, int N, int K, void* workspace, size_t workspace_size,
                              cudaStream_t stream) {
    if (N <= kSmallNThreshold)
        return gemm_nvfp4_cutlass_sm120_impl<GemmSmallN>(a_data, a_sf, b, d_fp16, M, N, K, workspace,
                                                         workspace_size, stream);
    if (use_streamk(M, N))
        return gemm_nvfp4_cutlass_sm120_impl<GemmStreamK>(
            a_data, a_sf, b, d_fp16, M, N, K, workspace, workspace_size, stream,
            /*force_streamk=*/imp::process_diag_nvfp4_cutlass_streamk() == 2);
    return gemm_nvfp4_cutlass_sm120_impl<Gemm>(a_data, a_sf, b, d_fp16, M, N, K, workspace, workspace_size,
                                               stream);
}

// A/B probe: the pingpong 128x64 tile at a large-N shape (the dispatch keeps
// it to N <= kSmallNThreshold).
bool gemm_nvfp4_cutlass_sm120_smalln(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b,
                                     void* d_fp16, int M, int N, int K, void* workspace,
                                     size_t workspace_size, cudaStream_t stream) {
    return gemm_nvfp4_cutlass_sm120_impl<GemmSmallN>(a_data, a_sf, b, d_fp16, M, N, K, workspace,
                                                     workspace_size, stream);
}

bool gemm_nvfp4_cutlass_sm120_streamk(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b,
                                      void* d_fp16, int M, int N, int K, void* workspace,
                                      size_t workspace_size, cudaStream_t stream, bool force) {
    return gemm_nvfp4_cutlass_sm120_impl<GemmStreamK>(a_data, a_sf, b, d_fp16, M, N, K, workspace,
                                                      workspace_size, stream, force);
}

// FP32-output entry — large-N cooperative tile only (LM head: N = vocab » 2048).
bool gemm_nvfp4_cutlass_sm120_fp32(const void* a_data, const void* a_sf, const CutlassNvFP4Weight& b,
                                   void* d_fp32, int M, int N, int K, void* workspace,
                                   size_t workspace_size, cudaStream_t stream) {
    return gemm_nvfp4_cutlass_sm120_impl<GemmFp32, ElementDFp32>(a_data, a_sf, b, d_fp32, M, N, K, workspace,
                                                                workspace_size, stream);
}

size_t gemm_nvfp4_cutlass_sm120_fp32_workspace(int M, int N, int K) {
    return cutlass_workspace_for<GemmFp32>(M, N, K);
}

bool cutlass_sm120_nvfp4_available() { return true; }

}  // namespace imp
