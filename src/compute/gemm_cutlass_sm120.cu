// CUTLASS sm_120 block-scaled NVFP4×NVFP4 GEMM for prefill acceleration.
//
// Uses CUTLASS 4.4.1 Example 79a pattern: Warp-Specialized persistent kernel
// with block-scaled tensor core MMA (mma.sync.aligned.block_scale) on
// Blackwell GeForce (sm_120).
//
// Both A (activation) and B (weight) use nv_float4_t<float_e2m1_t> with
// float_ue4m3_t unsigned scale factors in SfAtom interleaved layout.
// Output D is FP16 (cutlass::half_t) for direct use in the inference pipeline.
//
// Weight format conversion (once at init):
//   - Borrow packed FP4 pointer [N, K/2] (K-contiguous RowMajor)
//   - Convert micro_scales from linear [N, K/16] to SfAtom UE4M3 layout
//   - tensor_scale is NOT absorbed into scale factors (to avoid UE4M3
//     denormalized range precision loss); instead applied as GEMM alpha
//
// Activation quantization (per-prefill-call):
//   - FP16 [M, K] → NVFP4 packed [M, K/2] + SfAtom UE4M3 scales
//

#include "compute/gemm_cutlass_sm120.h"
#include "quant/nvfp4_quant.h"
#include "quant/fp8_utils.cuh"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cassert>

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

// ---------------------------------------------------------------------------
// CUTLASS GEMM type configuration: NVFP4 × NVFP4 → FP16
// Based on Example 79a but with half_t output instead of bfloat16_t.
// ---------------------------------------------------------------------------

using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag  = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD    = cutlass::half_t;           // FP16 output
using ElementC    = cutlass::half_t;           // C matrix type (unused, beta=0)
using LayoutCTag  = cutlass::layout::RowMajor;
using LayoutDTag  = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;   // 8
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;   // 8

using ElementAccumulator = float;
using ArchTag       = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape     = Shape<_1, _1, _1>;     // GeForce = no multicast

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Verify SFVecSize matches our constant (kSFVecSize = 16)
static_assert(Gemm::GemmKernel::CollectiveMainloop::TiledMma::Traits::SFVecSize == 16,
              "CUTLASS SFVecSize mismatch — expected 16 for nv_float4_t");


namespace imp {

// ---------------------------------------------------------------------------
// SfAtom layout computation (hardware-independent arithmetic)
// ---------------------------------------------------------------------------
// SfAtom for K-major, SFVecSize=16:
//   Shape:  ((32, 4), (16, 4))
//   Stride: ((16, 4), ( 0, 1))
//
// Each atom covers 128 rows × 4 scale-groups (= 64 data elements in K).
// Atom size = 128 * 4 = 512 bytes.
//
// tile_to_shape tiles atoms to cover (rows, K) with Step<_2, _1>:
//   K dimension tiles are inner (faster-changing), row tiles are outer.

static constexpr int kSFVecSize = 16;
static constexpr int kAtomRows = 128;     // 32 * 4
static constexpr int kAtomKGroups = 4;    // 4 scale groups per atom
static constexpr int kAtomKElems = kSFVecSize * kAtomKGroups;  // 64
static constexpr int kAtomSize = kAtomRows * kAtomKGroups;     // 512

// Compute SfAtom offset for logical scale factor at (row, k_group).
__device__ __host__ __forceinline__
int sfatom_offset(int row, int k_group, int n_k_tiles) {
    int tile_row = row / kAtomRows;
    int tile_k   = k_group / kAtomKGroups;
    int row_local = row % kAtomRows;
    int k_local   = k_group % kAtomKGroups;

    int n0 = row_local % 32;  // within 32-row sub-block
    int n1 = row_local / 32;  // which of 4 sub-blocks

    int atom_offset = n0 * 16 + n1 * 4 + k_local;
    int tile_base   = (tile_row * n_k_tiles + tile_k) * kAtomSize;
    return tile_base + atom_offset;
}

size_t cutlass_nvfp4_sf_size(int rows, int K) {
    int n_row_tiles = (rows + kAtomRows - 1) / kAtomRows;
    int n_k_tiles   = (K + kAtomKElems - 1) / kAtomKElems;
    return static_cast<size_t>(n_row_tiles) * n_k_tiles * kAtomSize;
}

// ---------------------------------------------------------------------------
// GPU kernels for weight conversion
// ---------------------------------------------------------------------------

// Convert micro_scales from linear layout to SfAtom layout (NO tensor_scale absorption).
// tensor_scale is deferred to the GEMM epilogue alpha parameter for precision.
// Source: [N, K/16] FP8 E4M3 (signed, but always positive for scale factors)
// Dest:   SfAtom layout UE4M3 (unsigned, just micro_scale — NOT combined)
__global__ void convert_scales_sfatom_kernel(
    const uint8_t* __restrict__ src_ms,    // [N, K/16] linear
    uint8_t*       __restrict__ dst_sf,    // SfAtom layout
    int N, int K, int n_k_tiles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_groups = K / kSFVecSize;
    int total = N * K_groups;
    if (idx >= total) return;

    int n = idx / K_groups;
    int k_group = idx % K_groups;

    // Read signed E4M3 micro-scale, drop its sign (always positive for scales),
    // then re-encode as UE4M3 via the shared float↔E4M3 helper. UE4M3 is
    // bit-identical to positive E4M3, so float_to_fp8_e4m3 with a positive
    // argument yields the UE4M3 byte directly (sign bit = 0).
    float combined = fabsf(fp8_e4m3_to_float_fast(src_ms[idx]));
    dst_sf[sfatom_offset(n, k_group, n_k_tiles)] = float_to_fp8_e4m3(combined);
}

// ---------------------------------------------------------------------------
// Activation quantization: FP16 [M, K] → NVFP4 packed + SfAtom UE4M3 scales
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint8_t quantize_abs_to_fp4(float abs_val) {
    // Branchless: count of midpoint thresholds exceeded gives the E2M1 code.
    uint8_t code = (abs_val >= 0.25f) + (abs_val >= 0.75f) + (abs_val >= 1.25f)
                 + (abs_val >= 1.75f) + (abs_val >= 2.5f)  + (abs_val >= 3.5f)
                 + (abs_val >= 5.0f);
    return code;  // 0..7
}

// HW FP4 conversion: two scaled FP32 → packed byte (low=v0, high=v1).
// IEEE RNE rounding; sm_120+ only.
__device__ __forceinline__ uint8_t pack_fp4_pair_hw(float v0, float v1) {
#if __CUDA_ARCH__ >= 1200
    uint32_t out;
    asm volatile(
        "{ .reg .b8 b;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
        "  cvt.u32.u8 %0, b; }\n"
        : "=r"(out)
        : "f"(v0), "f"(v1));
    return static_cast<uint8_t>(out);
#else
    uint8_t sign0 = (v0 < 0.0f) ? 1u : 0u;
    uint8_t sign1 = (v1 < 0.0f) ? 1u : 0u;
    uint8_t c0 = (sign0 << 3) | quantize_abs_to_fp4(fabsf(v0));
    uint8_t c1 = (sign1 << 3) | quantize_abs_to_fp4(fabsf(v1));
    return (c1 << 4) | c0;
#endif
}

// Given 16 pre-computed float values + their absmax, encode UE4M3 scale and
// pack FP4 bytes. The caller supplies the values (so this helper is reusable
// for fused paths like SwiGLU+quantize where values come from a computation
// rather than a direct FP16 load).
__device__ __forceinline__ void quantize_micro_block_nvfp4_from_vals(
    const float vals[kSFVecSize],
    float local_absmax,
    uint8_t* packed_out_row,
    int k_group,
    uint8_t* sfa_target)
{
    // Encode UE4M3 scale (positive — `float_to_fp8_e4m3` handles clamp + rounding
    // and returns sign=0 for non-negative input, which is a valid UE4M3 byte).
    float scale_f = local_absmax / 6.0f;
    uint8_t ue4m3 = float_to_fp8_e4m3(scale_f);

    // Reconstruct actual scale from UE4M3 for consistent quantization. If the
    // scale rounds to zero, fall back to the smallest denorm (2^-9) to avoid
    // division by zero — matches the >=2^-9 clamp used elsewhere in imp.
    float actual_scale = fp8_e4m3_to_float_fast(ue4m3);
    if (actual_scale == 0.0f) actual_scale = 1.0f / 512.0f;
    float inv_scale = 1.0f / actual_scale;

    *sfa_target = ue4m3;

    uint8_t* packed_at = packed_out_row + k_group * (kSFVecSize / 2);
    #pragma unroll
    for (int i = 0; i < kSFVecSize; i += 2) {
        float s0 = vals[i]     * inv_scale;
        float s1 = vals[i + 1] * inv_scale;
        packed_at[i / 2] = pack_fp4_pair_hw(s0, s1);
    }
}

// Direct FP16 quantize: load 16 FP16 values, pass to the above helper.
__device__ __forceinline__ void quantize_micro_block_nvfp4(
    const half* input_row_base,
    int k_group,
    uint8_t* packed_out_row,
    uint8_t* sfa_target)
{
    float vals[kSFVecSize];
    float local_absmax = 0.0f;
    const half2* src_h2 = reinterpret_cast<const half2*>(input_row_base + k_group * kSFVecSize);
    #pragma unroll
    for (int i = 0; i < kSFVecSize / 2; i++) {
        half2 h2 = src_h2[i];
        vals[i * 2]     = __half2float(h2.x);
        vals[i * 2 + 1] = __half2float(h2.y);
        local_absmax = fmaxf(local_absmax, fmaxf(fabsf(vals[i * 2]), fabsf(vals[i * 2 + 1])));
    }
    quantize_micro_block_nvfp4_from_vals(vals, local_absmax, packed_out_row, k_group, sfa_target);
}


// Single-tensor quantize: row numbering is direct, SFA is a single linear buffer
// with SfAtom layout over (row, k_group).
__global__ void quantize_fp16_nvfp4_cutlass_kernel(
    const half* __restrict__ input,        // [M, K] FP16
    uint8_t*    __restrict__ packed_out,    // [M, K/2] packed nibbles
    uint8_t*    __restrict__ sf_out,        // SfAtom layout UE4M3
    int M, int K, int n_k_tiles)
{
    int mb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_groups = K / kSFVecSize;
    int total_mb = M * K_groups;
    if (mb_idx >= total_mb) return;

    int row    = mb_idx / K_groups;
    int k_group = mb_idx % K_groups;

    quantize_micro_block_nvfp4(
        input + static_cast<int64_t>(row) * K,
        k_group,
        packed_out + static_cast<int64_t>(row) * (K / 2),
        sf_out + sfatom_offset(row, k_group, n_k_tiles));
}

// Device helper: binary-search `offsets` for the expert owning `row`.
// Returns expert index and writes `local_row` (row relative to expert's slab).
__device__ __forceinline__ int moe_find_expert(const int* offsets, int ne, int row, int& local_row) {
    int lo = 0, hi = ne;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (offsets[mid] <= row) lo = mid;
        else hi = mid;
    }
    local_row = row - offsets[lo];
    return lo;
}

// MoE variant: one kernel quantizes all [expanded, K] rows into contiguous
// packed output + per-expert SFA slabs (one per expert).
__global__ void quantize_fp16_nvfp4_cutlass_moe_kernel(
    const half* __restrict__ input,           // [expanded, K] FP16
    uint8_t*    __restrict__ packed_out,       // [expanded, K/2] contiguous
    uint8_t* const* __restrict__ sfa_bases,    // [ne] per-expert SFA base (may be null)
    const int*    __restrict__ offsets,        // [ne+1] cumulative row offsets
    int expanded, int K, int ne, int n_k_tiles)
{
    int mb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_groups = K / kSFVecSize;
    if (mb_idx >= expanded * K_groups) return;

    int row     = mb_idx / K_groups;
    int k_group = mb_idx % K_groups;
    int local_row;
    int expert = moe_find_expert(offsets, ne, row, local_row);
    uint8_t* sfa = sfa_bases[expert];
    if (!sfa) return;

    quantize_micro_block_nvfp4(
        input + static_cast<int64_t>(row) * K,
        k_group,
        packed_out + static_cast<int64_t>(row) * (K / 2),
        sfa + sfatom_offset(local_row, k_group, n_k_tiles));
}


// ---------------------------------------------------------------------------
// Host-callable functions
// ---------------------------------------------------------------------------

void convert_nvfp4_to_cutlass(const NvFP4QuantResult& src,
                               CutlassNvFP4Weight& dst,
                               cudaStream_t stream)
{
    assert(src.packed_data && "source must be quantized");
    int64_t N = src.N;
    int64_t K = src.K;

    // Data pointer is borrowed as RowMajor [N, K/2].
    // Despite LayoutBTag=ColumnMajor, CUTLASS block-scaled GEMM uses RowMajor stride.

    // Allocate SfAtom scale buffer
    size_t sf_bytes = cutlass_nvfp4_sf_size(static_cast<int>(N), static_cast<int>(K));
    void* d_sf = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_sf, sf_bytes));
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_sf, 0, sf_bytes, stream));  // zero-init for padding

    // Convert scales to SfAtom layout (micro_scale only, tensor_scale deferred to GEMM alpha)
    {
        int K_groups = static_cast<int>(K) / kSFVecSize;
        int total = static_cast<int>(N) * K_groups;
        int n_k_tiles = (static_cast<int>(K) + kAtomKElems - 1) / kAtomKElems;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        convert_scales_sfatom_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(src.micro_scales),
            reinterpret_cast<uint8_t*>(d_sf),
            static_cast<int>(N), static_cast<int>(K), n_k_tiles);
    }

    dst.data = src.packed_data;  // borrowed pointer (not owned)
    dst.scale_factors = d_sf;
    dst.tensor_scale = src.tensor_scale;
    dst.N = N;
    dst.K = K;
    dst.sf_bytes = sf_bytes;

    IMP_LOG_DEBUG("convert_nvfp4_to_cutlass: N=%lld K=%lld sf=%.2f MiB (data borrowed)",
                  (long long)N, (long long)K,
                  sf_bytes / (1024.0 * 1024.0));
}

void free_cutlass_nvfp4_weight(CutlassNvFP4Weight& w) {
    // data is borrowed from NvFP4QuantResult — do NOT free it
    w.data = nullptr;
    if (w.scale_factors) { IMP_CUDA_CHECK_LOG(cudaFree(w.scale_factors)); w.scale_factors = nullptr; }
    w.N = w.K = 0;
    w.sf_bytes = 0;
}

void quantize_fp16_to_nvfp4_cutlass(const void* src_fp16, void* dst_data,
                                     void* dst_sf, int M, int K,
                                     cudaStream_t stream)
{
    assert(K % kSFVecSize == 0 && "K must be multiple of 16");

    // Zero the SF buffer for padding safety
    size_t sf_bytes = cutlass_nvfp4_sf_size(M, K);
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(dst_sf, 0, sf_bytes, stream));

    int K_groups = K / kSFVecSize;
    int total_mb = M * K_groups;
    int n_k_tiles = (K + kAtomKElems - 1) / kAtomKElems;

    int threads = 256;
    int blocks = (total_mb + threads - 1) / threads;
    quantize_fp16_nvfp4_cutlass_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(src_fp16),
        reinterpret_cast<uint8_t*>(dst_data),
        reinterpret_cast<uint8_t*>(dst_sf),
        M, K, n_k_tiles);
}

void quantize_fp16_to_nvfp4_cutlass_moe(const void* src_fp16,
                                        void* dst_packed,
                                        uint8_t* const* d_sfa_bases,
                                        const int* d_offsets,
                                        int expanded, int K, int ne,
                                        cudaStream_t stream)
{
    assert(K % kSFVecSize == 0 && "K must be multiple of 16");
    if (expanded == 0) return;

    int K_groups = K / kSFVecSize;
    int total_mb = expanded * K_groups;
    int n_k_tiles = (K + kAtomKElems - 1) / kAtomKElems;

    int threads = 256;
    int blocks = (total_mb + threads - 1) / threads;
    quantize_fp16_nvfp4_cutlass_moe_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(src_fp16),
        reinterpret_cast<uint8_t*>(dst_packed),
        d_sfa_bases, d_offsets,
        expanded, K, ne, n_k_tiles);
}


// ---------------------------------------------------------------------------
// CUTLASS GEMM execution
// ---------------------------------------------------------------------------

// Persistent workspace and GEMM instance
static void* s_cutlass_workspace = nullptr;
static size_t s_cutlass_workspace_size = 0;

size_t gemm_nvfp4_cutlass_sm120_workspace(int M, int N, int K) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}
    };

    return Gemm::get_workspace_size(args);
}

bool gemm_nvfp4_cutlass_sm120(const void* a_data, const void* a_sf,
                               const CutlassNvFP4Weight& b,
                               void* d_fp16, int M, int N, int K,
                               void* workspace, size_t workspace_size,
                               cudaStream_t stream)
{
    // Flush any prior async errors — a sticky CUDA error will make
    // cuTensorMapEncodeTiled return 719 (LAUNCH_FAILED) instead of the real code.
    {
        cudaError_t prior = cudaGetLastError();
        if (prior != cudaSuccess) {
            IMP_LOG_ERROR("CUTLASS sm120: prior CUDA error before GEMM: %s", cudaGetErrorString(prior));
            return false;
        }
    }

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    auto* a_ptr = reinterpret_cast<const ElementA::DataType*>(a_data);
    auto* b_ptr = reinterpret_cast<const ElementB::DataType*>(b.data);
    auto* sfa_ptr = reinterpret_cast<const ElementA::ScaleFactorType*>(a_sf);
    auto* sfb_ptr = reinterpret_cast<const ElementB::ScaleFactorType*>(b.scale_factors);

    // C pointer must be valid even with beta=0 — CUTLASS creates a TMA
    // descriptor for C during initialize() and cuTensorMapEncodeTiled
    // fails on nullptr.  Re-use the D buffer since it's never read.
    auto* d_ptr = reinterpret_cast<ElementD*>(d_fp16);

    // Use tensor_scale as alpha: compensates for not absorbing it into SFB.
    // D = tensor_scale * (A_fp4 * SFA * B_fp4 * micro_scale_only) = correct result.
    float alpha = b.tensor_scale;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {a_ptr, stride_A, b_ptr, stride_B, sfa_ptr, layout_SFA, sfb_ptr, layout_SFB},
        {{alpha, 0.0f},
         d_ptr, stride_C,  // C = D buffer (beta=0, never read)
         d_ptr, stride_D}
    };

    Gemm gemm;
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_WARN("CUTLASS sm120 NVFP4 GEMM: can_implement failed (%d) for M=%d N=%d K=%d",
                     (int)st, M, N, K);
        return false;
    }

    // Ensure workspace
    size_t needed = Gemm::get_workspace_size(args);
    void* ws = workspace;
    if (needed > workspace_size) {
        if (needed > s_cutlass_workspace_size) {
            if (s_cutlass_workspace) IMP_CUDA_CHECK_LOG(cudaFree(s_cutlass_workspace));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&s_cutlass_workspace, needed));
            s_cutlass_workspace_size = needed;
        }
        ws = s_cutlass_workspace;
    }

    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS sm120 NVFP4 GEMM: initialize failed (%d) M=%d N=%d K=%d",
                      (int)st, M, N, K);
        return false;
    }

    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS sm120 NVFP4 GEMM: run failed (%d)", (int)st);
        return false;
    }

    return true;
}

bool cutlass_sm120_nvfp4_available() {
    return true;
}


} // namespace imp
