// CUTLASS sm_120 block-scaled MXFP8 x MXFP8 -> FP16 GEMM (CUTLASS example 79c operand
// types on the 79b kernel structure). Same cooperative 128x128x128 tile and SfAtom scale
// layout as the MXFP4 twin (gemm_cutlass_mxfp4_sm120.cu); the data operand is one E4M3 byte
// per element instead of a packed nibble pair.

#include "compute/gemm_cutlass_mxfp8_sm120.h"

#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

using Mx8ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using Mx8LayoutATag = cutlass::layout::RowMajor;
constexpr int Mx8AlignmentA = 16;
using Mx8ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using Mx8LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int Mx8AlignmentB = 16;
using Mx8ElementD = cutlass::half_t;
using Mx8ElementC = cutlass::half_t;
using Mx8LayoutCTag = cutlass::layout::RowMajor;
using Mx8LayoutDTag = cutlass::layout::RowMajor;
constexpr int Mx8AlignmentD = 128 / cutlass::sizeof_bits<Mx8ElementD>::value;
constexpr int Mx8AlignmentC = 128 / cutlass::sizeof_bits<Mx8ElementC>::value;
using Mx8ElementAccumulator = float;
using Mx8ArchTag = cutlass::arch::Sm120;
using Mx8OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using Mx8ThreadBlockShape = Shape<_128, _128, _128>;
using Mx8ClusterShape = Shape<_1, _1, _1>;

using Mx8CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Mx8ArchTag, Mx8OperatorClass, Mx8ThreadBlockShape, Mx8ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, Mx8ElementAccumulator, Mx8ElementAccumulator,
    Mx8ElementC, Mx8LayoutCTag, Mx8AlignmentC, Mx8ElementD, Mx8LayoutDTag, Mx8AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using Mx8CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Mx8ArchTag, Mx8OperatorClass, Mx8ElementA, Mx8LayoutATag, Mx8AlignmentA, Mx8ElementB, Mx8LayoutBTag,
    Mx8AlignmentB, Mx8ElementAccumulator, Mx8ThreadBlockShape, Mx8ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename Mx8CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using Mx8GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mx8CollectiveMainloop,
                                                           Mx8CollectiveEpilogue, void>;
using Mx8Gemm = cutlass::gemm::device::GemmUniversalAdapter<Mx8GemmKernel>;
using Mx8StrideA = typename Mx8Gemm::GemmKernel::StrideA;
using Mx8StrideB = typename Mx8Gemm::GemmKernel::StrideB;
using Mx8StrideC = typename Mx8Gemm::GemmKernel::StrideC;
using Mx8StrideD = typename Mx8Gemm::GemmKernel::StrideD;
using Mx8Sm1xxConfig = typename Mx8Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

static_assert(Mx8Gemm::GemmKernel::CollectiveMainloop::TiledMma::Traits::SFVecSize == 32,
              "CUTLASS SFVecSize mismatch: expected 32 for mx_float8_t");

namespace imp {

// SfAtom layout, SFVecSize 32: 128 rows x 4 K-groups per 512-byte atom, K tiles inner.
static constexpr int kMx8SFVecSize = 32;
static constexpr int kMx8AtomRows = 128;
static constexpr int kMx8AtomKGroups = 4;
static constexpr int kMx8AtomKElems = kMx8SFVecSize * kMx8AtomKGroups;  // 128
static constexpr int kMx8AtomSize = kMx8AtomRows * kMx8AtomKGroups;     // 512

__device__ __host__ __forceinline__ int mx8_sfatom_offset(int row, int k_group, int n_k_tiles) {
    const int tile_row = row / kMx8AtomRows;
    const int tile_k = k_group / kMx8AtomKGroups;
    const int row_local = row % kMx8AtomRows;
    const int k_local = k_group % kMx8AtomKGroups;
    const int n0 = row_local % 32;
    const int n1 = row_local / 32;
    return (tile_row * n_k_tiles + tile_k) * kMx8AtomSize + n0 * 16 + n1 * 4 + k_local;
}

size_t cutlass_mxfp8_sf_size(int rows, int K) {
    const int n_row_tiles = (rows + kMx8AtomRows - 1) / kMx8AtomRows;
    const int n_k_tiles = (K + kMx8AtomKElems - 1) / kMx8AtomKElems;
    return static_cast<size_t>(n_row_tiles) * n_k_tiles * kMx8AtomSize;
}

// UE8M0 exponent e with 2^(e-127) >= absmax / 448, i.e. the block absmax lands in (224, 448]
// of the E4M3 range. 448 = 1.75 * 2^8, so the target is exact at absmax = 448 * 2^k.
__device__ __forceinline__ uint8_t mx8_scale_ue8m0(float absmax) {
    if (absmax <= 0.0f)
        return 127;  // scale 1.0: an all-zero block encodes as zeros either way
    const float target = absmax * (1.0f / 448.0f);
    uint32_t bits = __float_as_uint(target);
    int e = static_cast<int>((bits >> 23) & 0xFF);
    if (bits & 0x7FFFFF)
        e++;  // ceil to the next power of two
    if (e < 1)
        e = 1;
    if (e > 254)
        e = 254;
    return static_cast<uint8_t>(e);
}

__device__ __forceinline__ float mx8_ue8m0_to_float(uint8_t e) {
    return __uint_as_float(static_cast<uint32_t>(e) << 23);
}

// Two FP32 -> one E4M3 pair, v0 in the low byte. RNE, saturating (E4M3 has no inf).
__device__ __forceinline__ uint16_t mx8_pack_e4m3_pair(float v0, float v1) {
#if __CUDA_ARCH__ >= 890
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(out) : "f"(v0), "f"(v1));
    return out;
#else
    return 0;
#endif
}

// One thread per 32-element block: 64 B in (4 x uint4), 32 B out (2 x uint4), one scale byte.
__global__ void quantize_fp16_mxfp8_cutlass_kernel(const half* __restrict__ input, uint8_t* __restrict__ out,
                                                   uint8_t* __restrict__ sf_out, int rows, int K,
                                                   int n_k_tiles) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int K_groups = K / kMx8SFVecSize;
    if (idx >= rows * K_groups)
        return;
    const int row = idx / K_groups;
    const int k_group = idx % K_groups;
    const size_t base = static_cast<size_t>(row) * K + static_cast<size_t>(k_group) * kMx8SFVecSize;

    float vals[kMx8SFVecSize];
    float absmax = 0.0f;
    const uint4* src = reinterpret_cast<const uint4*>(input + base);
#pragma unroll
    for (int v = 0; v < 4; v++) {
        const uint4 q = src[v];
        const half2* h = reinterpret_cast<const half2*>(&q);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 f = __half22float2(h[j]);
            vals[v * 8 + j * 2] = f.x;
            vals[v * 8 + j * 2 + 1] = f.y;
            absmax = fmaxf(absmax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    const uint8_t e = mx8_scale_ue8m0(absmax);
    sf_out[mx8_sfatom_offset(row, k_group, n_k_tiles)] = e;
    const float inv = 1.0f / mx8_ue8m0_to_float(e);

    uint32_t words[8];
#pragma unroll
    for (int w = 0; w < 8; w++) {
        const uint32_t lo = mx8_pack_e4m3_pair(vals[w * 4] * inv, vals[w * 4 + 1] * inv);
        const uint32_t hi = mx8_pack_e4m3_pair(vals[w * 4 + 2] * inv, vals[w * 4 + 3] * inv);
        words[w] = lo | (hi << 16);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + base);
    dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
    dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
}

void quantize_fp16_to_mxfp8_cutlass(const void* src_fp16, void* dst_data, void* dst_sf, int M, int K,
                                    cudaStream_t stream) {
    IMP_CHECK(K % kMx8SFVecSize == 0, "quantize_fp16_to_mxfp8_cutlass: K=%d must be a multiple of %d", K,
              kMx8SFVecSize);
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(dst_sf, 0, cutlass_mxfp8_sf_size(M, K), stream));
    const int K_groups = K / kMx8SFVecSize;
    const int total = M * K_groups;
    const int n_k_tiles = (K + kMx8AtomKElems - 1) / kMx8AtomKElems;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    quantize_fp16_mxfp8_cutlass_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(src_fp16), reinterpret_cast<uint8_t*>(dst_data),
        reinterpret_cast<uint8_t*>(dst_sf), M, K, n_k_tiles);
    IMP_CUDA_CHECK_LAUNCH();
}

static typename Mx8Gemm::Arguments mx8_arguments(const void* a_data, const void* a_sf,
                                                 const CutlassMxFP8Weight& b, void* d_fp16, int M, int N,
                                                 int K) {
    auto stride_A = cutlass::make_cute_packed_stride(Mx8StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(Mx8StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(Mx8StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(Mx8StrideD{}, {M, N, 1});
    auto layout_SFA = Mx8Sm1xxConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Mx8Sm1xxConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    auto* a_ptr = reinterpret_cast<const Mx8ElementA::DataType*>(a_data);
    auto* b_ptr = reinterpret_cast<const Mx8ElementB::DataType*>(b.data);
    auto* sfa_ptr = reinterpret_cast<const Mx8ElementA::ScaleFactorType*>(a_sf);
    auto* sfb_ptr = reinterpret_cast<const Mx8ElementB::ScaleFactorType*>(b.scale_factors);
    auto* d_ptr = reinterpret_cast<Mx8ElementD*>(d_fp16);
    return typename Mx8Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                       {M, N, K, 1},
                                       {a_ptr, stride_A, b_ptr, stride_B, sfa_ptr, layout_SFA, sfb_ptr,
                                        layout_SFB},
                                       {{1.0f, 0.0f}, d_ptr, stride_C, d_ptr, stride_D}};
}

size_t gemm_mxfp8_cutlass_sm120_workspace(int M, int N, int K) {
    CutlassMxFP8Weight b;
    b.N = N;
    b.K = K;
    return Mx8Gemm::get_workspace_size(mx8_arguments(nullptr, nullptr, b, nullptr, M, N, K));
}

bool gemm_mxfp8_cutlass_sm120(const void* a_data, const void* a_sf, const CutlassMxFP8Weight& b, void* d_fp16,
                              int M, int N, int K, void* workspace, size_t workspace_size,
                              cudaStream_t stream) {
    {
        const cudaError_t prior = cudaGetLastError();
        if (prior != cudaSuccess) {
            IMP_LOG_ERROR("CUTLASS MXFP8 sm120: prior CUDA error: %s", cudaGetErrorString(prior));
            return false;
        }
    }
    auto args = mx8_arguments(a_data, a_sf, b, d_fp16, M, N, K);
    Mx8Gemm gemm;
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_WARN("CUTLASS MXFP8 sm120: can_implement failed (%d) M=%d N=%d K=%d", (int)st, M, N, K);
        return false;
    }
    const size_t needed = Mx8Gemm::get_workspace_size(args);
    if (needed > workspace_size) {
        IMP_LOG_WARN("CUTLASS MXFP8 sm120: workspace %zu B < %zu B needed for M=%d N=%d K=%d, refusing",
                     workspace_size, needed, M, N, K);
        return false;
    }
    st = gemm.initialize(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS MXFP8 sm120: initialize failed (%d) M=%d N=%d K=%d", (int)st, M, N, K);
        return false;
    }
    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) {
        IMP_LOG_ERROR("CUTLASS MXFP8 sm120: run failed (%d)", (int)st);
        return false;
    }
    return true;
}

bool cutlass_sm120_mxfp8_available() { return true; }

}  // namespace imp
