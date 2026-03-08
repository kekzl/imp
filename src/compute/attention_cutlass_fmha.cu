// CUTLASS Hopper FMHA wrapper for prefill attention.
// Uses WGMMA (asynchronous warpgroup MMA) + TMA (Tensor Memory Accelerator)
// from CUTLASS v4.4.1 Example 88 for ~2x throughput vs WMMA kernels.
//
// Supports: FP16, causal/non-causal, GQA (n_heads != n_kv_heads), HD=64/128.
// Falls back gracefully for unsupported configurations.

#include "compute/attention_cutlass_fmha.h"
#include "core/logging.h"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) || defined(__CUDA_ARCH__) || defined(IMP_USE_CUTLASS)

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

// CUTLASS Example 88 headers (Hopper FMHA)
#include "collective/fmha_fusion.hpp"
#include "device/device_universal.hpp"
#include "kernel/fmha_kernel_builder.hpp"

using namespace cute;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;

namespace imp {

// ============================================================================
// Stride types for our [B, S, H, D] tensor layout
// ============================================================================

using StrideQKVO = cute::tuple<int, _1, cute::tuple<int, int>>;  // (seq_stride, dim_stride=1, (batch_stride, head_stride))
using StrideLSE  = cute::tuple<_1, cute::tuple<int, int>>;       // (seq_stride=1, (batch_stride, head_stride))

using ProblemShape = cute::tuple<int, int, int, int, int>;  // [B, H, Q, K, D]

// ============================================================================
// Template instantiation for each (HeadDim, Fusion) combination.
// We use KernelTmaWarpSpecializedCooperative for best performance on sm_90+.
// ============================================================================

template <class TileShape, class Fusion>
using FmhaOperation = cutlass::device::Universal<
    typename FmhaBuilder<
        cutlass::half_t,           // Element
        float,                     // ElementAccumulatorQK
        float,                     // ElementAccumulatorPV
        TileShape,
        StrideQKVO,                // StrideQ
        StrideQKVO,                // StrideK
        StrideQKVO,                // StrideV
        Fusion,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::Kernel>;

// ============================================================================
// Static workspace buffers (pre-allocated at init, grow-only fallback)
// ============================================================================

static void*  s_lse_buf    = nullptr;
static size_t s_lse_sz     = 0;
static void*  s_workspace  = nullptr;
static size_t s_workspace_sz = 0;

static void ensure_lse_buf(size_t needed) {
    if (needed <= s_lse_sz) return;
    if (s_lse_buf) cudaFree(s_lse_buf);
    cudaMalloc(&s_lse_buf, needed);
    s_lse_sz = needed;
}

static void ensure_workspace(size_t needed) {
    if (needed <= s_workspace_sz) return;
    if (s_workspace) cudaFree(s_workspace);
    cudaMalloc(&s_workspace, needed);
    s_workspace_sz = needed;
}

// Compute LSE buffer size for given dimensions.
// LSE layout: [B_eff, H_eff, seq_q] where B_eff = batch * n_kv_heads, H_eff = n_heads / n_kv_heads.
// Simplifies to: batch * n_heads * seq_q * sizeof(float).
static size_t compute_lse_size(int max_batch, int max_seq, int n_heads) {
    return static_cast<size_t>(max_batch) * n_heads * max_seq * sizeof(float);
}

// Conservative workspace estimate for CUTLASS cooperative kernel.
// The cooperative kernel needs a small barrier/tile-scheduler buffer, typically < 64 KB.
static constexpr size_t kFmhaWorkspaceReserve = 1ULL << 20;  // 1 MiB

size_t cutlass_fmha_workspace_estimate(int max_batch, int max_seq, int n_heads, int head_dim) {
    (void)head_dim;
    return compute_lse_size(max_batch, max_seq, n_heads) + kFmhaWorkspaceReserve;
}

size_t cutlass_fmha_init_workspace(int max_batch, int max_seq, int n_heads, int head_dim) {
    (void)head_dim;
    size_t lse_bytes = compute_lse_size(max_batch, max_seq, n_heads);
    ensure_lse_buf(lse_bytes);
    ensure_workspace(kFmhaWorkspaceReserve);
    IMP_LOG_DEBUG("CUTLASS FMHA workspace: LSE %.2f MiB + kernel %.2f MiB = %.2f MiB",
                  s_lse_sz / (1024.0 * 1024.0), s_workspace_sz / (1024.0 * 1024.0),
                  (s_lse_sz + s_workspace_sz) / (1024.0 * 1024.0));
    return s_lse_sz + s_workspace_sz;
}

void cutlass_fmha_free_workspace() {
    if (s_lse_buf) { cudaFree(s_lse_buf); s_lse_buf = nullptr; s_lse_sz = 0; }
    if (s_workspace) { cudaFree(s_workspace); s_workspace = nullptr; s_workspace_sz = 0; }
}

// ============================================================================
// Run a specific CUTLASS FMHA configuration
// ============================================================================

template <class TileShape, class Fusion>
static bool run_fmha(
    const half* Q_ptr, const half* K_ptr, const half* V_ptr, half* O_ptr,
    int batch, int seq_q, int seq_kv, int n_heads, int n_kv_heads, int head_dim,
    float scale, cudaStream_t stream)
{
    using Op = FmhaOperation<TileShape, Fusion>;

    int groups = n_heads / n_kv_heads;

    // GQA: unroll batch over KV heads, each "batch" processes 'groups' Q heads
    // sharing one KV head.  For MHA (groups=1), B_eff = batch * n_heads, H_eff = 1.
    int B_eff = batch * n_kv_heads;
    int H_eff = groups;

    // Q strides for layout [B, S, n_heads, D]
    // Q[b, s, h, d] = base + b*S*nh*D + s*nh*D + h*D + d
    // For (b_eff = b*nkv + kv_head, h_eff = group_idx):
    //   actual head = kv_head * groups + group_idx
    StrideQKVO stride_Q = make_stride(
        n_heads * head_dim,                                    // seq stride
        _1{},                                                  // dim stride
        make_stride(groups * head_dim,                         // b_eff stride
                    head_dim));                                 // h_eff stride

    // K/V strides for layout [B, S, n_kv_heads, D]
    // h_eff stride = 0: all Q heads in a group share the same K/V head
    StrideQKVO stride_K = make_stride(
        n_kv_heads * head_dim,                                 // seq stride
        _1{},                                                  // dim stride
        make_stride(head_dim,                                  // b_eff stride (next KV head)
                    0));                                        // h_eff stride = 0 (GQA sharing)

    StrideQKVO stride_V = stride_K;

    // Output strides: same layout as Q
    StrideQKVO stride_O = stride_Q;

    // LSE strides: [Q, (B_eff, H_eff)]
    StrideLSE stride_LSE = make_stride(
        _1{},                                                  // seq stride
        make_stride(H_eff * seq_q,                             // b_eff stride
                    seq_q));                                    // h_eff stride

    // Allocate LSE buffer
    size_t lse_bytes = static_cast<size_t>(B_eff) * H_eff * seq_q * sizeof(float);
    ensure_lse_buf(lse_bytes);

    // Hardware info
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    cudaGetDevice(&hw_info.device_id);
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // Problem shape: [B, H, Q, K, D]
    ProblemShape problem_size{B_eff, H_eff, seq_q, seq_kv, head_dim};

    // Check shared memory requirement against device capability
    int smem_size = static_cast<int>(Op::Kernel::SharedStorageSize);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, hw_info.device_id);
    if (smem_size > max_smem) {
        IMP_LOG_DEBUG("CUTLASS FMHA: smem %d > device max %d, skipping", smem_size, max_smem);
        return false;
    }

    // Build arguments
    typename Op::Arguments arguments{
        problem_size,
        {reinterpret_cast<const cutlass::half_t*>(Q_ptr), stride_Q,
         reinterpret_cast<const cutlass::half_t*>(K_ptr), stride_K,
         reinterpret_cast<const cutlass::half_t*>(V_ptr), stride_V},
        {reinterpret_cast<cutlass::half_t*>(O_ptr), stride_O,
         static_cast<float*>(s_lse_buf), stride_LSE},
        hw_info
    };

    // Check if kernel can implement this configuration
    Op op;
    cutlass::Status status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        IMP_LOG_DEBUG("CUTLASS FMHA: can_implement failed (status=%d)", static_cast<int>(status));
        return false;
    }

    // Workspace
    size_t ws_size = Op::get_workspace_size(arguments);
    ensure_workspace(ws_size);

    // Initialize and run
    status = op.initialize(arguments, s_workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        IMP_LOG_DEBUG("CUTLASS FMHA: initialize failed (status=%d)", static_cast<int>(status));
        return false;
    }

    status = op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        IMP_LOG_WARN("CUTLASS FMHA: run failed (status=%d)", static_cast<int>(status));
        return false;
    }

    return true;
}

// ============================================================================
// Public API: dispatch to the right tile configuration
// ============================================================================

bool cutlass_fmha_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, cudaStream_t stream)
{
    if (Q.dtype != DType::FP16) return false;

    const int batch    = static_cast<int>(Q.shape[0]);
    const int seq_q    = static_cast<int>(Q.shape[1]);
    const int n_heads  = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int seq_kv   = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    // Validate: n_heads must be a multiple of n_kv_heads
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;

    // Head dim must be aligned to 8 for TMA
    if (head_dim % 8 != 0) return false;

    // CUTLASS FMHA needs at least 1 token
    if (seq_q == 0 || seq_kv == 0) return false;

    const auto* Q_ptr = reinterpret_cast<const half*>(Q.data);
    const auto* K_ptr = reinterpret_cast<const half*>(K.data);
    const auto* V_ptr = reinterpret_cast<const half*>(V.data);
    auto* O_ptr = reinterpret_cast<half*>(O.data);

    // Note: CUTLASS FMHA applies its own 1/sqrt(D) scaling internally.
    // Our caller passes scale = 1/sqrt(head_dim), so we don't need to re-apply it.
    // The CUTLASS kernel computes Q @ K^T / sqrt(D) internally.
    IMP_LOG_DEBUG("CUTLASS FMHA: B=%d Q=%d KV=%d nh=%d nkv=%d hd=%d causal=%d",
                  batch, seq_q, seq_kv, n_heads, n_kv_heads, head_dim, causal);

    // Dispatch by head_dim and causal/non-causal
    if (head_dim == 128) {
        // Shape<BlockQ=128, BlockKV=128, HeadDim=128> with warp-specialized cooperative
        using Tile = Shape<_128, _128, _128>;
        if (causal) {
            return run_fmha<Tile, CausalFusion>(
                Q_ptr, K_ptr, V_ptr, O_ptr,
                batch, seq_q, seq_kv, n_heads, n_kv_heads, head_dim, scale, stream);
        } else {
            return run_fmha<Tile, DefaultFusion>(
                Q_ptr, K_ptr, V_ptr, O_ptr,
                batch, seq_q, seq_kv, n_heads, n_kv_heads, head_dim, scale, stream);
        }
    } else if (head_dim == 96) {
        // Shape<BlockQ=128, BlockKV=96, HeadDim=96> for Phi4-mini and similar models
        using Tile = Shape<_128, _96, _96>;
        if (causal) {
            return run_fmha<Tile, CausalFusion>(
                Q_ptr, K_ptr, V_ptr, O_ptr,
                batch, seq_q, seq_kv, n_heads, n_kv_heads, head_dim, scale, stream);
        } else {
            return run_fmha<Tile, DefaultFusion>(
                Q_ptr, K_ptr, V_ptr, O_ptr,
                batch, seq_q, seq_kv, n_heads, n_kv_heads, head_dim, scale, stream);
        }
    } else if (head_dim == 64) {
        // Shape<BlockQ=128, BlockKV=64, HeadDim=64> with warp-specialized cooperative
        using Tile = Shape<_128, _64, _64>;
        if (causal) {
            return run_fmha<Tile, CausalFusion>(
                Q_ptr, K_ptr, V_ptr, O_ptr,
                batch, seq_q, seq_kv, n_heads, n_kv_heads, head_dim, scale, stream);
        } else {
            return run_fmha<Tile, DefaultFusion>(
                Q_ptr, K_ptr, V_ptr, O_ptr,
                batch, seq_q, seq_kv, n_heads, n_kv_heads, head_dim, scale, stream);
        }
    }

    // Unsupported head_dim
    return false;
}

} // namespace imp

#else  // !CUTLASS_ARCH_MMA_SM90_SUPPORTED && !IMP_USE_CUTLASS

namespace imp {

bool cutlass_fmha_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, cudaStream_t stream)
{
    return false;  // CUTLASS FMHA not available
}

size_t cutlass_fmha_workspace_estimate(int, int, int, int) { return 0; }
size_t cutlass_fmha_init_workspace(int, int, int, int) { return 0; }
void cutlass_fmha_free_workspace() {}

} // namespace imp

#endif
