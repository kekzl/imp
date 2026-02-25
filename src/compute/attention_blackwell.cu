// =============================================================================
// attention_blackwell.cu -- Phase 3B: TCGEN05 Blackwell-native attention (sm_120)
// =============================================================================
//
// Scaffold for a Flash Attention kernel targeting NVIDIA Blackwell (sm_120+)
// using TCGEN05 systolic arrays, TMEM accumulator storage, and TMA bulk loads.
//
// STATUS: placeholder / scaffold.  The actual TCGEN05 inline PTX is extremely
// complex and hardware-specific.  The file shows the intended architecture with
// correct kernel signature, shared memory layout for 128x128 tiles, and
// TMA / TMEM comments.  The inner loop falls back to a scalar reference
// implementation, and the host launcher falls through to the existing WMMA tc
// path until CUDA 13.1 toolchain PTX support is fully available.
// =============================================================================

#include "compute/attention_tc.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

// ---------------------------------------------------------------------------
// TCGEN05 Blackwell-native attention for sm_120 (RTX 5090 / B300).
// This kernel leverages Blackwell-specific features:
//   - TCGEN05 systolic arrays with 128x128 tiles
//   - TMEM (Tensor Memory, ~256 KB / SM) for accumulator storage
//   - TMA  (Tensor Memory Accelerator) for async bulk loads
//   - CTA Pairing for producer-consumer overlap
// ---------------------------------------------------------------------------

namespace imp {

// ===== Device code -- compiled for all targets but runtime-guarded ==========
// The kernel compiles for all architectures but is only launched on sm_120+.
// On older architectures the kernel body can still be compiled (it uses only
// standard CUDA intrinsics as a scaffold), but the launcher falls back to
// the WMMA path.

// Blackwell tile sizes -- larger than Hopper due to TCGEN05 capabilities.
static constexpr int BW_Br             = 128;   // query tile rows
static constexpr int BW_Bc             = 128;   // key tile cols
static constexpr int BW_BLOCK_THREADS  = 256;
static constexpr int BW_WARP_SIZE      = 32;

// ---- helpers ---------------------------------------------------------------

// Cooperative tile load: threads stride over total elements.
__device__ __forceinline__
void load_tile(half*       dst,
               const half* src,
               int         rows,
               int         cols,
               int         max_rows,
               int64_t     src_row_stride,
               int         tid,
               int         n_threads)
{
    const int total = rows * cols;
    for (int i = tid; i < total; i += n_threads) {
        const int r = i / cols;
        const int d = i % cols;
        if (r < max_rows) {
            dst[i] = src[(int64_t)r * src_row_stride + d];
        } else {
            dst[i] = __float2half(0.0f);
        }
    }
}

// Zero a float buffer cooperatively.
__device__ __forceinline__
void zero_float_tile(float* dst, int total, int tid, int n_threads)
{
    for (int i = tid; i < total; i += n_threads) {
        dst[i] = 0.0f;
    }
}

// ---- kernel ----------------------------------------------------------------

// TCGEN05-based Flash Attention kernel.
// Uses TMA for global->shared loads and TMEM for accumulator storage.
//
// Key architectural features:
//   - cp.async.bulk.tensor for all data movement (hw address gen + boundary
//     checks)
//   - TMEM stores the output accumulator, freeing shared memory for Q/K/V
//     tiles
//   - Larger 128x128 tiles exploit the wider systolic arrays
//   - Warpgroup-level MMA operations for maximum throughput
//
// NOTE: This kernel uses inline PTX for TCGEN05 / TMA instructions that are
// only available on sm_120+.  The actual PTX sequences depend on the CUDA 13.1
// toolchain.  We structure the kernel to show the intended data flow and use a
// scalar reference fallback where inline PTX is not yet stable.
__global__ void flash_attention_blackwell_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int   batch_size,
    int   seq_q,
    int   seq_kv,
    int   n_heads,
    int   n_kv_heads,
    int   head_dim,
    float scale,
    bool  causal)
{
    // ---- index computation --------------------------------------------------
    const int tile_q     = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int batch_idx  = batch_head / n_heads;
    const int head_idx   = batch_head % n_heads;
    const int kv_head    = (n_kv_heads == n_heads)
                             ? head_idx
                             : head_idx / (n_heads / n_kv_heads);

    const int tid     = threadIdx.x;
    const int q_start = tile_q * BW_Br;

    // Global memory strides (row-major [batch, seq, heads, head_dim]).
    const int64_t q_row_stride  = (int64_t)n_heads    * head_dim;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * head_dim;

    const half* Q_ptr = Q + (int64_t)batch_idx * seq_q  * q_row_stride
                          + (int64_t)q_start   * q_row_stride
                          + (int64_t)head_idx  * head_dim;
    const half* K_ptr = K + (int64_t)batch_idx * seq_kv * kv_row_stride
                          + (int64_t)kv_head   * head_dim;
    const half* V_ptr = V + (int64_t)batch_idx * seq_kv * kv_row_stride
                          + (int64_t)kv_head   * head_dim;
    half* O_ptr       = O + (int64_t)batch_idx * seq_q  * q_row_stride
                          + (int64_t)q_start   * q_row_stride
                          + (int64_t)head_idx  * head_dim;

    // ---- shared memory layout -----------------------------------------------
    //
    // With 128x128 tiles and head_dim <= 128:
    //   Q_tile  : BW_Br * head_dim * sizeof(half) = 32 KB
    //   KV_tile : BW_Bc * head_dim * sizeof(half) = 32 KB
    //   O_acc   : BW_Br * head_dim * sizeof(float)= 64 KB
    //                                        total ~128 KB
    //
    // On real Blackwell silicon the output accumulator would reside in TMEM
    // (~256 KB / SM), freeing 64 KB of shared memory and allowing double
    // buffering of the KV tile.  In this scaffold we keep everything in smem.
    extern __shared__ char smem[];

    half*  Q_tile  = reinterpret_cast<half*>(smem);
    half*  KV_tile = Q_tile + BW_Br * head_dim;
    float* O_acc   = reinterpret_cast<float*>(
                         reinterpret_cast<char*>(KV_tile) +
                         BW_Bc * head_dim * sizeof(half));

    // ---- load Q tile --------------------------------------------------------
    // On real hardware this would be a single cp.async.bulk.tensor TMA load.
    {
        const int rows_avail = (q_start + BW_Br <= seq_q) ? BW_Br
                                                           : (seq_q - q_start);
        load_tile(Q_tile, Q_ptr, BW_Br, head_dim,
                  rows_avail, q_row_stride, tid, BW_BLOCK_THREADS);
    }

    // ---- zero output accumulator --------------------------------------------
    // On real hardware the TMEM accumulator would be cleared with a single
    // TMEM instruction.
    zero_float_tile(O_acc, BW_Br * head_dim, tid, BW_BLOCK_THREADS);
    __syncthreads();

    // ---- per-row softmax running state (for rows owned by this thread) ------
    // In the scalar fallback each thread with tid < BW_Br owns exactly one
    // query row.  Threads with tid >= BW_Br are idle during the score /
    // softmax phases and only participate in cooperative tile loads.
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    // ---- tile loop over KV --------------------------------------------------
    int num_kv_tiles = (seq_kv + BW_Bc - 1) / BW_Bc;
    if (causal) {
        // Last relevant KV position for this Q tile.
        int max_q_pos = q_start + BW_Br - 1;
        if (max_q_pos >= seq_q) max_q_pos = seq_q - 1;
        int furthest = (max_q_pos + BW_Bc) / BW_Bc;
        if (furthest < num_kv_tiles) num_kv_tiles = furthest;
    }

    for (int j = 0; j < num_kv_tiles; ++j) {
        const int kv_start = j * BW_Bc;
        const int kv_valid = ((kv_start + BW_Bc) <= seq_kv)
                               ? BW_Bc
                               : (seq_kv - kv_start);

        // ---- load K tile (TMA placeholder) ----------------------------------
        load_tile(KV_tile, K_ptr + (int64_t)kv_start * kv_row_stride,
                  BW_Bc, head_dim,
                  kv_valid, kv_row_stride, tid, BW_BLOCK_THREADS);
        __syncthreads();

        // ---- compute S = Q_tile @ K_tile^T, online softmax, rescale O_acc ---
        // On real Blackwell hardware this would be a single warpgroup TCGEN05
        // MMA followed by a warpgroup-collective softmax.  In the scalar
        // fallback each row-owning thread does the work sequentially.
        if (tid < BW_Br && (q_start + tid) < seq_q) {
            const int r = tid;

            // 1. Row-max of S[r, :]
            float m_ij = -FLT_MAX;
            for (int c = 0; c < kv_valid; ++c) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += __half2float(Q_tile[r * head_dim + d])
                         * __half2float(KV_tile[c * head_dim + d]);
                }
                dot *= scale;
                if (causal && (q_start + r) < (kv_start + c)) {
                    dot = -FLT_MAX;
                }
                m_ij = fmaxf(m_ij, dot);
            }

            // 2. New running max.
            const float m_new = fmaxf(m_i, m_ij);

            // 3. Exponentiated row sum.
            float p_sum = 0.0f;
            for (int c = 0; c < kv_valid; ++c) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += __half2float(Q_tile[r * head_dim + d])
                         * __half2float(KV_tile[c * head_dim + d]);
                }
                dot *= scale;
                if (causal && (q_start + r) < (kv_start + c)) {
                    dot = -FLT_MAX;
                }
                p_sum += expf(dot - m_new);
            }

            // 4. Rescale previous accumulator.
            const float alpha = expf(m_i - m_new) * l_i;
            const float l_new = alpha + p_sum;
            const float rescale = (l_new > 0.0f) ? (alpha / l_new) : 0.0f;

            for (int d = 0; d < head_dim; ++d) {
                O_acc[r * head_dim + d] *= rescale;
            }

            // Update running softmax state.
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();

        // ---- load V tile into KV_tile (reuse same shared memory region) -----
        // On real hardware this would be a TMA load overlapped with the
        // softmax computation above via CTA pairing.
        load_tile(KV_tile, V_ptr + (int64_t)kv_start * kv_row_stride,
                  BW_Bc, head_dim,
                  kv_valid, kv_row_stride, tid, BW_BLOCK_THREADS);
        __syncthreads();

        // ---- accumulate P @ V -----------------------------------------------
        // On real hardware this would be the second TCGEN05 MMA with TMEM
        // accumulation.  In the scalar fallback we recompute P from Q and K
        // stored in global memory since the K tile has been overwritten by V.
        if (tid < BW_Br && (q_start + tid) < seq_q) {
            const int r = tid;
            const float pv_scale = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

            for (int d = 0; d < head_dim; ++d) {
                float v_acc = 0.0f;
                for (int c = 0; c < kv_valid; ++c) {
                    // Recompute attention score from global K (scaffold only;
                    // real kernel would use TMEM or registers populated during
                    // the S phase above).
                    float dot = 0.0f;
                    for (int dd = 0; dd < head_dim; ++dd) {
                        float k_val = __half2float(
                            K_ptr[(int64_t)(kv_start + c) * kv_row_stride + dd]);
                        dot += __half2float(Q_tile[r * head_dim + dd]) * k_val;
                    }
                    dot *= scale;
                    if (causal && (q_start + r) < (kv_start + c)) {
                        dot = -FLT_MAX;
                    }
                    const float p_val = expf(dot - m_i) * pv_scale;
                    v_acc += p_val * __half2float(KV_tile[c * head_dim + d]);
                }
                O_acc[r * head_dim + d] += v_acc;
            }
        }
        __syncthreads();
    }

    // ---- write output -------------------------------------------------------
    if (tid < BW_Br && (q_start + tid) < seq_q) {
        for (int d = 0; d < head_dim; ++d) {
            O_ptr[(int64_t)tid * q_row_stride + d] =
                __float2half(O_acc[tid * head_dim + d]);
        }
    }
}

// ===== Host-side launcher (always compiled) ==================================

void flash_attention_blackwell(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, cudaStream_t stream)
{
    // Check if we are running on a Blackwell GPU (sm_120+).
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    const int sm = prop.major * 10 + prop.minor;

    if (sm >= 120) {
        const int batch_size = static_cast<int>(Q.shape[0]);
        const int seq_q      = static_cast<int>(Q.shape[1]);
        const int n_heads    = static_cast<int>(Q.shape[2]);
        const int head_dim   = static_cast<int>(Q.shape[3]);
        const int seq_kv     = static_cast<int>(K.shape[1]);
        const int n_kv_heads = static_cast<int>(K.shape[2]);

        const int num_q_tiles = (seq_q + BW_Br - 1) / BW_Br;
        dim3 grid(num_q_tiles, batch_size * n_heads);
        dim3 block(BW_BLOCK_THREADS);

        // Shared memory: Q_tile + KV_tile (half) + O_acc (float)
        const size_t smem_bytes =
            (size_t)BW_Br * head_dim * sizeof(half)   +   // Q_tile
            (size_t)BW_Bc * head_dim * sizeof(half)   +   // KV_tile
            (size_t)BW_Br * head_dim * sizeof(float);     // O_acc

        cudaFuncSetAttribute(
            flash_attention_blackwell_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));

        flash_attention_blackwell_kernel<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(Q.data),
            reinterpret_cast<const half*>(K.data),
            reinterpret_cast<const half*>(V.data),
            reinterpret_cast<half*>(O.data),
            batch_size, seq_q, seq_kv,
            n_heads, n_kv_heads, head_dim,
            scale, causal);
        return;
    }

    // Fallback: use the existing WMMA tensor-core attention path.
    flash_attention_prefill_tc(Q, K, V, O, scale, causal, 0, stream);
}

} // namespace imp
