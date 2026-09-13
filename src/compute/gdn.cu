#include "compute/gdn_internal.cuh"
#include "compute/gdn_factor.cuh"
#include "core/logging.h"

#include <cuda_bf16.h>
#include <stdexcept>
#include <string>
#include "core/pdl_device.cuh"
#include "core/pdl.h"

namespace imp {

// Forward declarations
__global__ void gdn_scan_decode_kernel(const float*, const float*, const float*, const half*, const half*,
                                       const float*, const float*, float*, half*, const half*, int, int, int,
                                       int, int);

// Fused multi-token GDN Delta Rule Scan: processes ALL tokens in one launch, recurrent state
// cached in registers (128 floats/thread = 512B), no per-token global round-trips.
// Grid:(n_heads), Block:(head_dim_ssm) (typically 128). Each block owns one head; thread d owns
// H[0..state_size-1,d]. Shared memory: K_norm[SS]+Q_norm[SS]+reduce[block_dim].
// SPLIT: threads cooperating on one state column. SPLIT=1 at SS=128 needs 128 regs for state
// alone and ptxas spills; SPLIT=2 halves the state per thread, partners in adjacent lanes reduce
// via one __shfl_xor_sync. Every HD=SS=128 launch runs SPLIT=2 (kScanSplit128, AUDIT_arch_2026
// A2-3); HD=SS=64 stays at 1.
// StateT: h_state storage type, float (default) or bf16 (gdn.state_bf16, halves state traffic).
// All arithmetic stays FP32 in registers; only loads/stores convert. FP16 state was refuted for
// GDN (subnormal truncation at ~6e-5 breaks near-zero heads); bf16 keeps FP32's range.
template <int HD, int SS, typename YOut, int SPLIT = 1, typename StateT = float>
__global__ void __launch_bounds__(HD * SPLIT, 1) gdn_scan_fused_kernel(
    const float* __restrict__ conv_f32,  // [n_tokens, conv_channels] FP32
    const half* __restrict__ alpha_all,  // [n_tokens, n_heads] FP16
    const half* __restrict__ beta_all,   // [n_tokens, n_heads] FP16
    const float* __restrict__ A_log,     // [n_heads] FP32
    const float* __restrict__ dt_bias,   // [n_heads] FP32
    StateT* __restrict__ h_state,        // [n_heads, SS, HD] in StateT
    YOut* __restrict__ y_out,            // [n_tokens, n_heads * HD] FP16 or FP32
    int n_tokens, int n_heads, int n_groups, int conv_channels, int grouped_layout,
    const int* __restrict__ d_real_n,    // device chunk length (padded verify chunk) or nullptr
    StateT* __restrict__ h_snap,         // second state output, or nullptr
    const int* __restrict__ d_snap_n,    // row count h_snap is taken at
    // Batched decode over independent sequences (#GDN-batch). seq_slots: per-blockIdx.y recurrent-
    // state slot id, nullptr for single-sequence (h_state is then the caller's slot pointer,
    // gridDim.y=1). Sequences are INDEPENDENT and parallelise across blockIdx.y like heads do across
    // blockIdx.x; tokens within one sequence stay sequential (cannot batch).
    const int* __restrict__ seq_slots = nullptr,
    int64_t h_state_seq_stride = 0,
    // Ragged batch (cross-sequence prefill batching, roadmap 0(d)): seq_row_offsets is a prefix sum
    // [n_seq+1] into concatenated row arrays; sequence `seq` owns rows [off[seq],off[seq+1]),
    // n_tokens is ignored per-seq. Uniform batches pass nullptr (byte-for-byte seq*n_tokens rebase).
    // d_real_n must be nullptr when ragged.
    const int* __restrict__ seq_row_offsets = nullptr,
    // Batched speculative verify: out_slots[seq]/snap_slots[seq] are commit/snapshot slot ids into
    // the seq_slots pool. out_slots[seq] receives state at real_n rows (nullptr = in place);
    // snap_slots[seq] receives state at snap_n rows for EVERY sequence (nullptr = the one-slab
    // h_snap contract). In-place snapshot (snap_slots[seq]==seq_slots[seq]) is safe: read precedes write.
    const int* __restrict__ out_slots = nullptr,
    const int* __restrict__ snap_slots = nullptr,
    // Factored spare (gdn_factor.cuh): fac_in carries a pending rank-1 row applied right after the
    // state load, fac_out receives the row for real_n instead of a second full state copy. Both
    // [n_seq,n_heads,fac_stride]; nullptr keeps the full-state contract above.
    float* __restrict__ fac_out = nullptr,
    const float* __restrict__ fac_in = nullptr,
    int fac_stride = 0) {
    const int h = blockIdx.x;
    if (h >= n_heads)
        return;
    pdl_wait();  // every global read below (row offsets, lengths, state, conv rows) may be a predecessor's write
    const int seq = blockIdx.y;
    size_t seq_row0 = static_cast<size_t>(seq) * n_tokens;
    if (seq_row_offsets) {
        seq_row0 = static_cast<size_t>(seq_row_offsets[seq]);
        n_tokens = seq_row_offsets[seq + 1] - seq_row_offsets[seq];
    }
    // SPLIT partners are ADJACENT lanes (d = tid / SPLIT), so the cross-partner
    // reductions below are a warp shuffle rather than a shared-memory round.
    const int d = (SPLIT == 1) ? static_cast<int>(threadIdx.x) : static_cast<int>(threadIdx.x) / SPLIT;
    const int part = (SPLIT == 1) ? 0 : static_cast<int>(threadIdx.x) % SPLIT;
    constexpr int SS_PER = SS / SPLIT;   // state rows this thread owns
    const int s_base = part * SS_PER;
    // Padded verify chunk (#847): y is produced for every row (pads are causally invisible
    // downstream), but the committed h_state is the register snapshot at the real last row - H_reg
    // keeps evolving through pads only to define their (discarded) y values.
    const int real_n = d_real_n ? min(n_tokens, __ldg(d_real_n)) : n_tokens;
    // Second commit row for speculative verify: state as of snap_n rows, written alongside real_n's
    // (0 disables). One slab, not per-sequence: a multi-candidate verify chunk (W sequences, same
    // row-0 token from the same committed state) takes it from sequence 0 only - other sequences
    // would write identical bits, but only one writer keeps the "state after the first row" contract.
    const bool per_seq_snap = seq_slots != nullptr && snap_slots != nullptr && d_snap_n != nullptr;
    const int snap_n = per_seq_snap                    ? min(n_tokens, __ldg(d_snap_n))
                       : (h_snap && d_snap_n && seq == 0) ? min(n_tokens, __ldg(d_snap_n))
                                                          : 0;

    // Head-to-K-group mapping: GGUF tiled layout has head h's group = h % n_groups; HF SafeTensors
    // (Qwen3.5/3.6) grouped layout has group = h / (n_heads/n_groups). grouped_layout=1 selects HF.
    const int g = grouped_layout ? (h / (n_heads / n_groups)) : (h % n_groups);
    const int inner = n_heads * HD;

    // Rebase every per-sequence pointer. With gridDim.y == 1 and no slot table
    // these are all +0.
    StateT* h_out = h_state;       // commit destination (real_n rows)
    StateT* h_snap_dst = h_snap;   // snapshot destination (snap_n rows)
    if (seq_slots) {
        StateT* const pool = h_state;
        h_state = pool + static_cast<size_t>(seq_slots[seq]) * static_cast<size_t>(h_state_seq_stride);
        h_out = out_slots ? pool + static_cast<size_t>(out_slots[seq]) * static_cast<size_t>(h_state_seq_stride)
                          : h_state;
        if (per_seq_snap)
            h_snap_dst = pool + static_cast<size_t>(snap_slots[seq]) * static_cast<size_t>(h_state_seq_stride);
    }
    conv_f32 += seq_row0 * conv_channels;
    alpha_all += seq_row0 * n_heads;
    beta_all += seq_row0 * n_heads;
    y_out += seq_row0 * inner;
    const int BC_size = n_groups * SS;
    const float scale = rsqrtf(static_cast<float>(HD));

    // Load per-head constants
    const float A_h = A_log[h];
    const float dtb_h = dt_bias[h];

    // Load state into registers — the critical optimization.
    // Each thread holds SS floats = one column of H[SS, HD].
    float H_reg[SS_PER];
    {
        const StateT* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
        for (int s = 0; s < SS_PER; s++)
            H_reg[s] = static_cast<float>(H_col[(s_base + s) * HD]);
    }
    // Factor rows are indexed by the recurrent SLOT, not by the batch
    // position: a request keeps its slot across steps but moves within the
    // batch, so indexing by seq would hand the next step another request's row.
    const size_t fac_row = (static_cast<size_t>(seq_slots ? seq_slots[seq] : seq) * n_heads + h) *
                           static_cast<size_t>(fac_stride);
    gdn_factor_apply<SS_PER>(H_reg, fac_in ? fac_in + fac_row : nullptr, SS, s_base, d);

    // Shared memory: K_norm[SS] + Q_norm[SS] + reduce_buf[HD]
    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_q = smem + SS;
    float* s_reduce = smem + 2 * SS;

    // Process each token
    for (int t = 0; t < n_tokens; t++) {
        const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
        const float* Q_g = row + g * SS;
        const float* K_g = row + BC_size + g * SS;
        const float* V_base = row + 2 * BC_size;

        // Load V for this thread's d index
        float v_d = V_base[h * HD + d];

        // Compute alpha → decay gate
        float alpha_h = __half2float(alpha_all[t * n_heads + h]);
        float dt_val = alpha_h + dtb_h;
        dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
        float g_t = expf(fmaxf(A_h * dt_val, -20.0f));

        // Compute beta → learning rate
        float beta_h = __half2float(beta_all[t * n_heads + h]);
        beta_h = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));

        // Parallel L2-normalize K and Q.
        // Each thread loads SS/HD elements and contributes to the reduction.
        // With HD=128 threads and SS=128 elements, each thread loads 1 element.
        {
            // Indexed by the REAL thread id, not by the column: with SPLIT > 1
            // two threads share `d` and would both write s_k[d] and s_reduce[d].
            constexpr int NT = HD * SPLIT;
            const int tid = static_cast<int>(threadIdx.x);
            // Load K and Q into shared memory
            if (tid < SS) {
                s_k[tid] = K_g[tid];
                s_q[tid] = Q_g[tid];
            }
            __syncthreads();

            // Parallel sum-of-squares reduction
            float k_sq = 0.0f, q_sq = 0.0f;
            for (int i = tid; i < SS; i += NT) {
                k_sq += s_k[i] * s_k[i];
                q_sq += s_q[i] * s_q[i];
            }
            // Block reduction for k_sq
            s_reduce[tid] = k_sq;
            __syncthreads();
            for (int stride = NT / 2; stride > 0; stride >>= 1) {
                if (tid < stride)
                    s_reduce[tid] += s_reduce[tid + stride];
                __syncthreads();
            }
            // PyTorch-style L2 norm (matches llama's ggml_l2_norm): rsqrtf(max(sum_sq, eps^2)). Additive
            // eps (sum+eps) over-clamps near-zero heads, breaking Qwen3.6 scan outputs at layers with
            // near-zero K.
            float k_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

            // Barrier required: every thread must finish reading s_reduce[0] (k_inv) before any thread
            // overwrites s_reduce for the q reduction, or a slow warp reads a partially-overwritten value.
            // Latent at grid (n_heads) <= 48 blocks (one SM each, in lockstep); batching decode over
            // sequences makes the grid (n_heads x n_seq), where at higher block counts the race fires and
            // results differ run to run.
            __syncthreads();
            s_reduce[tid] = q_sq;
            __syncthreads();
            for (int stride = NT / 2; stride > 0; stride >>= 1) {
                if (tid < stride)
                    s_reduce[tid] += s_reduce[tid + stride];
                __syncthreads();
            }
            float q_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

            // Normalize in-place
            __syncthreads();
            if (tid < SS) {
                s_k[tid] *= k_inv;
                s_q[tid] *= q_inv;
            }
            __syncthreads();
        }

        // Delta rule scan — all in registers, no global memory access
        // Step 1: kv = H^T @ k_norm (dot product of state column with k)
        float kv_d = 0.0f;
#pragma unroll
        for (int s = 0; s < SS_PER; s++)
            kv_d += H_reg[s] * s_k[s_base + s];
        if constexpr (SPLIT > 1) {
            // Sum the partners' halves. Adjacent lanes, so one shuffle.
#pragma unroll
            for (int off = 1; off < SPLIT; off <<= 1)
                kv_d += __shfl_xor_sync(0xffffffffu, kv_d, off);
        }

        // Step 2: delta = (v - g*kv) * beta
        float delta_d = (v_d - g_t * kv_d) * beta_h;

        // Step 3: Update state + compute output
        float y_partial = 0.0f;
#pragma unroll
        for (int s = 0; s < SS_PER; s++) {
            float h_new = g_t * H_reg[s] + s_k[s_base + s] * delta_d;
            H_reg[s] = h_new;
            y_partial += h_new * s_q[s_base + s];
        }
        if constexpr (SPLIT > 1) {
#pragma unroll
            for (int off = 1; off < SPLIT; off <<= 1)
                y_partial += __shfl_xor_sync(0xffffffffu, y_partial, off);
        }

        if (part == 0) {
            const float out_val = y_partial * scale;
            if constexpr (std::is_same_v<YOut, float>) {
                y_out[t * inner + h * HD + d] = out_val;
            } else {
                y_out[t * inner + h * HD + d] = __float2half(out_val);
            }
        }

        // Commit the state at the real last row (per-thread register column,
        // no barrier needed). For the unpadded case real_n == n_tokens and
        // this is the single end-of-scan store the kernel always did.
        if (t + 1 == real_n) {
            // Factored only when a drafted row exists past the snapshot: otherwise the row would re-describe
            // a transition the snapshot already holds and the next step would apply it twice; the 0 sentinel
            // keeps the buffer defined against a stale row from an earlier step.
            if (fac_out && snap_n >= 1 && real_n > snap_n) {
                gdn_factor_store<SS_PER>(fac_out + fac_row, s_k, SS, s_base, d, part, g_t, delta_d);
            } else {
                if (fac_out && d == 0 && part == 0)
                    fac_out[fac_row] = 0.0f;
                StateT* H_col = h_out + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
                for (int s = 0; s < SS_PER; s++)
                    H_col[(s_base + s) * HD] = static_cast<StateT>(H_reg[s]);
            }
        }
        if (t + 1 == snap_n) {
            StateT* S_col = h_snap_dst + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
            for (int s = 0; s < SS_PER; s++)
                S_col[(s_base + s) * HD] = static_cast<StateT>(H_reg[s]);
        }

        // Sync before next token — the next iteration overwrites s_k/s_q in
        // shared memory. Without this barrier, fast threads can overwrite
        // s_k/s_q while slow threads are still reading them in the loops above.
        if (t + 1 < n_tokens)
            __syncthreads();
    }
    pdl_trigger();  // state committed inside the loop; nothing global remains
}

// Gated-norm family (FP16/FP32 variants) lives in gdn_gated_norm.cu.

// V-head reorder: tiled -> grouped (undoes the GGUF converter reorder for ssm_out). FP32 variant
// for conv1d-SiLU output (= scan V input): used when GGUF stored V in tiled layout but the scan
// kernel reads V[h*HD+d] assuming grouped layout.
__global__ void vhead_tiled_to_grouped_f32_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                                  int n_tokens, int n_heads, int head_dim, int n_groups) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * n_heads * head_dim;
    if (tid >= total)
        return;

    int d = tid % head_dim;
    int h_tiled = (tid / head_dim) % n_heads;
    int t = tid / (n_heads * head_dim);

    int n_v_per_k = n_heads / n_groups;
    int replica = h_tiled / n_groups;
    int group = h_tiled % n_groups;
    int h_grouped = group * n_v_per_k + replica;

    dst[t * n_heads * head_dim + h_grouped * head_dim + d] =
        src[t * n_heads * head_dim + h_tiled * head_dim + d];
}

void vhead_tiled_to_grouped_f32(const float* src, float* dst, int n_tokens, int n_heads, int head_dim,
                                int n_groups, cudaStream_t stream) {
    if (n_heads == n_groups)
        return;
    int total = n_tokens * n_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    vhead_tiled_to_grouped_f32_kernel<<<blocks, threads, 0, stream>>>(src, dst, n_tokens, n_heads, head_dim,
                                                                      n_groups);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------


// Batched fused scan over N independent sequences, n_tokens rows each: lets concurrent GDN
// decode batch. The scan is sequential in tokens (can't parallelise one sequence's timeline),
// but sequences share nothing but weights, so they map to blockIdx.y like heads map to
// blockIdx.x (decode: (n_heads) blocks -> (n_heads x n_seq)).
// h_state_pool: base of the recurrent-state pool; seq_slots[i] selects sequence i's slot;
// h_state_seq_stride is the distance between slots in floats. conv_f32/alpha/beta/y are
// [n_seq*n_tokens,...] sequence-major (the layout a batched decode step already produces).
void gdn_scan_fused_f32_batched(const float* conv_f32, int conv_channels, const half* alpha,
                                const half* beta, const float* A_log, const float* dt_bias,
                                float* h_state_pool, const int* seq_slots, int64_t h_state_seq_stride,
                                half* y, int n_seq, int n_tokens, int n_heads, int head_dim_ssm,
                                int state_size, int n_groups, cudaStream_t stream, int grouped_layout,
                                const int* d_real_n, const int* seq_row_offsets, float* h_snap,
                                const int* d_snap_n, const int* out_slots, const int* snap_slots,
                                float* fac_out, const float* fac_in, int fac_stride) {
    if (n_seq <= 0)
        return;
    if (seq_row_offsets && d_real_n)
        throw std::runtime_error(
            "gdn_scan_fused_f32_batched: d_real_n is a single-sequence contract - nullptr when ragged");
    if (seq_row_offsets && (h_snap || out_slots || snap_slots))
        throw std::runtime_error(
            "gdn_scan_fused_f32_batched: h_snap/out_slots/snap_slots are uniform-batch contracts - nullptr when ragged");
    dim3 grid(n_heads, n_seq);
    if (head_dim_ssm == 128 && state_size == 128) {
        // SPLIT=2 (not 4 or 8): 4/8 give better register pressure/occupancy but measured WORSE
        // throughput - kernel is bandwidth-bound, so occupancy isn't binding; coalescing is. SPLIT
        // threads share one column, so a warp touches 32/SPLIT distinct columns at 128/SPLIT
        // bytes/access. SPLIT=2 is where the register spill is gone and access is still half a cache
        // line. Do not raise it without re-measuring.
        constexpr int SPLIT = 2;
        const size_t smem = (2 * 128 + 128 * SPLIT) * sizeof(float);
        pdl::enable_kernel(gdn_scan_fused_kernel<128, 128, half, SPLIT>);
        pdl::launch(gdn_scan_fused_kernel<128, 128, half, SPLIT>, dim3(grid), dim3(128 * SPLIT), size_t(smem), stream,
            conv_f32, alpha, beta, A_log, dt_bias, h_state_pool, y, n_tokens, n_heads, n_groups,
            conv_channels, grouped_layout, d_real_n, h_snap, d_snap_n, seq_slots, h_state_seq_stride,
            seq_row_offsets, out_slots, snap_slots, fac_out, fac_in, fac_stride);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (head_dim_ssm == 64 && state_size == 64) {
        const size_t smem = (2 * 64 + 64) * sizeof(float);
        pdl::enable_kernel(gdn_scan_fused_kernel<64, 64, half>);
        pdl::launch(gdn_scan_fused_kernel<64, 64, half>, dim3(grid), dim3(64), size_t(smem), stream,
            conv_f32, alpha, beta, A_log, dt_bias, h_state_pool, y, n_tokens, n_heads, n_groups,
            conv_channels, grouped_layout, d_real_n, h_snap, d_snap_n, seq_slots, h_state_seq_stride,
            seq_row_offsets, out_slots, snap_slots, fac_out, fac_in, fac_stride);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        IMP_LOG_ERROR("gdn_scan_fused_f32_batched: unsupported head_dim=%d state_size=%d", head_dim_ssm,
                      state_size);
    }
}

// BF16-state twin (gdn.state_bf16). h_state_seq_stride is in BF16 ELEMENTS.
// HD=SS=128 only — the init resolver refuses BF16 state for other shapes,
// so reaching the else here is a wiring bug, not a serving condition.
void gdn_scan_fused_bf16_batched(const float* conv_f32, int conv_channels, const half* alpha,
                                 const half* beta, const float* A_log, const float* dt_bias,
                                 __nv_bfloat16* h_state_pool, const int* seq_slots,
                                 int64_t h_state_seq_stride, half* y, int n_seq, int n_tokens, int n_heads,
                                 int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                                 int grouped_layout, const int* d_real_n, const int* seq_row_offsets,
                                 __nv_bfloat16* h_snap, const int* d_snap_n, const int* out_slots,
                                 const int* snap_slots, float* fac_out, const float* fac_in,
                                 int fac_stride) {
    if (n_seq <= 0)
        return;
    if (seq_row_offsets && d_real_n)
        throw std::runtime_error(
            "gdn_scan_fused_bf16_batched: d_real_n is a single-sequence contract - nullptr when ragged");
    if (seq_row_offsets && (h_snap || out_slots || snap_slots))
        throw std::runtime_error(
            "gdn_scan_fused_bf16_batched: h_snap/out_slots/snap_slots are uniform-batch contracts - nullptr when ragged");
    if (head_dim_ssm != 128 || state_size != 128)
        throw std::runtime_error("gdn_scan_fused_bf16_batched: no kernel for HD=" +
                                 std::to_string(head_dim_ssm) + " SS=" + std::to_string(state_size));
    dim3 grid(n_heads, n_seq);
    constexpr int SPLIT = 2;
    const size_t smem = (2 * 128 + 128 * SPLIT) * sizeof(float);
    pdl::enable_kernel(gdn_scan_fused_kernel<128, 128, half, SPLIT, __nv_bfloat16>);
        pdl::launch(gdn_scan_fused_kernel<128, 128, half, SPLIT, __nv_bfloat16>, dim3(grid), dim3(128 * SPLIT), size_t(smem), stream,
        conv_f32, alpha, beta, A_log, dt_bias, h_state_pool, y, n_tokens, n_heads, n_groups, conv_channels,
        grouped_layout, d_real_n, h_snap, d_snap_n, seq_slots, h_state_seq_stride, seq_row_offsets, out_slots,
        snap_slots, fac_out, fac_in, fac_stride);
    IMP_CUDA_CHECK_LAUNCH();
}

// Fused scan: all tokens in one launch. conv_f32: [n_tokens,conv_channels] FP32 (Q|K|V
// interleaved per token). grouped_layout: 0=GGUF tiled (g=h%n_groups), 1=HF SafeTensors grouped
// (g=h/n_v_per_k). Single-sequence HD=SS=128 launches run SPLIT=2 (255 regs+88-96B local frame
// at SPLIT=1 vs 180 regs, none, at SPLIT=2); 256 threads, reduce buffer HD*SPLIT floats.
constexpr int kScanSplit128 = 2;
constexpr size_t kScanSmem128 = (2 * 128 + 128 * kScanSplit128) * sizeof(float);

void gdn_scan_fused_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                        const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                        int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                        int grouped_layout, const int* d_real_n) {
    // Shared memory: K_norm[SS] + Q_norm[SS] + reduce[HD]
    size_t smem = (2 * state_size + head_dim_ssm) * sizeof(float);

    // Template dispatch for common sizes
    if (head_dim_ssm == 128 && state_size == 128) {
        pdl::enable_kernel(gdn_scan_fused_kernel<128, 128, half, kScanSplit128>);
        pdl::launch(gdn_scan_fused_kernel<128, 128, half, kScanSplit128>, dim3(n_heads),
                    dim3(128 * kScanSplit128), size_t(kScanSmem128), stream, conv_f32, alpha, beta, A_log,
                    dt_bias, h_state, y, n_tokens, n_heads, n_groups, conv_channels, grouped_layout, d_real_n,
                    static_cast<float*>(nullptr), nullptr, static_cast<const int*>(nullptr), int64_t(0),
                    static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), static_cast<float*>(nullptr), static_cast<const float*>(nullptr), 0);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (head_dim_ssm == 64 && state_size == 64) {
        pdl::enable_kernel(gdn_scan_fused_kernel<64, 64, half>);
        pdl::launch(gdn_scan_fused_kernel<64, 64, half>, dim3(n_heads), dim3(64), size_t(smem), stream, conv_f32, alpha, beta, A_log, dt_bias, h_state, y, n_tokens,
                                            n_heads, n_groups, conv_channels, grouped_layout, d_real_n,
                                            static_cast<float*>(nullptr), nullptr, static_cast<const int*>(nullptr), int64_t(0), static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), static_cast<float*>(nullptr), static_cast<const float*>(nullptr), 0);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        // Fallback: per-token loop (for unsupported HD/SS sizes). The host
        // loop cannot bound itself by a device-side length — reject padded
        // verify chunks loudly instead of advancing state through pad rows.
        if (d_real_n != nullptr)
            throw std::runtime_error("gdn_scan_fused_f32: padded verify chunk unsupported for HD=" +
                                     std::to_string(head_dim_ssm) + " SS=" + std::to_string(state_size));
        int inner = n_heads * head_dim_ssm;
        int BC_size = n_groups * state_size;
        size_t smem_old = 2 * state_size * sizeof(float) + 2 * sizeof(float);
        for (int t = 0; t < n_tokens; t++) {
            const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
            gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem_old, stream>>>(
                row + 2 * BC_size, row + BC_size, row, alpha + t * n_heads, beta + t * n_heads, A_log,
                dt_bias, h_state, y + t * inner, nullptr, n_heads, head_dim_ssm, state_size, n_groups,
                grouped_layout);
            IMP_CUDA_CHECK_LAUNCH();
        }
    }
}

// BF16-state twin of gdn_scan_fused_f32 (prefill / single-sequence path).
// HD=SS=128 only — see the batched twin's comment.
void gdn_scan_fused_bf16(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                         const float* A_log, const float* dt_bias, __nv_bfloat16* h_state, half* y,
                         int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                         cudaStream_t stream, int grouped_layout, const int* d_real_n) {
    if (head_dim_ssm != 128 || state_size != 128)
        throw std::runtime_error("gdn_scan_fused_bf16: no kernel for HD=" + std::to_string(head_dim_ssm) +
                                 " SS=" + std::to_string(state_size));
    pdl::enable_kernel(gdn_scan_fused_kernel<128, 128, half, kScanSplit128, __nv_bfloat16>);
    pdl::launch(gdn_scan_fused_kernel<128, 128, half, kScanSplit128, __nv_bfloat16>, dim3(n_heads),
                dim3(128 * kScanSplit128), size_t(kScanSmem128), stream, conv_f32, alpha, beta, A_log,
                dt_bias, h_state, y, n_tokens, n_heads, n_groups, conv_channels, grouped_layout, d_real_n,
                static_cast<__nv_bfloat16*>(nullptr), nullptr, static_cast<const int*>(nullptr), int64_t(0),
                static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), static_cast<float*>(nullptr), static_cast<const float*>(nullptr), 0);
    IMP_CUDA_CHECK_LAUNCH();
}

// FP32-output variant — writes scan result as FP32 for downstream
// FP32-input RMSNorm+Gate (avoids FP16 subnormal-truncation at ~6e-5).
void gdn_scan_fused_fp32out(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, float* y_fp32,
                            int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                            cudaStream_t stream, int grouped_layout, const int* d_real_n, float* h_snap,
                            const int* d_snap_n) {
    size_t smem = (2 * state_size + head_dim_ssm) * sizeof(float);
    if (head_dim_ssm == 128 && state_size == 128) {
        pdl::enable_kernel(gdn_scan_fused_kernel<128, 128, float, kScanSplit128>);
        pdl::launch(gdn_scan_fused_kernel<128, 128, float, kScanSplit128>, dim3(n_heads),
                    dim3(128 * kScanSplit128), size_t(kScanSmem128), stream, conv_f32, alpha, beta, A_log,
                    dt_bias, h_state, y_fp32, n_tokens, n_heads, n_groups, conv_channels, grouped_layout,
                    d_real_n, h_snap, d_snap_n, static_cast<const int*>(nullptr), int64_t(0),
                    static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), static_cast<float*>(nullptr), static_cast<const float*>(nullptr), 0);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (head_dim_ssm == 64 && state_size == 64) {
        pdl::enable_kernel(gdn_scan_fused_kernel<64, 64, float>);
        pdl::launch(gdn_scan_fused_kernel<64, 64, float>, dim3(n_heads), dim3(64), size_t(smem), stream, conv_f32, alpha, beta, A_log, dt_bias, h_state, y_fp32, n_tokens,
                                            n_heads, n_groups, conv_channels, grouped_layout, d_real_n,
                                            h_snap, d_snap_n, static_cast<const int*>(nullptr), int64_t(0), static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), static_cast<float*>(nullptr), static_cast<const float*>(nullptr), 0);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        // Refuse, do not approximate: an unsupported HD throws immediately rather than silently no-op
        // (same failure shape as #654 - an unchecked fallback that keeps serving). Every GDN checkpoint
        // today is HD=SS=128 (Qwen3.5/3.6/3.8), so nothing reaches this path; other Mamba2 shapes
        // (Nemotron-3.5: head_dim 64, state 128) use the separate Mamba2 scan, not this one.
        throw std::runtime_error("gdn_scan_fused_fp32out: no kernel for HD=" + std::to_string(head_dim_ssm) +
                                 " SS=" + std::to_string(state_size) +
                                 " (supported: 128/128 and 64/64). The FP16 decode fallback never wrote "
                                 "its FP32 output, so serving this shape would return a scan that did "
                                 "nothing.");
    }
}

// BF16-state twin of gdn_scan_fused_fp32out (the gdn.fp32_scan combo).
// HD=SS=128 only — see the batched twin's comment.
void gdn_scan_fused_fp32out_bf16(const float* conv_f32, int conv_channels, const half* alpha,
                                 const half* beta, const float* A_log, const float* dt_bias,
                                 __nv_bfloat16* h_state, float* y_fp32, int n_tokens, int n_heads,
                                 int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                                 int grouped_layout, const int* d_real_n, __nv_bfloat16* h_snap,
                                 const int* d_snap_n) {
    if (head_dim_ssm != 128 || state_size != 128)
        throw std::runtime_error("gdn_scan_fused_fp32out_bf16: no kernel for HD=" +
                                 std::to_string(head_dim_ssm) + " SS=" + std::to_string(state_size));
    pdl::enable_kernel(gdn_scan_fused_kernel<128, 128, float, kScanSplit128, __nv_bfloat16>);
    pdl::launch(gdn_scan_fused_kernel<128, 128, float, kScanSplit128, __nv_bfloat16>, dim3(n_heads),
                dim3(128 * kScanSplit128), size_t(kScanSmem128), stream, conv_f32, alpha, beta, A_log,
                dt_bias, h_state, y_fp32, n_tokens, n_heads, n_groups, conv_channels, grouped_layout,
                d_real_n, h_snap, d_snap_n, static_cast<const int*>(nullptr), int64_t(0),
                static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), static_cast<const int*>(nullptr), static_cast<float*>(nullptr), static_cast<const float*>(nullptr), 0);
    IMP_CUDA_CHECK_LAUNCH();
}

// Reference scan kernel: unfused semantics for validation. One block/v_head, block size =
// head_dim_v; state kept in SHARED memory (not registers) for the token loop, written back at
// the end. Math is identical to the fused kernel - if outputs differ, the fused kernel has a
// correctness bug (register lifetime, sync, or dataflow).
__global__ void gdn_scan_reference_kernel(
    const float* __restrict__ conv_f32,  // [n_tokens, conv_channels] FP32
    const half* __restrict__ alpha_all,  // [n_tokens, n_heads] FP16
    const half* __restrict__ beta_all,   // [n_tokens, n_heads] FP16
    const float* __restrict__ A_log,     // [n_heads] FP32
    const float* __restrict__ dt_bias,   // [n_heads] FP32
    float* __restrict__ h_state,         // [n_heads, state_size, head_dim_v] FP32
    half* __restrict__ y_out,            // [n_tokens, n_heads * head_dim_v] FP16
    int n_tokens, int n_heads, int n_groups, int head_dim_v, int state_size, int conv_channels,
    int grouped_layout, const int* __restrict__ d_real_n) {
    const int h = blockIdx.x;
    if (h >= n_heads)
        return;
    const int d = threadIdx.x;  // [0, head_dim_v)
    const int SS = state_size;
    const int HD = head_dim_v;
    // Padded verify chunk (#847): commit the state at the real last row; pads
    // only define their (discarded) y rows. See gdn_scan_fused_kernel.
    const int real_n = d_real_n ? min(n_tokens, __ldg(d_real_n)) : n_tokens;

    const int g = grouped_layout ? (h / (n_heads / n_groups)) : (h % n_groups);
    const int inner = n_heads * HD;
    const int BC_size = n_groups * SS;

    const float A_h = A_log[h];
    const float dtb_h = dt_bias[h];

    // Shared memory: s_H[SS*HD] state slab, s_k[SS]/s_q[SS] K/Q post-L2-norm, s_v[HD] V,
    // s_reduce[HD] block reduction scratch (all for this block's head, this token).
    extern __shared__ float smem[];
    float* s_H = smem;
    float* s_k = s_H + SS * HD;
    float* s_q = s_k + SS;
    float* s_v = s_q + SS;
    float* s_reduce = s_v + HD;

    // Load state from global into shared.
    {
        const float* H_src = h_state + static_cast<size_t>(h) * SS * HD;
        for (int idx = d; idx < SS * HD; idx += HD) {
            s_H[idx] = H_src[idx];
        }
    }
    __syncthreads();

    for (int t = 0; t < n_tokens; t++) {
        const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
        const float* Q_g = row + g * SS;
        const float* K_g = row + BC_size + g * SS;
        const float* V_base = row + 2 * BC_size + h * HD;

        // Load V (one element per thread)
        s_v[d] = V_base[d];

        // Load Q, K into shared (only first SS threads)
        if (d < SS) {
            s_q[d] = Q_g[d];
            s_k[d] = K_g[d];
        }
        __syncthreads();

        // Per-head scalars
        float alpha_h = __half2float(alpha_all[t * n_heads + h]);
        float dt_val = alpha_h + dtb_h;
        dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
        float g_t = expf(fmaxf(A_h * dt_val, -20.0f));

        float beta_h = __half2float(beta_all[t * n_heads + h]);
        beta_h = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));

        // L2-normalize Q, K across state_size.
        // Each thread contributes one element (if d < SS) to sum-of-squares.
        float k_sq = (d < SS) ? s_k[d] * s_k[d] : 0.0f;
        float q_sq = (d < SS) ? s_q[d] * s_q[d] : 0.0f;

        s_reduce[d] = k_sq;
        __syncthreads();
        for (int stride = HD / 2; stride > 0; stride >>= 1) {
            if (d < stride)
                s_reduce[d] += s_reduce[d + stride];
            __syncthreads();
        }
        // PyTorch-style L2 norm (see note in fused kernel above).
        float k_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

        s_reduce[d] = q_sq;
        __syncthreads();
        for (int stride = HD / 2; stride > 0; stride >>= 1) {
            if (d < stride)
                s_reduce[d] += s_reduce[d + stride];
            __syncthreads();
        }
        float q_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

        if (d < SS) {
            s_k[d] *= k_inv;
            s_q[d] *= q_inv;
        }
        __syncthreads();

        // Delta rule scan for this token: thread d owns column d of the state, s_H[s*HD+d] for s in
        // [0,SS). kv[d] = sum_s H[s,d] * k_norm[s].
        float kv_d = 0.0f;
        for (int s = 0; s < SS; s++) {
            kv_d += s_H[s * HD + d] * s_k[s];
        }

        // delta[d] = (v[d] - g*kv[d]) * beta
        float delta_d = (s_v[d] - g_t * kv_d) * beta_h;

        // H_new[s, d] = g * H[s, d] + k_norm[s] * delta[d]
        // y[d] = sum_s H_new[s, d] * q_norm[s]
        float y_partial = 0.0f;
        for (int s = 0; s < SS; s++) {
            float h_new = g_t * s_H[s * HD + d] + s_k[s] * delta_d;
            s_H[s * HD + d] = h_new;
            y_partial += h_new * s_q[s];
        }

        y_out[t * inner + h * HD + d] = __float2half(y_partial * rsqrtf(static_cast<float>(HD)));

        // Commit the state at the real last row. Each thread copies exactly
        // the column (d, d+HD, ...) it wrote above — no barrier needed.
        if (t + 1 == real_n) {
            float* H_dst = h_state + static_cast<size_t>(h) * SS * HD;
            for (int idx = d; idx < SS * HD; idx += HD) {
                H_dst[idx] = s_H[idx];
            }
        }

        __syncthreads();  // before next token overwrites s_k/s_q/s_v
    }
}

void gdn_scan_reference_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                            int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                            int grouped_layout, const int* d_real_n) {
    // Shared memory: state slab [SS*HD] + K[SS] + Q[SS] + V[HD] + reduce[HD]
    // For Qwen 3.6 (SS=128, HD=128): ~66 KB — exceeds 48 KB default per block,
    // needs the opt-in dynamic-shared attribute.
    size_t smem = (state_size * head_dim_ssm + 2 * state_size + 2 * head_dim_ssm) * sizeof(float);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(&gdn_scan_reference_kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        attr_set = true;
    }
    gdn_scan_reference_kernel<<<n_heads, head_dim_ssm, smem, stream>>>(conv_f32, alpha, beta, A_log, dt_bias,
                                                                       h_state, y, n_tokens, n_heads,
                                                                       n_groups, head_dim_ssm, state_size,
                                                                       conv_channels, grouped_layout,
                                                                       d_real_n);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Legacy interfaces (kept for backward compatibility)
// ---------------------------------------------------------------------------

// Old per-token decode kernel (still available for reference)
__global__ void gdn_scan_decode_kernel(const float* __restrict__ x, const float* __restrict__ B_in,
                                       const float* __restrict__ C_in, const half* __restrict__ alpha_raw,
                                       const half* __restrict__ beta_raw, const float* __restrict__ A_log,
                                       const float* __restrict__ dt_bias, float* __restrict__ h_state,
                                       half* __restrict__ y, const half* __restrict__ z, int n_heads,
                                       int head_dim_ssm, int state_size, int n_groups, int grouped_layout) {
    const int h = blockIdx.x;
    if (h >= n_heads)
        return;
    const int d = threadIdx.x;
    if (d >= head_dim_ssm)
        return;

    const int g = grouped_layout ? (h / (n_heads / n_groups)) : (h % n_groups);
    float* H = h_state + static_cast<size_t>(h) * state_size * head_dim_ssm;
    const float* K_g = B_in + g * state_size;
    const float* Q_g = C_in + g * state_size;

    float v_d = x[h * head_dim_ssm + d];
    float alpha_h = __half2float(alpha_raw[h]);
    float dt_val = alpha_h + dt_bias[h];
    dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
    float g_t = expf(fmaxf(A_log[h] * dt_val, -20.0f));

    float beta_h = __half2float(beta_raw[h]);
    beta_h = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));

    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_q = smem + state_size;
    __shared__ float s_k_inv, s_q_inv;
    if (d == 0) {
        float k_sq = 0.0f, q_sq = 0.0f;
        for (int s = 0; s < state_size; s++) {
            float ks = K_g[s];
            float qs = Q_g[s];
            s_k[s] = ks;
            s_q[s] = qs;
            k_sq += ks * ks;
            q_sq += qs * qs;
        }
        // PyTorch-style L2 norm: rsqrtf(max(sum_sq, eps^2)), matches llama's ggml_l2_norm.
        s_k_inv = rsqrtf(fmaxf(k_sq, 1e-12f));
        s_q_inv = rsqrtf(fmaxf(q_sq, 1e-12f));
        for (int s = 0; s < state_size; s++) {
            s_k[s] *= s_k_inv;
            s_q[s] *= s_q_inv;
        }
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(head_dim_ssm));
    float kv_d = 0.0f;
    for (int s = 0; s < state_size; s++)
        kv_d += H[s * head_dim_ssm + d] * s_k[s];
    float delta_d = (v_d - g_t * kv_d) * beta_h;
    float y_partial = 0.0f;
    for (int s = 0; s < state_size; s++) {
        float h_new = g_t * H[s * head_dim_ssm + d] + s_k[s] * delta_d;
        H[s * head_dim_ssm + d] = h_new;
        y_partial += h_new * s_q[s];
    }
    y[h * head_dim_ssm + d] = __float2half(y_partial * scale);
}

void gdn_scan_decode_f32(const float* x, const float* B, const float* C, const half* alpha, const half* beta,
                         const float* A_log, const float* dt_bias, float* h_state, half* y, const half* z,
                         int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                         int grouped_layout) {
    size_t smem = 2 * state_size * sizeof(float) + 2 * sizeof(float);
    gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem, stream>>>(x, B, C, alpha, beta, A_log, dt_bias,
                                                                    h_state, y, z, n_heads, head_dim_ssm,
                                                                    state_size, n_groups, grouped_layout);
    IMP_CUDA_CHECK_LAUNCH();
}

void gdn_scan_prefill_f32(const float* x, const float* B, const float* C, const half* alpha, const half* beta,
                          const float* A_log, const float* dt_bias, float* h_state, half* y, const half* z,
                          int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                          cudaStream_t stream, int grouped_layout) {
    int inner = n_heads * head_dim_ssm;
    int BC_size = n_groups * state_size;
    size_t smem = 2 * state_size * sizeof(float) + 2 * sizeof(float);
    for (int t = 0; t < n_tokens; t++) {
        gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem, stream>>>(x + t * inner, B + t * BC_size,
                                                                        C + t * BC_size, alpha + t * n_heads,
                                                                        beta + t * n_heads, A_log, dt_bias,
                                                                        h_state, y + t * inner, nullptr,
                                                                        n_heads, head_dim_ssm, state_size,
                                                                        n_groups, grouped_layout);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

}  // namespace imp
