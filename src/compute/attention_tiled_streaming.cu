#include "compute/attention_tiled_streaming.h"
#include "core/logging.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

namespace {

// 1 producer + 7 consumers = 8 warps × 32 threads = 256 threads/CTA.
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kProducerWarp = 0;

// MMA tile dimensions (m16n8k16 FP16).
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;

// Bkv per hd. Br baked into kernel template.
template <int HD>
constexpr int default_Bkv() {
    return (HD <= 128) ? 64 : 32;
}

// Br per hd. Picked in §2 of the spec.
template <int HD>
constexpr int default_Br() {
    if constexpr (HD == 64)  return 128;
    else if constexpr (HD == 96)  return 96;
    else if constexpr (HD == 128) return 64;
    else if constexpr (HD == 256) return 32;
    else if constexpr (HD == 512) return 32;
    else return -1;  // SFINAE-ish: unsupported.
}

// HD chunk size for hd=512 chunked path.
constexpr int kHDChunkBytes = 128 * 2;  // 128 halves = 256 B
constexpr int kHDChunkHalves = 128;

}  // namespace

namespace {

__device__ __forceinline__ void cp_async_16(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(glob));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(s), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(s));
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra DONE_%=;\n"
        "bra WAIT_%=;\n"
        "DONE_%=:\n"
        "}\n"
        :: "r"(s), "r"(phase));
}

// ldmatrix x4 (loads 4 fragments, 16x16 halves, into 4 32-bit regs per lane).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

// ldmatrix x4 with .trans modifier for column-major operand loading.
// Used for B-operand fragments in mma.row.col layout (K in QKᵀ, V in PV).
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

// mma.sync.m16n8k16 FP16 in/out (acc FP32). D += A·B.
__device__ __forceinline__ void mma_m16n8k16_f16(
        float (&d)[4],
        const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ float redux_max_f32(float x) {
    float result;
    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;\n"
                 : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ float redux_add_f32(float x) {
    float result;
    asm volatile("redux.sync.add.f32 %0, %1, 0xffffffff;\n"
                 : "=f"(result) : "f"(x));
    return result;
}

}  // namespace

template <int Br, int HD>
__global__ void __launch_bounds__(kThreads, 1)
attention_tiled_streaming_kernel(
        const __half* __restrict__ Q,
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        __half* __restrict__ O,
        int seq_q, int seq_kv,
        int n_heads, int n_kv_heads,
        float scale, bool causal,
        int sliding_window, float softcap, int q_offset) {
    constexpr int Bkv = default_Bkv<HD>();

    // Suppress unused-parameter warnings for params used in later tasks.
    (void)causal; (void)sliding_window; (void)softcap; (void)q_offset;

    // Block coordinates: x=row-block, y=head, z=batch.
    const int row_block = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int kv_head = head / (n_heads / n_kv_heads);

    const int q_row0 = row_block * Br;
    if (q_row0 >= seq_q) return;

    const int tid = threadIdx.x;

    // ------------------------------------------------------------------
    // Shared memory layout
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) uint8_t smem_raw[];

    __half* Q_smem = reinterpret_cast<__half*>(smem_raw);
    __half* K_smem[2];                          // double-buffered
    K_smem[0] = Q_smem + Br * HD;
    K_smem[1] = K_smem[0] + Bkv * HD;
    __half* V_smem = K_smem[1] + Bkv * HD;
    uint64_t* mbar = reinterpret_cast<uint64_t*>(V_smem + Bkv * HD);

    // mbar layout: [Q_ready, K_ready[0], K_ready[1], V_ready,
    //               QKt_done, V_consumed]
    if (tid == 0) {
        mbar_init(&mbar[0], 1);         // Q_ready
        mbar_init(&mbar[1], 1);         // K_ready[0]
        mbar_init(&mbar[2], 1);         // K_ready[1]
        mbar_init(&mbar[3], 1);         // V_ready
        mbar_init(&mbar[4], 7);         // QKt_done
        mbar_init(&mbar[5], 7);         // V_consumed
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Q load: one-time. All 256 threads cooperate.
    // ------------------------------------------------------------------
    const __half* Q_gmem = Q
        + static_cast<size_t>(batch) * seq_q * n_heads * HD
        + static_cast<size_t>(q_row0) * n_heads * HD
        + static_cast<size_t>(head) * HD;

    constexpr int kHalvesPerChunk = 8;          // 16 bytes per cp.async
    constexpr int kQChunks = (Br * HD) / kHalvesPerChunk;
    for (int c = tid; c < kQChunks; c += kThreads) {
        int elem = c * kHalvesPerChunk;
        int r = elem / HD;
        int d = elem % HD;
        const __half* src = Q_gmem + static_cast<size_t>(r) * n_heads * HD + d;
        cp_async_16(&Q_smem[r * HD + d], src);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    if (tid == 0) mbar_arrive(&mbar[0]);

    // Phase counters per mbarrier (parity-based wait).
    uint32_t phase_K[2] = {0u, 0u};
    uint32_t phase_V = 0u;
    uint32_t phase_QKt = 0u;
    uint32_t phase_VC = 0u;

    const int n_kv_tiles = (seq_kv + Bkv - 1) / Bkv;
    int k_slot = 0;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    // ------------------------------------------------------------------
    // Producer warp (warp 0): cp.async-loads K (double-buffered) + V (single-buffered).
    // ------------------------------------------------------------------
    if (warp_id == kProducerWarp) {
        // Pre-load K[0] into K_smem[0] before the iter loop.
        const __half* K_gmem0 = K
            + static_cast<size_t>(batch) * seq_kv * n_kv_heads * HD
            + static_cast<size_t>(0) * Bkv * n_kv_heads * HD
            + static_cast<size_t>(kv_head) * HD;
        for (int c = lane; c < (Bkv * HD) / kHalvesPerChunk; c += 32) {
            int elem = c * kHalvesPerChunk;
            int r = elem / HD;
            int d = elem % HD;
            cp_async_16(&K_smem[0][r * HD + d],
                         K_gmem0 + static_cast<size_t>(r) * n_kv_heads * HD + d);
        }
        cp_async_commit();
        cp_async_wait_all();
        if (lane == 0) mbar_arrive(&mbar[1]);          // K_ready[0]

        for (int i = 0; i < n_kv_tiles; ++i) {
            // Prefetch K[i+1] into the OTHER slot if not last iter.
            if (i + 1 < n_kv_tiles) {
                int next_slot = 1 - k_slot;
                const __half* K_gmem_next = K
                    + static_cast<size_t>(batch) * seq_kv * n_kv_heads * HD
                    + static_cast<size_t>(i + 1) * Bkv * n_kv_heads * HD
                    + static_cast<size_t>(kv_head) * HD;
                for (int c = lane; c < (Bkv * HD) / kHalvesPerChunk; c += 32) {
                    int elem = c * kHalvesPerChunk;
                    int r = elem / HD;
                    int d = elem % HD;
                    cp_async_16(&K_smem[next_slot][r * HD + d],
                                 K_gmem_next + static_cast<size_t>(r) * n_kv_heads * HD + d);
                }
                cp_async_commit();
                cp_async_wait_all();
                if (lane == 0) mbar_arrive(&mbar[1 + next_slot]);
            }

            // Wait for consumers to finish QKᵀ before loading V[i].
            mbar_wait(&mbar[4], phase_QKt);
            phase_QKt ^= 1u;

            // Load V[i] (single buffer).
            const __half* V_gmem = V
                + static_cast<size_t>(batch) * seq_kv * n_kv_heads * HD
                + static_cast<size_t>(i) * Bkv * n_kv_heads * HD
                + static_cast<size_t>(kv_head) * HD;
            for (int c = lane; c < (Bkv * HD) / kHalvesPerChunk; c += 32) {
                int elem = c * kHalvesPerChunk;
                int r = elem / HD;
                int d = elem % HD;
                cp_async_16(&V_smem[r * HD + d],
                             V_gmem + static_cast<size_t>(r) * n_kv_heads * HD + d);
            }
            cp_async_commit();
            cp_async_wait_all();
            if (lane == 0) mbar_arrive(&mbar[3]);     // V_ready

            // Wait for consumers to finish PV before reusing V buffer next iter.
            mbar_wait(&mbar[5], phase_VC);
            phase_VC ^= 1u;

            k_slot ^= 1;
        }
        return;
    }

    // ------------------------------------------------------------------
    // Consumer warps 1..7: own one row-tile of Q each (warps 1..4 active
    // for Br=64, warps 5..7 are helpers idle until Task 8 softmax helpers).
    // ------------------------------------------------------------------
    const int consumer_id = warp_id - 1;   // 0..6
    const bool is_mma_warp = (consumer_id >= 0 && consumer_id < Br / kMmaM);

    // Per-warp register state — only valid if is_mma_warp.
    float O_frag[HD / kMmaN][4];      // FP32 O accumulator, used in Task 9.
    float row_m[2];                    // per-lane row-max [row_a, row_b], Task 8.
    float row_l[2];                    // per-lane row-sum [row_a, row_b], Task 8.
    if (is_mma_warp) {
        #pragma unroll
        for (int n = 0; n < HD / kMmaN; ++n) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) O_frag[n][k] = 0.0f;
        }
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            row_m[k] = -INFINITY;
            row_l[k] = 0.0f;
        }
    }

    // Wait for Q tile to be ready.
    mbar_wait(&mbar[0], /*phase=*/0u);

    // Load Q fragments into registers (one-time per CTA).
    uint32_t Q_frag[HD / kMmaK][4];   // [k_iter][4 regs]
    if (is_mma_warp) {
        const int row_in_warp_base = consumer_id * kMmaM;
        #pragma unroll
        for (int k_it = 0; k_it < HD / kMmaK; ++k_it) {
            __half* Q_tile_ptr = &Q_smem[row_in_warp_base * HD + k_it * kMmaK];
            ldmatrix_x4(Q_frag[k_it], Q_tile_ptr);
        }
    }

    for (int i = 0; i < n_kv_tiles; ++i) {
        mbar_wait(&mbar[1 + k_slot], phase_K[k_slot]);
        phase_K[k_slot] ^= 1u;

        // ----- QKᵀ -----
        // Each mma m16n8k16 produces a 16×8 tile of S.
        // For Bkv=64 → 8 col-tiles per warp.
        // For HD=128 → 8 k-iters per col-tile.
        float S_frag[64 / kMmaN][4];
        if (is_mma_warp) {
            #pragma unroll
            for (int n_it = 0; n_it < 64 / kMmaN; ++n_it) {
                #pragma unroll
                for (int k = 0; k < 4; ++k) S_frag[n_it][k] = 0.0f;

                #pragma unroll
                for (int k_it = 0; k_it < HD / kMmaK; ++k_it) {
                    // K is laid out [Bkv, HD]; for mma.col we read columns.
                    // K_smem[k_slot] tile: 8 cols at [n_it*8, k_it*16].
                    __half* K_tile_ptr =
                        &K_smem[k_slot][n_it * kMmaN * HD + k_it * kMmaK];
                    uint32_t K_full[4];
                    ldmatrix_x4_trans(K_full, K_tile_ptr);
                    uint32_t K_frag[2] = {K_full[0], K_full[1]};
                    mma_m16n8k16_f16(S_frag[n_it], Q_frag[k_it], K_frag);
                }

                // Scale by 1/sqrt(hd).
                #pragma unroll
                for (int k = 0; k < 4; ++k) S_frag[n_it][k] *= scale;
            }
        }

        if (is_mma_warp) {
            // Online softmax across S_frag[0..Bkv/kMmaN-1][4].
            //
            // m16n8k16 D-fragment per-lane layout:
            //   frag[0] = row(lane/4)      col((lane%4)*2)
            //   frag[1] = row(lane/4)      col((lane%4)*2 + 1)
            //   frag[2] = row(lane/4 + 8)  col((lane%4)*2)
            //   frag[3] = row(lane/4 + 8)  col((lane%4)*2 + 1)
            //
            // Per lane: row_a = lane/4 (frag[0,1]), row_b = lane/4+8 (frag[2,3]).
            // To reduce across col-tiles within a row, shfl_xor across the
            // 4 lanes sharing the same row-pair (offsets 1 and 2 within group-of-4).

            // Compute per-row local max across all Bkv/kMmaN col-tiles.
            float r_max_ab[2] = {-INFINITY, -INFINITY};
            #pragma unroll
            for (int n_it = 0; n_it < Bkv / kMmaN; ++n_it) {
                r_max_ab[0] = fmaxf(r_max_ab[0], fmaxf(S_frag[n_it][0], S_frag[n_it][1]));
                r_max_ab[1] = fmaxf(r_max_ab[1], fmaxf(S_frag[n_it][2], S_frag[n_it][3]));
            }
            // Reduce across 4 lanes sharing the same row-pair.
            #pragma unroll
            for (int off : {1, 2}) {
                r_max_ab[0] = fmaxf(r_max_ab[0], __shfl_xor_sync(0xffffffffu, r_max_ab[0], off));
                r_max_ab[1] = fmaxf(r_max_ab[1], __shfl_xor_sync(0xffffffffu, r_max_ab[1], off));
            }

            // Update running max and compute O rescale factor.
            float new_m[2];
            float scale_prev[2];
            #pragma unroll
            for (int rb = 0; rb < 2; ++rb) {
                new_m[rb]     = fmaxf(row_m[rb], r_max_ab[rb]);
                scale_prev[rb] = __expf(row_m[rb] - new_m[rb]);
                row_m[rb]     = new_m[rb];
            }

            // Apply P = exp(S - new_m) and accumulate r_sum.
            float r_sum[2] = {0.0f, 0.0f};
            #pragma unroll
            for (int n_it = 0; n_it < Bkv / kMmaN; ++n_it) {
                S_frag[n_it][0] = __expf(S_frag[n_it][0] - new_m[0]);
                S_frag[n_it][1] = __expf(S_frag[n_it][1] - new_m[0]);
                S_frag[n_it][2] = __expf(S_frag[n_it][2] - new_m[1]);
                S_frag[n_it][3] = __expf(S_frag[n_it][3] - new_m[1]);
                r_sum[0] += S_frag[n_it][0] + S_frag[n_it][1];
                r_sum[1] += S_frag[n_it][2] + S_frag[n_it][3];
            }
            // Warp-reduce r_sum across the 4-lane row-group.
            #pragma unroll
            for (int off : {1, 2}) {
                r_sum[0] += __shfl_xor_sync(0xffffffffu, r_sum[0], off);
                r_sum[1] += __shfl_xor_sync(0xffffffffu, r_sum[1], off);
            }

            // Update l and rescale O accumulator by exp(prev_m - new_m).
            #pragma unroll
            for (int rb = 0; rb < 2; ++rb) {
                row_l[rb] = scale_prev[rb] * row_l[rb] + r_sum[rb];
            }
            #pragma unroll
            for (int n = 0; n < HD / kMmaN; ++n) {
                O_frag[n][0] *= scale_prev[0];
                O_frag[n][1] *= scale_prev[0];
                O_frag[n][2] *= scale_prev[1];
                O_frag[n][3] *= scale_prev[1];
            }
        }

        if (lane == 0) mbar_arrive(&mbar[4]);
        mbar_wait(&mbar[3], phase_V);
        phase_V ^= 1u;

        if (is_mma_warp) {
            // PV: O += P × V. P is in S_frag (post-softmax in registers).
            // V_smem layout: [Bkv][HD] row-major. Use ldmatrix.trans for col-major B.
            #pragma unroll
            for (int n_it_v = 0; n_it_v < HD / kMmaN; ++n_it_v) {
                #pragma unroll
                for (int k_it_v = 0; k_it_v < Bkv / kMmaK; ++k_it_v) {
                    // Load V_frag (16 K-rows × 8 N-cols of V at (k_it_v*16, n_it_v*8))
                    __half* V_tile_ptr =
                        &V_smem[k_it_v * kMmaK * HD + n_it_v * kMmaN];
                    uint32_t V_full[4];
                    ldmatrix_x4_trans(V_full, V_tile_ptr);
                    uint32_t V_frag[2] = {V_full[0], V_full[1]};

                    // Repack S_frag → P_frag (FP32 → FP16, pair into b32).
                    // A-tile at (warp-row × k_it_v): 16 rows × 16 cols.
                    // S_frag[2*k_it_v + 0] = cols 0..7 of A-tile (8 cols).
                    // S_frag[2*k_it_v + 1] = cols 8..15 of A-tile (8 cols).
                    int sa = 2 * k_it_v + 0;
                    int sb = 2 * k_it_v + 1;
                    __half2 h_row_a_lo = __floats2half2_rn(S_frag[sa][0], S_frag[sa][1]);
                    __half2 h_row_a_hi = __floats2half2_rn(S_frag[sb][0], S_frag[sb][1]);
                    __half2 h_row_b_lo = __floats2half2_rn(S_frag[sa][2], S_frag[sa][3]);
                    __half2 h_row_b_hi = __floats2half2_rn(S_frag[sb][2], S_frag[sb][3]);
                    uint32_t P_frag[4];
                    P_frag[0] = *reinterpret_cast<uint32_t*>(&h_row_a_lo);
                    P_frag[1] = *reinterpret_cast<uint32_t*>(&h_row_a_hi);
                    P_frag[2] = *reinterpret_cast<uint32_t*>(&h_row_b_lo);
                    P_frag[3] = *reinterpret_cast<uint32_t*>(&h_row_b_hi);

                    mma_m16n8k16_f16(O_frag[n_it_v], P_frag, V_frag);
                }
            }
        }

        if (lane == 0) mbar_arrive(&mbar[5]);
        k_slot ^= 1;
    }
    (void)Q_frag;  // used in QKt above.

    // ------------------------------------------------------------------
    // Epilogue: normalize O by 1/row_l, downcast to FP16, write to gmem.
    // ------------------------------------------------------------------
    if (is_mma_warp) {
        // Normalize.
        #pragma unroll
        for (int n = 0; n < HD / kMmaN; ++n) {
            O_frag[n][0] *= (1.0f / row_l[0]);
            O_frag[n][1] *= (1.0f / row_l[0]);
            O_frag[n][2] *= (1.0f / row_l[1]);
            O_frag[n][3] *= (1.0f / row_l[1]);
        }

        // Store: convert each (16-row × 8-col) D-tile to FP16 and write to gmem.
        // m16n8k16 D-fragment layout per lane:
        //   row_a = lane / 4         (covers frag[0], frag[1])
        //   row_b = lane / 4 + 8     (covers frag[2], frag[3])
        //   col_a = (lane % 4) * 2   (frag[0], frag[2])
        //   col_b = col_a + 1        (frag[1], frag[3])
        const int row_in_warp_base = consumer_id * kMmaM;
        const int row_a = lane / 4;
        const int row_b = row_a + 8;
        const int col_a = (lane % 4) * 2;

        #pragma unroll
        for (int n = 0; n < HD / kMmaN; ++n) {
            int col_base = n * kMmaN + col_a;
            int abs_row_a = q_row0 + row_in_warp_base + row_a;
            int abs_row_b = q_row0 + row_in_warp_base + row_b;
            if (abs_row_a < seq_q) {
                __half2 packed = __floats2half2_rn(O_frag[n][0], O_frag[n][1]);
                __half* dst = reinterpret_cast<__half*>(O)
                    + static_cast<size_t>(batch) * seq_q * n_heads * HD
                    + static_cast<size_t>(abs_row_a) * n_heads * HD
                    + static_cast<size_t>(head) * HD
                    + col_base;
                *reinterpret_cast<__half2*>(dst) = packed;
            }
            if (abs_row_b < seq_q) {
                __half2 packed = __floats2half2_rn(O_frag[n][2], O_frag[n][3]);
                __half* dst = reinterpret_cast<__half*>(O)
                    + static_cast<size_t>(batch) * seq_q * n_heads * HD
                    + static_cast<size_t>(abs_row_b) * n_heads * HD
                    + static_cast<size_t>(head) * HD
                    + col_base;
                *reinterpret_cast<__half2*>(dst) = packed;
            }
        }
    }

    (void)scale; (void)Q_gmem;
}

bool attention_tiled_streaming_prefill(const Tensor& Q, const Tensor& K,
                                       const Tensor& V, Tensor& O, float scale,
                                       bool causal, int sliding_window,
                                       float softcap, int q_offset,
                                       cudaStream_t stream) {
    // v1: only hd=128 supported at this task. Other hds bail to cuBLAS.
    if (Q.qtype != QType::F16 || K.qtype != QType::F16 || V.qtype != QType::F16)
        return false;
    if (Q.ndim != 4) return false;
    const int batch = static_cast<int>(Q.shape[0]);
    const int seq_q = static_cast<int>(Q.shape[1]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int seq_kv = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    if (seq_q == 0 || seq_kv == 0) return false;
    if (head_dim != 128) return false;       // expanding in Task 7+

    constexpr int Br = 64;
    constexpr int HD = 128;
    constexpr int Bkv = 64;

    // Smem: Q + K_dbuf + V + 6 mbarriers.
    const size_t smem_bytes =
          Br * HD * sizeof(__half)
        + 2 * Bkv * HD * sizeof(__half)
        + Bkv * HD * sizeof(__half)
        + 6 * sizeof(uint64_t);

    cudaFuncSetAttribute(
        attention_tiled_streaming_kernel<Br, HD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));

    dim3 grid((seq_q + Br - 1) / Br, n_heads, batch);
    attention_tiled_streaming_kernel<Br, HD><<<grid, kThreads, smem_bytes, stream>>>(
        static_cast<const __half*>(Q.data),
        static_cast<const __half*>(K.data),
        static_cast<const __half*>(V.data),
        static_cast<__half*>(O.data),
        seq_q, seq_kv, n_heads, n_kv_heads,
        scale, causal, sliding_window, softcap, q_offset);

    if (cudaGetLastError() != cudaSuccess) return false;
    (void)scale; // referenced for compile, kernel will use later
    return true;
}

}  // namespace imp
