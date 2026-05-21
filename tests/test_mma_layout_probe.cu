// =============================================================================
// test_mma_layout_probe.cu — empirical verification of sm_120a m16n8k16
// layout + ldmatrix.x4 behavior
// =============================================================================
//
// Two empirical questions for the Track E kernel:
//
// (1) ldmatrix.sync.aligned.x4.m8n8.shared.b16: does it work correctly when
//     all 32 lanes provide the SAME ptr (current Track E Q-load pattern)?
//     Or does PTX require per-lane row pointers?
//
// (2) mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32: what is the actual
//     D-fragment per-lane layout? Standard says frag[0,1] = row(lane/4),
//     frag[2,3] = row(lane/4+8). Earlier Task 13 investigation claimed
//     INVERTED. PR #353 changed back to standard. Both verified on uniform
//     fill which masks errors. We need direct lane-by-lane verification.
//
// Test design:
//   - A = identity-like matrix where A[r,k] = r (each row contains its row index)
//   - B = all-ones matrix B[k,n] = 1
//   - Software-expected D[m,n] = sum_k A[m,k] * B[k,n] = 16 * m
//   - Each lane t reads d[0..3] and we check what row m each maps to.
//
//   Standard PTX m16n8k16 D-frag:
//     d[0] = D[lane/4,     (lane%4)*2]     → 16 * (lane/4)
//     d[1] = D[lane/4,     (lane%4)*2 + 1] → 16 * (lane/4)
//     d[2] = D[lane/4 + 8, (lane%4)*2]     → 16 * (lane/4 + 8)
//     d[3] = D[lane/4 + 8, (lane%4)*2 + 1] → 16 * (lane/4 + 8)
//
//   Inverted hypothesis:
//     d[0] = D[lane/4 + 8, ...] → 16 * (lane/4 + 8)
//     d[2] = D[lane/4,     ...] → 16 * (lane/4)
//
//   Lane 0: standard → (d[0]=0, d[1]=0, d[2]=128, d[3]=128)
//           inverted → (d[0]=128, d[1]=128, d[2]=0, d[3]=0)
//   Lane 4: standard → (d[0]=16, d[1]=16, d[2]=144, d[3]=144)
//           inverted → (d[0]=144, d[1]=144, d[2]=16, d[3]=16)
//
// Also tests A load via ldmatrix.x4 with same-ptr vs per-lane-ptr.
// =============================================================================

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

// ----- PTX helpers (lifted from attention_tiled_streaming.cu) ---------------

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

__device__ __forceinline__ void mma_m16n8k16_f16(
        float (&d)[4],
        const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// ----- Test 1: A=[r], B=[1], verify mma D-layout -----------------------------

// Loads A and B with per-lane row pointers (the "correct" PTX usage).
// Also writes the per-lane ldmatrix A-fragment (as 8 halves per lane) to a_out.
__global__ void probe_mma_layout(float* d_out, uint32_t* a_out) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    if (tid >= 32) return;

    // SMEM: A (16x16 halves) + B (16x8 halves). A row-major, B col-major but
    // we'll store as 16x8 row-major and load with .trans to fit B operand.
    __shared__ __align__(128) __half A[16 * 16];
    __shared__ __align__(128) __half B[16 * 8];

    // Initialize A[r][k] = r (each row contains its row index).
    if (lane == 0) {
        for (int r = 0; r < 16; ++r)
            for (int k = 0; k < 16; ++k)
                A[r * 16 + k] = __float2half(static_cast<float>(r));
        // Initialize B[k][n] = 1 (all ones).
        for (int k = 0; k < 16; ++k)
            for (int n = 0; n < 8; ++n)
                B[k * 8 + n] = __float2half(1.0f);
    }
    __syncwarp();

    // Per-lane row ptrs:
    //   ldmatrix.x4 for A: T0-T7 → rows 0-7, T8-T15 → rows 0-7 (cols 8..15
    //   read by the trans? No, this is non-trans ldmatrix loading A row-major.)
    //
    // Actually for A operand of m16n8k16 (row.col), A is row-major in mma.
    // ldmatrix.x4 loads 4 8x8 tiles. For A (16x16), the 4 tiles are:
    //   tile 0: rows 0..7, cols 0..7
    //   tile 1: rows 0..7, cols 8..15
    //   tile 2: rows 8..15, cols 0..7
    //   tile 3: rows 8..15, cols 8..15
    //
    // Each tile's 8 rows are addressed by 8 lanes:
    //   lanes 0-7  → tile 0 rows 0..7
    //   lanes 8-15 → tile 1 rows 0..7 (but cols 8..15 — same row index!)
    //   lanes 16-23 → tile 2 rows 8..15
    //   lanes 24-31 → tile 3 rows 8..15
    uint32_t a[4];
    {
        int row_in_tile = lane % 8;       // 0..7
        int tile_row_off = (lane / 16) * 8; // 0 or 8 (tiles 0,1 → 0, tiles 2,3 → 8)
        int tile_col_off = ((lane / 8) & 1) * 8;  // tile 0,2 → 0, tile 1,3 → 8
        int row = tile_row_off + row_in_tile;
        int col = tile_col_off;
        __half* ptr = &A[row * 16 + col];
        ldmatrix_x4(a, ptr);
    }

    // For B (16 K × 8 N col-major), use ldmatrix.x4.trans (or non-trans with
    // the right layout). B is stored row-major as B[k][n] in SMEM.
    // mma.col.B = col-major operand. The B-tile is 16 rows × 8 cols. Need
    // ldmatrix.trans to get col-major register packing.
    uint32_t b_full[4];
    {
        // For B 16x8 stored row-major, ldmatrix.x4.trans with per-lane row ptrs
        // gives 4 8x8 sub-tiles. We only need 2 (B is 16x8, not 16x16).
        // Lanes 0-7 provide rows 0..7 of B (the only 8-row 8-col tile we need).
        // Lanes 8-15 provide rows 8..15.
        // Lanes 16-31 duplicate.
        int row_in_tile = lane % 8;
        int tile_row_off = (lane / 8) * 8;  // 0, 8, 16, 24
        if (tile_row_off >= 16) tile_row_off = 0;  // wrap for duplicates
        int row = tile_row_off + row_in_tile;
        __half* ptr = &B[row * 8];
        ldmatrix_x4_trans(b_full, ptr);
    }
    uint32_t b[2] = {b_full[0], b_full[1]};

    // Dump a-frag (ldmatrix output) before mma.
    for (int i = 0; i < 4; ++i) a_out[lane * 4 + i] = a[i];

    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mma_m16n8k16_f16(d, a, b);

    // Output: each lane writes its 4 d values to d_out[lane * 4 + i]
    d_out[lane * 4 + 0] = d[0];
    d_out[lane * 4 + 1] = d[1];
    d_out[lane * 4 + 2] = d[2];
    d_out[lane * 4 + 3] = d[3];
}

// ----- Test 2: ldmatrix.x4 with same-ptr (Track E Q-load pattern) -----------

// Load A with SAME ptr per lane (Track E's current Q-load pattern).
// Print each lane's 4 b32 regs as 8 halves.
__global__ void probe_ldmatrix_same_ptr(uint32_t* d_out) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    if (tid >= 32) return;

    __shared__ __align__(128) __half A[16 * 16];

    // Fill A with A[r][c] = r * 100 + c (so we can identify which row each
    // half came from).
    if (lane == 0) {
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c)
                A[r * 16 + c] = __float2half(static_cast<float>(r * 100 + c));
    }
    __syncwarp();

    // SAME pointer for all lanes.
    __half* ptr = &A[0];
    uint32_t a[4];
    ldmatrix_x4(a, ptr);

    // Store the 4 regs of each lane to d_out.
    for (int i = 0; i < 4; ++i) {
        d_out[lane * 4 + i] = a[i];
    }
}

}  // anonymous namespace

TEST(MmaLayoutProbe, M16N8K16_DFrag) {
    float* d_out = nullptr;
    uint32_t* a_out = nullptr;
    cudaMalloc(&d_out, 32 * 4 * sizeof(float));
    cudaMalloc(&a_out, 32 * 4 * sizeof(uint32_t));
    cudaMemset(d_out, 0, 32 * 4 * sizeof(float));
    cudaMemset(a_out, 0, 32 * 4 * sizeof(uint32_t));

    probe_mma_layout<<<1, 32>>>(d_out, a_out);
    cudaDeviceSynchronize();

    std::vector<float> h(32 * 4);
    cudaMemcpy(h.data(), d_out, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a-frag first.
    std::vector<uint32_t> ha(32 * 4);
    cudaMemcpy(ha.data(), a_out, 32 * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::printf("\n=== ldmatrix A-frag (per-lane), A[r][k]=r ===\n");
    std::printf("Lane | a[0] lo,hi    a[1] lo,hi    a[2] lo,hi    a[3] lo,hi\n");
    for (int lane = 0; lane < 32; ++lane) {
        char buf[256]; int off = 0;
        off += std::snprintf(buf + off, sizeof(buf) - off, "%4d | ", lane);
        for (int i = 0; i < 4; ++i) {
            uint32_t r = ha[lane * 4 + i];
            uint16_t lo = r & 0xFFFF;
            uint16_t hi = (r >> 16) & 0xFFFF;
            __half hlo = *reinterpret_cast<__half*>(&lo);
            __half hhi = *reinterpret_cast<__half*>(&hi);
            off += std::snprintf(buf + off, sizeof(buf) - off, "%3.0f,%3.0f  ",
                                  __half2float(hlo), __half2float(hhi));
        }
        std::printf("%s\n", buf);
    }

    std::printf("\n=== m16n8k16 D-frag layout probe ===\n");
    std::printf("A[r][k] = r (each row = row index)\n");
    std::printf("B[k][n] = 1 (all ones)\n");
    std::printf("Expected D[m][n] = sum_k A[m][k] * B[k][n] = 16 * m (for all n).\n");
    std::printf("Standard layout:   d[0,1]→row(lane/4),   d[2,3]→row(lane/4+8)\n");
    std::printf("Inverted layout:   d[0,1]→row(lane/4+8), d[2,3]→row(lane/4)\n\n");

    std::printf("Lane | d[0]  d[1]  d[2]  d[3]  | infer\n");
    std::printf("-----+------------------------+---------------------\n");
    for (int lane = 0; lane < 32; ++lane) {
        float d0 = h[lane * 4 + 0];
        float d1 = h[lane * 4 + 1];
        float d2 = h[lane * 4 + 2];
        float d3 = h[lane * 4 + 3];

        // Expected row for standard: d[0,1] map to row=lane/4; d[2,3] to row=lane/4+8.
        // Each row m's value should be 16*m.
        int row_lo = lane / 4;
        int row_hi = row_lo + 8;
        int exp_std_lo = 16 * row_lo;       // for d[0,1]
        int exp_std_hi = 16 * row_hi;       // for d[2,3]

        const char* verdict = "?";
        if (d0 == exp_std_lo && d2 == exp_std_hi) {
            verdict = "STANDARD";
        } else if (d0 == exp_std_hi && d2 == exp_std_lo) {
            verdict = "INVERTED";
        } else if (d0 == 0 && d1 == 0 && d2 == 0 && d3 == 0) {
            verdict = "ZERO (mma didn't fire?)";
        }

        std::printf("%4d | %5.0f %5.0f %5.0f %5.0f  | exp_std_lo=%d exp_std_hi=%d → %s\n",
                    lane, d0, d1, d2, d3, exp_std_lo, exp_std_hi, verdict);
    }
    std::fflush(stdout);

    cudaFree(d_out);
    cudaFree(a_out);
}

TEST(MmaLayoutProbe, LdmatrixX4_SamePointer) {
    uint32_t* d_out = nullptr;
    cudaMalloc(&d_out, 32 * 4 * sizeof(uint32_t));
    cudaMemset(d_out, 0, 32 * 4 * sizeof(uint32_t));

    probe_ldmatrix_same_ptr<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();

    std::vector<uint32_t> h(32 * 4);
    cudaMemcpy(h.data(), d_out, 32 * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::printf("\n=== ldmatrix.x4 with same-ptr-per-lane probe ===\n");
    std::printf("A[r][c] = r*100 + c. All 32 lanes pass &A[0].\n");
    std::printf("Each lane's 4 b32 regs decode as 8 halves; we print them.\n\n");

    std::printf("Lane | reg0 lo,hi    reg1 lo,hi    reg2 lo,hi    reg3 lo,hi\n");
    std::printf("-----+-------------+-------------+-------------+-------------\n");
    for (int lane = 0; lane < 32; ++lane) {
        // Each b32 reg holds 2 halves (lo and hi).
        char buf[256];
        int off = 0;
        off += std::snprintf(buf + off, sizeof(buf) - off, "%4d | ", lane);
        for (int i = 0; i < 4; ++i) {
            uint32_t r = h[lane * 4 + i];
            uint16_t lo = r & 0xFFFF;
            uint16_t hi = (r >> 16) & 0xFFFF;
            __half hlo = *reinterpret_cast<__half*>(&lo);
            __half hhi = *reinterpret_cast<__half*>(&hi);
            float flo = __half2float(hlo);
            float fhi = __half2float(hhi);
            off += std::snprintf(buf + off, sizeof(buf) - off, "%4.0f,%4.0f  ",
                                  flo, fhi);
        }
        std::printf("%s\n", buf);
    }
    std::fflush(stdout);

    cudaFree(d_out);
}
