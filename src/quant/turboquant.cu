#include "quant/turboquant.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace imp {

// ---------------------------------------------------------------------------
// QJL projection matrix generation kernel
//
// Generates a Rademacher random matrix (±1 entries) using Philox PRNG.
// Each thread generates one row of the matrix (sketch_dim rows total).
// The signs are packed as bits: bit j in byte j/8 = sign of R[row][j].
// ---------------------------------------------------------------------------

__global__ void qjl_generate_matrix_kernel(
    uint8_t* __restrict__ matrix,   // [sketch_dim, head_dim/8]
    int sketch_dim,
    int head_dim,
    uint64_t seed)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= sketch_dim) return;

    int bytes_per_row = head_dim / 8;
    uint8_t* row_ptr = matrix + row * bytes_per_row;

    // Initialize Philox PRNG with unique sequence per row
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(row), 0, &state);

    // Generate 8 random signs per byte using Rademacher distribution
    for (int b = 0; b < bytes_per_row; b++) {
        uint8_t packed = 0;
        // Generate 4 random values at a time (Philox generates float4)
        float4 r1 = curand_uniform4(&state);
        float4 r2 = curand_uniform4(&state);

        // Pack 8 signs: bit = 1 if uniform > 0.5 (i.e., +1), else 0 (i.e., -1)
        packed |= (r1.x > 0.5f) ? (1u << 0) : 0;
        packed |= (r1.y > 0.5f) ? (1u << 1) : 0;
        packed |= (r1.z > 0.5f) ? (1u << 2) : 0;
        packed |= (r1.w > 0.5f) ? (1u << 3) : 0;
        packed |= (r2.x > 0.5f) ? (1u << 4) : 0;
        packed |= (r2.y > 0.5f) ? (1u << 5) : 0;
        packed |= (r2.z > 0.5f) ? (1u << 6) : 0;
        packed |= (r2.w > 0.5f) ? (1u << 7) : 0;

        row_ptr[b] = packed;
    }
}

bool qjl_init(QJLProjection& proj, int head_dim, int sketch_dim, uint64_t seed,
              cudaStream_t stream) {
    if (head_dim <= 0 || sketch_dim <= 0) {
        IMP_LOG_ERROR("QJL: invalid dimensions head_dim=%d sketch_dim=%d", head_dim, sketch_dim);
        return false;
    }
    if (head_dim % 8 != 0) {
        IMP_LOG_ERROR("QJL: head_dim=%d must be divisible by 8", head_dim);
        return false;
    }

    int bytes_per_row = head_dim / 8;
    size_t total_bytes = static_cast<size_t>(sketch_dim) * bytes_per_row;

    cudaError_t err = cudaMalloc(&proj.matrix, total_bytes);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("QJL: cudaMalloc failed for %.2f KiB projection matrix",
                      static_cast<double>(total_bytes) / 1024.0);
        return false;
    }

    proj.sketch_dim = sketch_dim;
    proj.head_dim = head_dim;
    proj.seed = seed;

    // Launch generation: one thread per row
    int threads = 256;
    int blocks = (sketch_dim + threads - 1) / threads;
    qjl_generate_matrix_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<uint8_t*>(proj.matrix), sketch_dim, head_dim, seed);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("QJL: matrix generation kernel failed: %s", cudaGetErrorString(err));
        cudaFree(proj.matrix);
        proj.matrix = nullptr;
        return false;
    }

    IMP_LOG_INFO("QJL: initialized %dx%d Rademacher projection matrix (%.1f KiB, seed=%llu)",
                 sketch_dim, head_dim, static_cast<double>(total_bytes) / 1024.0,
                 static_cast<unsigned long long>(seed));
    return true;
}

void qjl_destroy(QJLProjection& proj) {
    if (proj.matrix) {
        cudaFree(proj.matrix);
        proj.matrix = nullptr;
    }
    proj.sketch_dim = 0;
    proj.head_dim = 0;
}

} // namespace imp
