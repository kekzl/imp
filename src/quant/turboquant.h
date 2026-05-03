#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

namespace imp {

// QJL (Quantized Johnson-Lindenstrauss) random projection matrix.
// Used by TurboQuant for 1-bit error-correction sketches of KV cache keys.
//
// The matrix stores Rademacher random signs (±1) as packed bits:
//   matrix[i * (head_dim/8) + j/8] bit (j%8) = sign of R[i][j]
//   1 = +1, 0 = -1
//
// Shape: [sketch_dim, head_dim], packed as [sketch_dim, head_dim/8] bytes.
// sketch_dim = head_dim (paper default) for quality-neutral 4-bit quantization.
struct QJLProjection {
    void* matrix = nullptr;  // [sketch_dim, head_dim/8] packed bits on GPU
    int sketch_dim = 0;
    int head_dim = 0;
    uint64_t seed = 0;
};

// Initialize QJL projection matrix with seeded Philox PRNG.
// sketch_dim should equal head_dim for best accuracy.
// The matrix is deterministic for a given seed, ensuring reproducibility.
bool qjl_init(QJLProjection& proj, int head_dim, int sketch_dim, uint64_t seed,
              cudaStream_t stream = nullptr);

// Free the GPU memory for the projection matrix.
void qjl_destroy(QJLProjection& proj);

}  // namespace imp
