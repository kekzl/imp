#pragma once

// AWQ-class calibration is always a paired elementwise rewrite: an input-channel scale s is
// legal only because the producer divides that channel by s first, so consumer columns are
// multiplied and producer outputs divided, keeping (x/s)(W diag(s))^T == x W^T unchanged
// before quantization. Both helpers live here rather than in imp-quantize so the invariance
// can be tested without a GPU or a checkpoint.

#include "quant/awq_norm_fold.h"

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// In-place on a row-major [N,K] FP16 matrix: W[i][j] = W[i][j]/row_div[i]*col_scale[j].
// Either vector may be empty (skipped); a wrong-length vector is ignored, so a
// partially-populated plan is safe on every tensor. Row division applies first, matching
// how the plan builder searches column scales on top of already-divided rows.
void awq_apply_matrix(std::vector<uint16_t>& fp16, int64_t N, int64_t K, const std::vector<float>& row_div,
                      const std::vector<float>& col_scale);

// In-place on n_elems values of `dtype`, dividing element j by div[j]. Used for 1-D
// producers (RMSNorm weights, biases of a scaled linear's output channels). Returns false
// (bytes untouched) on an unsupported dtype or length mismatch - a silently skipped fold
// would leave a subtly wrong checkpoint. `offset`: on a unit-offset RMSNorm the kernel
// applies (1+g), so dividing g divides the wrong thing; the offset arm stores (1+g)/s-1
// instead. Biases and plain norms take the default.
bool awq_apply_vector_div(unsigned char* bytes, size_t n_elems, const std::string& dtype,
                          const std::vector<float>& div, NormOffset offset = NormOffset::Plain);

}  // namespace imp
