#pragma once

// The two elementwise rewrites an AWQ-class calibration plan consists of.
//
// A scale s on an input channel is only legal because something else divides
// that channel by s first — the producer. So the transform always comes in
// pairs: the consumer's columns get multiplied, the producer's outputs get
// divided, and the product is unchanged before quantization:
//
//     (x / s) (W diag(s))^T  ==  x W^T
//
// Both helpers live here rather than in imp-quantize so the invariance above
// can be tested directly, without a GPU or a checkpoint.

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// In-place on a row-major [N, K] matrix of FP16 bits:
//     W[i][j] = W[i][j] / row_div[i] * col_scale[j]
// Either vector may be empty (skipped); a wrong-length vector is ignored, which
// is what makes a partially-populated plan safe to apply to every tensor.
// Row division is applied first — the plan builder searches column scales on
// top of already-divided rows, so the writer must compose them the same way.
void awq_apply_matrix(std::vector<uint16_t>& fp16, int64_t N, int64_t K, const std::vector<float>& row_div,
                      const std::vector<float>& col_scale);

// In-place on `n_elems` values of `dtype` ("F32" / "F16" / "BF16"), dividing
// element j by div[j]. Used for the 1-D producers: RMSNorm weights and the
// biases of a linear whose output channels were scaled. Returns false (and
// leaves the bytes untouched) for an unsupported dtype or a length mismatch —
// a silently skipped fold would produce a checkpoint that is subtly wrong.
bool awq_apply_vector_div(unsigned char* bytes, size_t n_elems, const std::string& dtype,
                          const std::vector<float>& div);

}  // namespace imp
