#pragma once

#include <cuda_runtime.h>

namespace imp {

// Classifier-free guidance (row 117): cond[i] += scale * (cond[i] - uncond[i]) for i < n, in place.
void cfg_combine(float* cond, const float* uncond, float scale, int n, cudaStream_t stream);

}  // namespace imp
