#include "vision/vision_model.h"
#include <cuda_runtime.h>

namespace imp {

// The tower's blocks are T2 arena slices since F-12; the arena releases them
// wholesale on close, so there is nothing to hand back per tensor. Keeping the
// old cudaFree loop would now free arena memory out from under the arena.
VisionModel::~VisionModel() = default;

}  // namespace imp
