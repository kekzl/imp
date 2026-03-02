#include "vision/vision_model.h"
#include <cuda_runtime.h>

namespace imp {

VisionModel::~VisionModel() {
    free_gpu();
}

void VisionModel::free_gpu() {
    for (void* ptr : gpu_allocs) {
        if (ptr) cudaFree(ptr);
    }
    gpu_allocs.clear();
}

} // namespace imp
