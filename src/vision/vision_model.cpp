#include "vision/vision_model.h"
#include "memory/vram_allocator.h"
#include <cuda_runtime.h>

namespace imp {

VisionModel::~VisionModel() { free_gpu(); }

void VisionModel::free_gpu() {
    for (void* ptr : gpu_allocs) {
        if (!ptr)
            continue;
        if (allocator)
            allocator->free(ptr);
        else
            cudaFree(ptr);
    }
    gpu_allocs.clear();
    allocator = nullptr;
}

}  // namespace imp
