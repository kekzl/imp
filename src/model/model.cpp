#include "model/model.h"
#include <cuda_runtime.h>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace imp {

Model::~Model() {
    // Free all GPU-side weight buffers.
    for (void* ptr : gpu_allocations_) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    gpu_allocations_.clear();
    gpu_weights_ready_ = false;

#ifdef __linux__
    if (mmap_base_ && mmap_size_ > 0) {
        munmap(mmap_base_, mmap_size_);
        mmap_base_ = nullptr;
        mmap_size_ = 0;
    }
#endif
}

const char* model_arch_name(ModelArch arch) {
    switch (arch) {
        case ModelArch::LLAMA:    return "llama";
        case ModelArch::MISTRAL:  return "mistral";
        case ModelArch::MIXTRAL:  return "mixtral";
        case ModelArch::DEEPSEEK: return "deepseek";
        case ModelArch::GENERIC:  return "generic";
    }
    return "unknown";
}

} // namespace imp
