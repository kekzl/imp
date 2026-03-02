#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cuda_fp16.h>

namespace imp {

struct ImageData {
    std::vector<half> pixels;  // [3, H, W] CHW layout, normalized FP16
    int width = 0;
    int height = 0;
};

// Load image from file, resize to target_size x target_size, normalize, convert to FP16 CHW.
bool load_and_preprocess_image(const std::string& path, int target_size,
                                const float mean[3], const float std[3],
                                ImageData& out);

// Load image from memory buffer, resize + normalize + FP16 CHW.
bool load_and_preprocess_image_from_memory(const uint8_t* data, size_t len,
                                            int target_size,
                                            const float mean[3], const float std[3],
                                            ImageData& out);

} // namespace imp
