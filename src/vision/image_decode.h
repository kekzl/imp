#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace imp {

// Decoded picture: [height, width, 3] u8 RGB, row-major.
struct DecodedImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
};

// The one image decoder of every vision family. JPEG: libjpeg-turbo with Pillow's settings
// (ISLOW IDCT, fancy upsampling, Pillow's CMYK->RGB), bit-identical to Image.convert("RGB").
// PNG: stb_image. Every other format is refused (#2401). Either side <= 16384 px; false on any
// failure.
[[nodiscard]] bool decode_image(std::span<const uint8_t> data, DecodedImage& out);
[[nodiscard]] bool decode_image_file(const std::string& path, DecodedImage& out);

}  // namespace imp
