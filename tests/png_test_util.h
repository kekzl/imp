#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace imp {
namespace test {

// 8-bit RGB PNG, stored (uncompressed) deflate, filter 0. rows[y][x] = {r,g,b}, row 0 = top.
// stb decodes PNG only (#2401), so in-memory test pictures are PNG.
inline std::vector<uint8_t> make_png_rgb(const std::vector<std::vector<std::array<uint8_t, 3>>>& rows) {
    const uint32_t h = static_cast<uint32_t>(rows.size());
    const uint32_t w = static_cast<uint32_t>(rows[0].size());

    std::vector<uint8_t> raw;
    for (const auto& row : rows) {
        raw.push_back(0);  // filter: none
        for (const auto& p : row)
            raw.insert(raw.end(), p.begin(), p.end());
    }

    auto be32 = [](std::vector<uint8_t>& v, uint32_t x) {
        for (int s = 24; s >= 0; s -= 8)
            v.push_back(static_cast<uint8_t>(x >> s));
    };
    auto crc32 = [](const uint8_t* p, size_t n) {
        uint32_t c = 0xFFFFFFFFu;
        for (size_t i = 0; i < n; ++i) {
            c ^= p[i];
            for (int k = 0; k < 8; ++k)
                c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
        }
        return c ^ 0xFFFFFFFFu;
    };

    std::vector<uint8_t> z = {0x78, 0x01};  // zlib header, deflate 32K window
    for (size_t off = 0;;) {
        const size_t len = std::min<size_t>(raw.size() - off, 65535);
        const bool last = off + len == raw.size();
        z.push_back(last ? 1 : 0);  // BFINAL, BTYPE=00 (stored)
        z.push_back(static_cast<uint8_t>(len));
        z.push_back(static_cast<uint8_t>(len >> 8));
        z.push_back(static_cast<uint8_t>(~len));
        z.push_back(static_cast<uint8_t>(~len >> 8));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(off),
                 raw.begin() + static_cast<std::ptrdiff_t>(off + len));
        off += len;
        if (last)
            break;
    }
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) {
        a = (a + c) % 65521;
        b = (b + a) % 65521;
    }
    be32(z, (b << 16) | a);

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    auto chunk = [&](const char* type, const std::vector<uint8_t>& data) {
        be32(png, static_cast<uint32_t>(data.size()));
        const size_t start = png.size();
        png.insert(png.end(), type, type + 4);
        png.insert(png.end(), data.begin(), data.end());
        be32(png, crc32(png.data() + start, png.size() - start));
    };
    std::vector<uint8_t> ihdr;
    be32(ihdr, w);
    be32(ihdr, h);
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // depth 8, RGB, deflate, filter 0, no interlace
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    return png;
}

}  // namespace test
}  // namespace imp
