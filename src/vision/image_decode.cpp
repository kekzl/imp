// stb's default (1<<24/side, bounded only by w*h*3<=INT_MAX) lets a 700 KB PNG of a
// 26000x26000 flat image decode to ~2 GiB of host RGB before any resize (AUDIT_arch_2026
// F2-4). 16384 is far above every tower's useful input; both decoders refuse a larger picture.
#define STBI_MAX_DIMENSIONS 16384
// JPEG never reaches stb: its IDCT/upsampling differ from Pillow by up to 3/255 (#2381).
#define STBI_NO_JPEG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "vision/image_decode.h"
#include "core/logging.h"

// clang-format off
#include <cstdio>
#include <jpeglib.h>
#include <jerror.h>
// clang-format on

#include <climits>
#include <csetjmp>
#include <fstream>
#include <iterator>

namespace imp {

namespace {

constexpr int kMaxSide = STBI_MAX_DIMENSIONS;
// Progressive scan cap, TurboJPEG's TJPARAM_SCANLIMIT value: bounds decode time on hostile input.
constexpr int kMaxScans = 500;

struct JpegErr {
    jpeg_error_mgr mgr;
    jmp_buf jump;
    bool truncated;
    bool too_many_scans;
};

[[noreturn]] void jpeg_fail(j_common_ptr c) { std::longjmp(reinterpret_cast<JpegErr*>(c->err)->jump, 1); }

// Pillow raises on a truncated stream ("image file is truncated"); libjpeg only warns.
void jpeg_note(j_common_ptr c, int level) {
    if (level < 0 && c->err->msg_code == JWRN_JPEG_EOF)
        reinterpret_cast<JpegErr*>(c->err)->truncated = true;
}

void jpeg_quiet(j_common_ptr) {}

void jpeg_scan_limit(j_common_ptr c) {
    auto* d = reinterpret_cast<j_decompress_ptr>(c);
    if (d->progressive_mode && d->input_scan_number > kMaxScans) {
        reinterpret_cast<JpegErr*>(c->err)->too_many_scans = true;
        std::longjmp(reinterpret_cast<JpegErr*>(c->err)->jump, 1);
    }
}

// Pillow's "CMYK;I" unpack (Adobe inverted) then cmyk2rgb (Convert.c).
void cmyk_inverted_to_rgb(const uint8_t* in, uint8_t* out, int n) {
    for (int i = 0; i < n; ++i, in += 4, out += 3) {
        const int nk = in[3];
        for (int ch = 0; ch < 3; ++ch) {
            const int t = (255 - in[ch]) * nk + 128;
            const int v = nk - (((t >> 8) + t) >> 8);
            out[ch] = static_cast<uint8_t>(v < 0 ? 0 : v);
        }
    }
}

// No automatic objects with destructors live in this frame: longjmp skips none.
bool decode_jpeg(std::span<const uint8_t> data, DecodedImage& out, std::vector<uint8_t>& cmyk_row) {
    jpeg_decompress_struct c;
    JpegErr err{};
    jpeg_progress_mgr progress{};
    c.err = jpeg_std_error(&err.mgr);
    err.mgr.error_exit = jpeg_fail;
    err.mgr.emit_message = jpeg_note;
    err.mgr.output_message = jpeg_quiet;
    if (setjmp(err.jump)) {
        char msg[JMSG_LENGTH_MAX] = "too many progressive scans";
        if (!err.too_many_scans)
            err.mgr.format_message(reinterpret_cast<j_common_ptr>(&c), msg);
        IMP_LOG_ERROR("Vision: JPEG decode failed (%s)", msg);
        jpeg_destroy_decompress(&c);
        return false;
    }
    jpeg_create_decompress(&c);
    progress.progress_monitor = jpeg_scan_limit;
    c.progress = &progress;
    jpeg_mem_src(&c, data.data(), static_cast<unsigned long>(data.size()));
    jpeg_read_header(&c, TRUE);
    if (c.image_width > static_cast<JDIMENSION>(kMaxSide) ||
        c.image_height > static_cast<JDIMENSION>(kMaxSide)) {
        IMP_LOG_ERROR("Vision: JPEG %ux%u exceeds %d px per side", c.image_width, c.image_height, kMaxSide);
        jpeg_destroy_decompress(&c);
        return false;
    }
    const bool cmyk = c.jpeg_color_space == JCS_CMYK || c.jpeg_color_space == JCS_YCCK;
    c.out_color_space = cmyk ? JCS_CMYK : JCS_RGB;
    c.dct_method = JDCT_ISLOW;
    c.do_fancy_upsampling = TRUE;
    jpeg_start_decompress(&c);
    out.width = static_cast<int>(c.output_width);
    out.height = static_cast<int>(c.output_height);
    const size_t row = static_cast<size_t>(out.width) * 3;
    out.rgb.resize(row * static_cast<size_t>(out.height));
    if (cmyk)
        cmyk_row.resize(static_cast<size_t>(out.width) * 4);
    while (c.output_scanline < c.output_height) {
        uint8_t* dst = out.rgb.data() + static_cast<size_t>(c.output_scanline) * row;
        JSAMPROW r = cmyk ? cmyk_row.data() : dst;
        if (jpeg_read_scanlines(&c, &r, 1) != 1)
            break;
        if (cmyk)
            cmyk_inverted_to_rgb(cmyk_row.data(), dst, out.width);
    }
    if (!err.truncated && c.output_scanline == c.output_height)
        jpeg_finish_decompress(&c);
    const bool ok = !err.truncated && c.output_scanline == c.output_height;
    if (!ok)
        IMP_LOG_ERROR("Vision: JPEG data is truncated (%u of %u rows)", c.output_scanline, c.output_height);
    jpeg_destroy_decompress(&c);
    return ok;
}

bool is_jpeg(std::span<const uint8_t> d) {
    return d.size() >= 3 && d[0] == 0xFF && d[1] == 0xD8 && d[2] == 0xFF;
}

}  // namespace

bool decode_image(std::span<const uint8_t> data, DecodedImage& out) {
    out = DecodedImage{};
    if (is_jpeg(data)) {
        std::vector<uint8_t> cmyk_row;
        if (decode_jpeg(data, out, cmyk_row))
            return true;
        out = DecodedImage{};
        return false;
    }
    if (data.size() > static_cast<size_t>(INT_MAX)) {
        IMP_LOG_ERROR("Vision: image of %zu bytes is too large", data.size());
        return false;
    }
    int w = 0, h = 0, ch = 0;
    uint8_t* px = stbi_load_from_memory(data.data(), static_cast<int>(data.size()), &w, &h, &ch, 3);
    if (!px) {
        IMP_LOG_ERROR("Vision: image decode failed (%s)", stbi_failure_reason());
        return false;
    }
    out.width = w;
    out.height = h;
    out.rgb.assign(px, px + static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
    stbi_image_free(px);
    return true;
}

bool decode_image_file(const std::string& path, DecodedImage& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        out = DecodedImage{};
        IMP_LOG_ERROR("Vision: cannot open image file %s", path.c_str());
        return false;
    }
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    return decode_image(bytes, out);
}

}  // namespace imp
