// write_npy_fp32: the .npy the diagnostic dumps are read back through.
//
// Untested until now, and everything downstream of it is a measurement: the
// reference-parity harness compares imp's logits against HF by loading these
// files in numpy. A header this writer got subtly wrong (a shape written
// column-major, a header length not padded to the 64-byte boundary) would not
// crash — numpy would either refuse the file or, worse, read it transposed, and
// the resulting "parity failure" would be in the writer, not in the engine.
//
// So this asserts the format against the NPY v1.0 spec by hand: magic, version,
// the little-endian header length, the padding rule, the dict fields, and the
// payload laid out row-major.

#include "exec/executor_debug.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace imp {
namespace {

std::string temp_path(const char* stem) {
    auto p = std::filesystem::temp_directory_path() /
             (std::string(stem) + std::to_string(::getpid()) + ".npy");
    return p.string();
}

std::vector<char> read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Npy {
    std::string header;
    const float* data = nullptr;
    size_t n_floats = 0;
    std::vector<char> raw;
};

// Parse just enough of NPY v1.0 to assert on it.
Npy parse(const std::string& path) {
    Npy out;
    out.raw = read_all(path);
    EXPECT_GE(out.raw.size(), 10u);
    EXPECT_EQ(std::string(out.raw.data(), 6), std::string("\x93NUMPY", 6));
    EXPECT_EQ(static_cast<unsigned char>(out.raw[6]), 1u) << "major version";
    EXPECT_EQ(static_cast<unsigned char>(out.raw[7]), 0u) << "minor version";
    uint16_t hlen = 0;
    std::memcpy(&hlen, out.raw.data() + 8, 2);
    out.header.assign(out.raw.data() + 10, hlen);
    const size_t off = 10u + hlen;
    out.data = reinterpret_cast<const float*>(out.raw.data() + off);
    out.n_floats = (out.raw.size() - off) / sizeof(float);
    return out;
}

TEST(NpyWriterTest, WritesParseableHeaderAndRowMajorPayload) {
    const std::string path = temp_path("imp_npy_basic_");
    // 2 rows x 3 cols, values chosen so a transpose would be visible.
    const std::vector<float> v = {1.f, 2.f, 3.f, 10.f, 20.f, 30.f};
    write_npy_fp32(path, v.data(), 2, 3);

    Npy n = parse(path);
    EXPECT_NE(n.header.find("'descr': '<f4'"), std::string::npos) << n.header;
    EXPECT_NE(n.header.find("'fortran_order': False"), std::string::npos) << n.header;
    EXPECT_NE(n.header.find("'shape': (2, 3)"), std::string::npos) << n.header;
    ASSERT_EQ(n.n_floats, v.size());
    for (size_t i = 0; i < v.size(); i++)
        EXPECT_FLOAT_EQ(n.data[i], v[i]) << "element " << i << " — a transpose shows up here";
    std::filesystem::remove(path);
}

TEST(NpyWriterTest, HeaderIsPaddedToTheSpecBoundary) {
    // numpy requires len(magic) + 2 + 2 + len(header) to be a multiple of 64,
    // and the header to end with '\n'. Getting this wrong is the failure that
    // makes np.load refuse the file outright.
    for (int cols : {1, 3, 250, 248320}) {
        const std::string path = temp_path("imp_npy_pad_");
        std::vector<float> v(static_cast<size_t>(cols), 0.5f);
        write_npy_fp32(path, v.data(), 1, cols);
        auto raw = read_all(path);
        uint16_t hlen = 0;
        std::memcpy(&hlen, raw.data() + 8, 2);
        EXPECT_EQ((10u + hlen) % 64u, 0u) << "cols=" << cols << " total header not 64-aligned";
        EXPECT_EQ(raw[10 + hlen - 1], '\n') << "cols=" << cols << " header must end with newline";
        std::filesystem::remove(path);
    }
}

TEST(NpyWriterTest, ShapeMatchesTheRequestedRowsAndCols) {
    // The logit dump writes [rows, vocab] and the GDN dump writes
    // [layers, per_layer]; a swapped pair would silently reinterpret both.
    const std::string path = temp_path("imp_npy_shape_");
    std::vector<float> v(4 * 7, 1.0f);
    write_npy_fp32(path, v.data(), 4, 7);
    Npy n = parse(path);
    EXPECT_NE(n.header.find("'shape': (4, 7)"), std::string::npos) << n.header;
    EXPECT_EQ(n.n_floats, 28u);
    std::filesystem::remove(path);
}

TEST(NpyWriterTest, SingleRowKeepsTwoDimensionalShape) {
    // The prefill dump is one row. numpy must still see (1, N), not (N,):
    // the comparison script reshapes, but a rank change would break any
    // consumer that indexes [0].
    const std::string path = temp_path("imp_npy_single_");
    std::vector<float> v(5, 2.0f);
    write_npy_fp32(path, v.data(), 1, 5);
    Npy n = parse(path);
    EXPECT_NE(n.header.find("'shape': (1, 5)"), std::string::npos) << n.header;
    std::filesystem::remove(path);
}

}  // namespace
}  // namespace imp
