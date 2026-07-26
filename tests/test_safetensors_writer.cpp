// =============================================================================
// SafeTensors writer tests.
//
// The writer is what lets imp produce checkpoints instead of only consuming
// them (docs/roadmap.md gap 1: the NVFP4 path otherwise depends on a third
// party publishing an export). A checkpoint that writes "successfully" but
// round-trips wrong is a silent corruption of model weights, so these tests
// pin the wire layout byte for byte, and pin that a REJECTED write leaves no
// file behind at all.
// =============================================================================

#include "model/safetensors_writer.h"
#include "model/safetensors_raw.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

using namespace imp;

namespace {

std::string temp_path(const char* stem) {
    return (std::filesystem::temp_directory_path() /
            (std::string("imp_stw_") + stem + "_" + std::to_string(::getpid()) + ".safetensors"))
        .string();
}

struct Blob {
    std::vector<unsigned char> bytes;
    uint64_t header_len = 0;
    std::string header;
    const unsigned char* data() const { return bytes.data() + 8 + header_len; }
    size_t data_size() const { return bytes.size() - 8 - header_len; }
};

// Read a written file back the way the format specifies.
Blob read_back(const std::string& path) {
    Blob b;
    FILE* f = fopen(path.c_str(), "rb");
    EXPECT_NE(f, nullptr) << "cannot reopen " << path;
    if (!f)
        return b;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    b.bytes.resize(static_cast<size_t>(sz));
    size_t got = fread(b.bytes.data(), 1, b.bytes.size(), f);
    fclose(f);
    EXPECT_EQ(got, b.bytes.size());
    if (b.bytes.size() >= 8) {
        for (int i = 7; i >= 0; i--)
            b.header_len = (b.header_len << 8) | b.bytes[static_cast<size_t>(i)];
        if (8 + b.header_len <= b.bytes.size())
            b.header.assign(reinterpret_cast<const char*>(b.bytes.data() + 8),
                            static_cast<size_t>(b.header_len));
    }
    return b;
}

TEST(SafeTensorsWriter, RoundTripsTensorBytesAndHeader) {
    // The exact trio a Modelopt NVFP4 export ships per quantized matrix, plus
    // an unquantized F16 tensor.
    const std::vector<uint16_t> w = {0x3C00, 0x4000, 0xBC00, 0x0000};  // F16 1, 2, -1, 0
    const std::vector<unsigned char> packed = {0x12, 0x34, 0x56};      // NVFP4 nibbles
    const std::vector<unsigned char> micro = {0x38, 0x3C};             // F8_E4M3 micro-scales
    const float scale = 0.125f;

    std::vector<SafeTensorsOut> ts = {
        {"model.layers.0.mlp.up_proj.weight", "U8", {3, 1}, packed.data(), packed.size()},
        {"model.layers.0.mlp.up_proj.weight_scale", "F8_E4M3", {2}, micro.data(), micro.size()},
        {"model.layers.0.mlp.up_proj.weight_scale_2", "F32", {1}, &scale, sizeof(float)},
        {"model.embed_tokens.weight", "F16", {2, 2}, w.data(), w.size() * 2},
    };

    const std::string path = temp_path("roundtrip");
    ASSERT_EQ(write_safetensors(path, ts, {{"format", "pt"}, {"producer", "imp"}}), "");

    Blob b = read_back(path);
    ASSERT_FALSE(b.header.empty());

    // Data block must start 8-byte aligned (mmap-friendly; readers assume it).
    EXPECT_EQ((8 + b.header_len) % 8, 0u);

    // Header describes every tensor, with the packed shape for the U8 weight.
    EXPECT_NE(b.header.find("\"model.layers.0.mlp.up_proj.weight\""), std::string::npos);
    EXPECT_NE(b.header.find("\"dtype\":\"U8\""), std::string::npos);
    EXPECT_NE(b.header.find("\"shape\":[3,1]"), std::string::npos);
    EXPECT_NE(b.header.find("\"dtype\":\"F8_E4M3\""), std::string::npos);

    // Metadata is reserved-key, string->string.
    EXPECT_NE(b.header.find("\"__metadata__\""), std::string::npos);
    EXPECT_NE(b.header.find("\"producer\":\"imp\""), std::string::npos);

    // Tensor payloads follow header order, contiguous, byte-identical.
    ASSERT_EQ(b.data_size(), packed.size() + micro.size() + sizeof(float) + w.size() * 2);
    size_t at = 0;
    EXPECT_EQ(memcmp(b.data() + at, packed.data(), packed.size()), 0);
    at += packed.size();
    EXPECT_EQ(memcmp(b.data() + at, micro.data(), micro.size()), 0);
    at += micro.size();
    EXPECT_EQ(memcmp(b.data() + at, &scale, sizeof(float)), 0);
    at += sizeof(float);
    EXPECT_EQ(memcmp(b.data() + at, w.data(), w.size() * 2), 0);

    std::filesystem::remove(path);
}

TEST(SafeTensorsWriter, OffsetsAreContiguousAndDataRelative) {
    const std::vector<unsigned char> a(5, 0xAA), c(7, 0xCC);
    std::vector<SafeTensorsOut> ts = {{"a", "U8", {5}, a.data(), a.size()},
                                      {"c", "U8", {7}, c.data(), c.size()}};
    const std::string path = temp_path("offsets");
    ASSERT_EQ(write_safetensors(path, ts), "");

    Blob b = read_back(path);
    // First tensor starts at 0 (offsets are relative to the data block, not the
    // file) and the second picks up exactly where the first ended.
    EXPECT_NE(b.header.find("\"data_offsets\":[0,5]"), std::string::npos) << b.header;
    EXPECT_NE(b.header.find("\"data_offsets\":[5,12]"), std::string::npos) << b.header;
    EXPECT_EQ(b.data_size(), 12u);
    std::filesystem::remove(path);
}

TEST(SafeTensorsWriter, EscapesNamesThatWouldBreakTheHeaderJson) {
    const std::vector<unsigned char> d(2, 1);
    std::vector<SafeTensorsOut> ts = {{"we\"ird\\name", "U8", {2}, d.data(), d.size()}};
    const std::string path = temp_path("escape");
    ASSERT_EQ(write_safetensors(path, ts), "");
    Blob b = read_back(path);
    EXPECT_NE(b.header.find("\"we\\\"ird\\\\name\""), std::string::npos) << b.header;
    std::filesystem::remove(path);
}

// A rejected write must not leave a file — a half-written checkpoint that still
// parses is worse than no checkpoint.
TEST(SafeTensorsWriter, RejectsBadInputAndLeavesNoFile) {
    const std::vector<unsigned char> d(4, 7);
    const std::string path = temp_path("reject");

    struct Case {
        const char* what;
        std::vector<SafeTensorsOut> ts;
    };
    const std::vector<Case> cases = {
        {"empty tensor list", {}},
        {"unknown dtype", {{"t", "F4_MADE_UP", {4}, d.data(), d.size()}}},
        {"shape/nbytes mismatch", {{"t", "U8", {99}, d.data(), d.size()}}},
        {"F16 shape implies 8 bytes, got 4", {{"t", "F16", {4}, d.data(), d.size()}}},
        {"empty name", {{"", "U8", {4}, d.data(), d.size()}}},
        {"reserved name", {{"__metadata__", "U8", {4}, d.data(), d.size()}}},
        {"null data", {{"t", "U8", {4}, nullptr, 4}}},
    };

    for (const auto& c : cases) {
        std::filesystem::remove(path);
        const std::string err = write_safetensors(path, c.ts);
        EXPECT_FALSE(err.empty()) << "should have been rejected: " << c.what;
        EXPECT_FALSE(std::filesystem::exists(path)) << "left a file behind after: " << c.what;
        EXPECT_FALSE(std::filesystem::exists(path + ".partial"))
            << "left a .partial behind after: " << c.what;
    }
}

// Writer and raw reader validated against each other: what imp writes, imp must
// read back identically. This is the contract the quantizer depends on — a
// mismatch here means produced checkpoints hold different weights than intended.
TEST(SafeTensorsRaw, ReadsBackEverythingTheWriterWrote) {
    const std::vector<uint16_t> emb = {0x3C00, 0x4000, 0xBC00, 0x0000, 0x3800, 0x3C00};
    const std::vector<unsigned char> packed = {0x12, 0x34, 0x56, 0x78};
    const std::vector<unsigned char> micro = {0x38, 0x3C};
    const float ts = 0.0625f;

    std::vector<SafeTensorsOut> out = {
        {"model.layers.0.self_attn.q_proj.weight", "U8", {2, 2}, packed.data(), packed.size()},
        {"model.layers.0.self_attn.q_proj.weight_scale", "F8_E4M3", {2, 1}, micro.data(), micro.size()},
        {"model.layers.0.self_attn.q_proj.weight_scale_2", "F32", {1}, &ts, sizeof(float)},
        {"model.embed_tokens.weight", "BF16", {3, 2}, emb.data(), emb.size() * 2},
    };
    const std::string path = temp_path("rawread");
    ASSERT_EQ(write_safetensors(path, out, {{"producer", "imp-quantize"}}), "");

    RawSafeTensors in;
    ASSERT_EQ(in.open(path), "");
    ASSERT_EQ(in.tensors().size(), out.size());

    for (size_t i = 0; i < out.size(); i++) {
        const auto& a = out[i];
        const auto& b = in.tensors()[i];
        EXPECT_EQ(b.name, a.name);
        EXPECT_EQ(b.dtype, a.dtype) << a.name;  // dtype survives verbatim, incl. BF16
        EXPECT_EQ(b.shape, a.shape) << a.name;
        ASSERT_EQ(b.nbytes, a.nbytes) << a.name;
        EXPECT_EQ(memcmp(b.data, a.data, a.nbytes), 0) << "payload differs for " << a.name;
    }
    ASSERT_EQ(in.metadata().size(), 1u);
    EXPECT_EQ(in.metadata()[0].first, "producer");
    EXPECT_EQ(in.metadata()[0].second, "imp-quantize");

    std::filesystem::remove(path);
}

TEST(SafeTensorsRaw, RejectsTruncatedAndBogusFiles) {
    const std::string path = temp_path("bogus");
    RawSafeTensors r;

    EXPECT_FALSE(r.open(path + ".does.not.exist").empty());

    // Too small to hold even the length prefix.
    {
        FILE* f = fopen(path.c_str(), "wb");
        fwrite("abc", 1, 3, f);
        fclose(f);
    }
    EXPECT_FALSE(r.open(path).empty());

    // Header length points past the end of the file.
    {
        FILE* f = fopen(path.c_str(), "wb");
        unsigned char len[8] = {0xFF, 0xFF, 0, 0, 0, 0, 0, 0};
        fwrite(len, 1, 8, f);
        fwrite("{}", 1, 2, f);
        fclose(f);
    }
    EXPECT_FALSE(r.open(path).empty());

    // Well-formed length, but the header is not JSON.
    {
        FILE* f = fopen(path.c_str(), "wb");
        unsigned char len[8] = {4, 0, 0, 0, 0, 0, 0, 0};
        fwrite(len, 1, 8, f);
        fwrite("not!", 1, 4, f);
        fclose(f);
    }
    EXPECT_FALSE(r.open(path).empty());

    std::filesystem::remove(path);
}

TEST(SafeTensorsWriter, DtypeSizesMatchTheLoadersTable) {
    EXPECT_EQ(safetensors_dtype_size("F32"), 4u);
    EXPECT_EQ(safetensors_dtype_size("F16"), 2u);
    EXPECT_EQ(safetensors_dtype_size("BF16"), 2u);
    EXPECT_EQ(safetensors_dtype_size("F8_E4M3"), 1u);  // NVFP4 micro-scales
    EXPECT_EQ(safetensors_dtype_size("U8"), 1u);       // NVFP4 packed nibbles
    EXPECT_EQ(safetensors_dtype_size("nonsense"), 0u);
}

}  // namespace
