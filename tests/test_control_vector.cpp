// Control-vector GGUF parsing (roadmap row 114): llama.cpp layout, direction.N F32 [d_model].

#include "lora/control_vector.h"
#include "model/gguf_loader.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

using namespace imp;

namespace {

struct Tensor1d {
    std::string name;
    std::vector<float> data;
    uint32_t type = 0;  // GGUF F32
};

class GgufBuilder {
public:
    std::vector<uint8_t> build(const std::vector<Tensor1d>& ts) {
        buf_.clear();
        u32(GGUF_MAGIC);
        u32(3);
        u64(ts.size());
        u64(1);
        str("general.architecture");
        u32(8);  // STRING
        str("controlvector");
        uint64_t off = 0;
        for (const auto& t : ts) {
            str(t.name);
            u32(1);
            u64(t.data.size());
            u32(t.type);
            u64(off);
            off += (t.data.size() * sizeof(float) + 31) / 32 * 32;
        }
        while (buf_.size() % 32)
            buf_.push_back(0);
        for (const auto& t : ts) {
            const auto* p = reinterpret_cast<const uint8_t*>(t.data.data());
            buf_.insert(buf_.end(), p, p + t.data.size() * sizeof(float));
            while (buf_.size() % 32)
                buf_.push_back(0);
        }
        return buf_;
    }

private:
    void raw(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        buf_.insert(buf_.end(), b, b + n);
    }
    void u32(uint32_t v) { raw(&v, sizeof(v)); }
    void u64(uint64_t v) { raw(&v, sizeof(v)); }
    void str(const std::string& s) {
        u64(s.size());
        raw(s.data(), s.size());
    }
    std::vector<uint8_t> buf_;
};

std::vector<float> ramp(int n, float base) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = base + 0.5f * i;
    return v;
}

constexpr int kD = 8;
constexpr int kLayers = 4;

}  // namespace

TEST(ControlVector, DirectionNLandsOnLayerN) {
    GgufBuilder b;
    const auto bytes = b.build({{"direction.1", ramp(kD, 1.0f)}, {"direction.3", ramp(kD, -2.0f)}});
    ControlVectorHost cv;
    std::string err;
    ASSERT_TRUE(parse_control_vector_gguf(bytes, kD, kLayers, cv, &err)) << err;
    ASSERT_EQ(cv.dirs.size(), static_cast<size_t>(kLayers * kD));
    EXPECT_EQ(cv.layer_mask, (std::vector<uint8_t>{0, 1, 0, 1}));
    EXPECT_FLOAT_EQ(cv.dirs[1 * kD + 0], 1.0f);
    EXPECT_FLOAT_EQ(cv.dirs[1 * kD + 7], 4.5f);
    EXPECT_FLOAT_EQ(cv.dirs[3 * kD + 2], -1.0f);
    for (int c = 0; c < kD; ++c) {
        EXPECT_EQ(cv.dirs[0 * kD + c], 0.0f);
        EXPECT_EQ(cv.dirs[2 * kD + c], 0.0f);
    }
}

TEST(ControlVector, RefusesLayerZeroAndPastLastLayer) {
    GgufBuilder b;
    ControlVectorHost cv;
    std::string err;
    EXPECT_FALSE(parse_control_vector_gguf(b.build({{"direction.0", ramp(kD, 0)}}), kD, kLayers, cv, &err));
    EXPECT_NE(err.find("direction.0"), std::string::npos) << err;
    EXPECT_FALSE(parse_control_vector_gguf(b.build({{"direction.4", ramp(kD, 0)}}), kD, kLayers, cv, &err));
    EXPECT_FALSE(parse_control_vector_gguf(b.build({{"direction.x", ramp(kD, 0)}}), kD, kLayers, cv, &err));
}

TEST(ControlVector, RefusesWrongWidthAndType) {
    GgufBuilder b;
    ControlVectorHost cv;
    std::string err;
    EXPECT_FALSE(
        parse_control_vector_gguf(b.build({{"direction.1", ramp(kD + 1, 0)}}), kD, kLayers, cv, &err));
    EXPECT_NE(err.find("d_model"), std::string::npos) << err;
    Tensor1d f16{"direction.1", ramp(kD / 2, 0), 1};  // 1 = F16: bytes of kD halves
    EXPECT_FALSE(parse_control_vector_gguf(b.build({f16}), kD, kLayers, cv, &err));
}

TEST(ControlVector, RefusesEmptyTruncatedAndForeignFiles) {
    GgufBuilder b;
    ControlVectorHost cv;
    std::string err;
    EXPECT_FALSE(
        parse_control_vector_gguf(b.build({{"blk.0.attn_q.weight", ramp(kD, 0)}}), kD, kLayers, cv, &err));
    EXPECT_NE(err.find("no direction"), std::string::npos) << err;
    auto bytes = b.build({{"direction.1", ramp(kD, 0)}});
    bytes.resize(bytes.size() - 8);  // last 2 floats gone
    EXPECT_FALSE(parse_control_vector_gguf(bytes, kD, kLayers, cv, &err));
    const std::vector<uint8_t> junk = {'N', 'O', 'P', 'E', 0, 0, 0, 0};
    EXPECT_FALSE(parse_control_vector_gguf(junk, kD, kLayers, cv, &err));
}
