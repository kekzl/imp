// Host-placed token embedding (#2484): same FP16 bits as the VRAM upload's staging.
#include "model/host_embedding.h"
#include "model/weight_upload_traits.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <vector>

using imp::QType;

namespace {

std::vector<uint32_t> probe_floats() {
    std::vector<uint32_t> v = {0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u, 0x33000000u,  // 2^-25 tie
                               0x33800000u, 0x387FC000u, 0x477FE000u, 0x47800000u, 0x7F800000u, 0x3DCCCCCDu};
    for (uint32_t i = 0; i < 4096; ++i)
        v.push_back(0x30000000u + i * 0x00031337u);
    return v;
}

}  // namespace

TEST(HostEmbedding, Bf16MatchesUploadStagingBits) {
    std::vector<uint16_t> src;
    for (uint32_t f : probe_floats())
        src.push_back(static_cast<uint16_t>(f >> 16));
    std::vector<uint16_t> got(src.size());
    imp::embedding_to_fp16(src.data(), QType::BF16, src.size(), got.data(), 7);
    for (size_t i = 0; i < src.size(); ++i) {
        const float f = std::bit_cast<float>(static_cast<uint32_t>(src[i]) << 16);
        ASSERT_EQ(got[i], imp::wupload::float_to_fp16(f)) << "element " << i;
    }
}

TEST(HostEmbedding, F32AndF16MatchUploadStagingBits) {
    std::vector<float> f32;
    for (uint32_t f : probe_floats())
        f32.push_back(std::bit_cast<float>(f));
    std::vector<uint16_t> got(f32.size());
    imp::embedding_to_fp16(f32.data(), QType::F32, f32.size(), got.data(), 3);
    for (size_t i = 0; i < f32.size(); ++i)
        ASSERT_EQ(got[i], imp::wupload::float_to_fp16(f32[i])) << "element " << i;
    std::vector<uint16_t> copy(got.size());
    imp::embedding_to_fp16(got.data(), QType::F16, got.size(), copy.data(), 5);
    EXPECT_EQ(copy, got);
}

TEST(HostEmbedding, SupportedTypesAreWhatEmbeddingLookupReads) {
    for (QType q : {QType::F32, QType::F16, QType::BF16, QType::Q8_0, QType::Q6_K})
        EXPECT_TRUE(imp::host_embedding_supported(q)) << imp::qtype_name(q);
    for (QType q : {QType::Q4_K, QType::NVFP4, QType::INT8})
        EXPECT_FALSE(imp::host_embedding_supported(q)) << imp::qtype_name(q);
}
