// #2317: GemmKernelRegistry must hold the 13 produced keys in every binary linking libimp.a.
// CPU only: contains() never invokes a handler.

#include "exec/gemm_kernel_registry.h"

#include <gtest/gtest.h>

#include <iterator>

using namespace imp;

TEST(GemmKernelRegistryLink, HoldsExactlyTheProducedKeys) {
    const auto& reg = GemmKernelRegistry::instance();
    const GemmStrategy produced[] = {
        {StorageTier::FP16, QType::NONE, false},  // generic dequant catch-all
        {StorageTier::FP16, QType::Q4_K, true},   // GGUF small-M x11
        {StorageTier::FP16, QType::Q5_K, true},         {StorageTier::FP16, QType::Q5_1, true},
        {StorageTier::FP16, QType::Q8_0, true},         {StorageTier::FP16, QType::Q6_K, true},
        {StorageTier::FP16, QType::Q4_0, true},         {StorageTier::FP16, QType::Q2_K, true},
        {StorageTier::FP16, QType::Q3_K, true},         {StorageTier::FP16, QType::IQ4_NL, true},
        {StorageTier::FP16, QType::IQ4_XS, true},       {StorageTier::FP16, QType::IQ3_S, true},
        {StorageTier::NVFP4_SFATOM, QType::F16, false},  // NVFP4 prefill (own tile)
    };
    EXPECT_EQ(reg.size(), std::size(produced))
        << "1 generic dequant + 11 GGUF small-M qtypes + 1 NVFP4_SFATOM";
    for (const auto& s : produced)
        EXPECT_TRUE(reg.contains(s)) << "tier=" << static_cast<int>(s.tier)
                                     << " qtype=" << static_cast<int>(s.weight_qtype)
                                     << " m_is_one=" << s.m_is_one;
}

// Pin the GemmStrategy operator== contract — used by the linear-scan
// lookup in dispatch().
TEST(GemmStrategy, EqualityIsByValue) {
    GemmStrategy a{StorageTier::FP16, QType::F16, /*m_is_one=*/true};
    GemmStrategy b{StorageTier::FP16, QType::F16, /*m_is_one=*/true};
    GemmStrategy c{StorageTier::FP16, QType::F16, /*m_is_one=*/false};
    GemmStrategy d{StorageTier::FP8, QType::F16, /*m_is_one=*/true};
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(a == d);
}
