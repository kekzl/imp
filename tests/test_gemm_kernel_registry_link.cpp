// #2317: GemmKernelRegistry must hold the 10 produced keys in every binary linking libimp.a.
// CPU only: contains() never invokes a handler.

#include "exec/gemm_kernel_registry.h"

#include <gtest/gtest.h>

#include <iterator>

using namespace imp;

TEST(GemmKernelRegistryLink, HoldsExactlyTheProducedKeys) {
    const auto& reg = GemmKernelRegistry::instance();
    const GemmStrategy produced[] = {
        {StorageTier::FP16, QType::NONE, false},           // generic dequant catch-all
        {StorageTier::FP16, QType::Q4_K, true},            // GGUF small-M x8
        {StorageTier::FP16, QType::Q5_K, true},
        {StorageTier::FP16, QType::Q5_1, true},
        {StorageTier::FP16, QType::Q8_0, true},
        {StorageTier::FP16, QType::Q6_K, true},
        {StorageTier::FP16, QType::Q4_0, true},
        {StorageTier::FP16, QType::Q2_K, true},
        {StorageTier::FP16, QType::Q3_K, true},
        {StorageTier::CUTLASS_NVFP4, QType::F16, false},   // CUTLASS NVFP4 prefill
    };
    EXPECT_EQ(reg.size(), std::size(produced))
        << "1 generic dequant + 8 GGUF small-M qtypes + 1 CUTLASS_NVFP4";
    for (const auto& s : produced)
        EXPECT_TRUE(reg.contains(s)) << "tier=" << static_cast<int>(s.tier)
                                     << " qtype=" << static_cast<int>(s.weight_qtype)
                                     << " m_is_one=" << s.m_is_one;
}
