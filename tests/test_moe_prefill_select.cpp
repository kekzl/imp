#include "compute/gemm.h"
#include "core/qtype.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace imp {
namespace {

// #2459: a qtype without a prefill arm is refused, never routed to a default kernel.
TEST(MoePrefillSelect, FusedDp4aRefusesUnsupportedQType) {
    for (QType qt : {QType::Q8_0, QType::Q5_1, QType::Q4_0, QType::NVFP4, QType::F16}) {
        EXPECT_FALSE(moe_fused_dp4a_prefill_supported(qt)) << qtype_name(qt);
        EXPECT_THROW(moe_fused_dp4a_prefill_kernel(qt), std::invalid_argument) << qtype_name(qt);
    }
    for (QType qt : {QType::Q4_K, QType::Q5_K, QType::Q6_K}) {
        EXPECT_TRUE(moe_fused_dp4a_prefill_supported(qt)) << qtype_name(qt);
        EXPECT_NE(moe_fused_dp4a_prefill_kernel(qt), nullptr) << qtype_name(qt);
    }
}

// qkind must match mmq_imma_moe_gemm's table (mmq_q8_imma.h).
TEST(MoePrefillSelect, ImmaQkindRefusesUnsupportedQType) {
    for (QType qt : {QType::Q4_0, QType::Q2_K, QType::Q3_K, QType::NVFP4, QType::F16}) {
        EXPECT_FALSE(moe_imma_prefill_supported(qt)) << qtype_name(qt);
        EXPECT_THROW((void)moe_imma_prefill_qkind(qt), std::invalid_argument) << qtype_name(qt);
    }
    EXPECT_EQ(moe_imma_prefill_qkind(QType::Q8_0), 0);
    EXPECT_EQ(moe_imma_prefill_qkind(QType::Q4_K), 1);
    EXPECT_EQ(moe_imma_prefill_qkind(QType::Q6_K), 2);
    EXPECT_EQ(moe_imma_prefill_qkind(QType::Q5_1), 3);
    EXPECT_EQ(moe_imma_prefill_qkind(QType::Q5_K), 4);
}

}  // namespace
}  // namespace imp
