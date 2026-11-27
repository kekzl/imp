// cuBLASLt launcher argument structs (compute/gemm_lt_ex.h, audit D-11, B-133): every field is
// named at the call site, so a declaration-order swap fails the build there. Pinned here: the
// omitted-field defaults and lt_gemm_ex's one-GEMM rewrite.

#include <gtest/gtest.h>

#include "compute/gemm_lt_ex.h"

#include <array>
#include <type_traits>

namespace {

using imp::lt_gemm_ex_args;
using imp::LtGemmArgs;
using imp::LtGemmBatchedArgs;

static_assert(std::is_aggregate_v<LtGemmArgs>);
static_assert(std::is_aggregate_v<LtGemmBatchedArgs>);

int g_stream_tag = 0;
auto* const kStream = reinterpret_cast<cudaStream_t>(&g_stream_tag);

// Defaults = lt_gemm_ex's old implicit values: strides 0, batch 1 (one GEMM), the legacy stream.
TEST(LtGemmArgs, OmittedFieldsTakeTheOldDefaults) {
    const LtGemmArgs a{.m = 1};
    EXPECT_EQ((std::array{a.stride_a, a.stride_b, a.stride_c}), (std::array{0LL, 0LL, 0LL}));
    EXPECT_EQ(a.batch, 1);
    EXPECT_EQ(a.stream, nullptr);
}

// lt_gemm_ex is one GEMM whatever the caller set: strides 0, batch 1, every other field kept.
TEST(LtGemmArgs, GemmExForcesOneUnstridedGemm) {
    const LtGemmArgs in{.transa = CUBLAS_OP_T,
                        .m = 1,
                        .n = 2,
                        .k = 3,
                        .lda = 4,
                        .stride_a = 5,
                        .ldb = 6,
                        .stride_b = 7,
                        .ldc = 8,
                        .stride_c = 9,
                        .batch = 10,
                        .compute = CUBLAS_COMPUTE_32I,
                        .stream = kStream};
    const LtGemmArgs a = lt_gemm_ex_args(in);
    EXPECT_EQ((std::array{a.stride_a, a.stride_b, a.stride_c}), (std::array{0LL, 0LL, 0LL}));
    EXPECT_EQ(a.batch, 1);
    EXPECT_EQ((std::array{a.m, a.n, a.k, a.lda, a.ldb, a.ldc}), (std::array{1, 2, 3, 4, 6, 8}));
    EXPECT_EQ(a.transa, CUBLAS_OP_T);
    EXPECT_EQ(a.compute, CUBLAS_COMPUTE_32I);
    EXPECT_EQ(a.stream, kStream);
}

// The pointer-array form had no defaults; omitted batch = 1 like LtGemmArgs, the legacy stream.
TEST(LtGemmBatchedArgs, OmittedFieldsTakeTheOldDefaults) {
    const LtGemmBatchedArgs a{.m = 1};
    EXPECT_EQ(a.batch, 1);
    EXPECT_EQ(a.stream, nullptr);
    EXPECT_EQ(a.a, nullptr);
}

}  // namespace
