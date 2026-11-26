// cuBLASLt launcher argument structs (compute/gemm_lt_ex.h, audit D-11, B-133): every field is
// named at the call site. The structured bindings pin the field count (an added or dropped field
// fails to compile); positional init in the old parameter order pins the declaration order.

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

int g_slot[6] = {};
int g_stream_tag = 0;
auto* const kStream = reinterpret_cast<cudaStream_t>(&g_stream_tag);

// 22 fields, the old lt_gemm_strided_batched parameter list in order (loc stays a parameter).
TEST(LtGemmArgs, CarriesEveryFieldInDeclarationOrder) {
    const LtGemmArgs a{CUBLAS_OP_T,        CUBLAS_OP_N, 1,          2, 3,
                       &g_slot[0],         &g_slot[1],  CUDA_R_32F, 4, 5,
                       &g_slot[2],         CUDA_R_8I,   6,          7, &g_slot[3],
                       &g_slot[4],         CUDA_R_16BF, 8,          9, 10,
                       CUBLAS_COMPUTE_32I, kStream};
    const auto& [ta, tb, m, n, k, alpha, A, at, lda, sa, B, bt, ldb, sb, beta, C, ct, ldc, sc, batch, comp,
                 st] = a;
    EXPECT_EQ((std::array{a.m, a.n, a.k, a.lda, a.ldb, a.ldc, a.batch}), (std::array{1, 2, 3, 4, 6, 8, 10}));
    EXPECT_EQ((std::array{a.stride_a, a.stride_b, a.stride_c}), (std::array{5LL, 7LL, 9LL}));
    EXPECT_EQ((std::array{a.transa, a.transb}), (std::array{CUBLAS_OP_T, CUBLAS_OP_N}));
    EXPECT_EQ((std::array{a.a_type, a.b_type, a.c_type}), (std::array{CUDA_R_32F, CUDA_R_8I, CUDA_R_16BF}));
    EXPECT_EQ((std::array<const void*, 5>{a.alpha, a.A, a.B, a.beta, a.C}),
              (std::array<const void*, 5>{&g_slot[0], &g_slot[1], &g_slot[2], &g_slot[3], &g_slot[4]}));
    EXPECT_EQ(a.compute, CUBLAS_COMPUTE_32I);
    EXPECT_EQ(a.stream, kStream);
    EXPECT_EQ(&batch, &a.batch);
    EXPECT_EQ(&st, &a.stream);
}

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

// 19 fields, the old lt_gemm_batched parameter list in order.
TEST(LtGemmBatchedArgs, CarriesEveryFieldInDeclarationOrder) {
    static const void* const arr_a[1] = {};
    static const void* const arr_b[1] = {};
    static void* const arr_c[1] = {};
    const LtGemmBatchedArgs a{CUBLAS_OP_T, CUBLAS_OP_N, 1,           2,     3,         &g_slot[0],
                              arr_a,       CUDA_R_32F,  4,           arr_b, CUDA_R_8I, 5,
                              &g_slot[1],  arr_c,       CUDA_R_16BF, 6,     7,         CUBLAS_COMPUTE_32I,
                              kStream};
    const auto& [ta, tb, m, n, k, alpha, pa, at, lda, pb, bt, ldb, beta, pc, ct, ldc, batch, comp, st] = a;
    EXPECT_EQ((std::array{a.m, a.n, a.k, a.lda, a.ldb, a.ldc, a.batch}), (std::array{1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ((std::array{a.a_type, a.b_type, a.c_type}), (std::array{CUDA_R_32F, CUDA_R_8I, CUDA_R_16BF}));
    EXPECT_EQ(a.a, arr_a);
    EXPECT_EQ(a.b, arr_b);
    EXPECT_EQ(a.c, arr_c);
    EXPECT_EQ(a.alpha, &g_slot[0]);
    EXPECT_EQ(a.beta, &g_slot[1]);
    EXPECT_EQ(a.compute, CUBLAS_COMPUTE_32I);
    EXPECT_EQ(&st, &a.stream);
}

// The pointer-array form had no defaults; omitted batch = 1 like LtGemmArgs, the legacy stream.
TEST(LtGemmBatchedArgs, OmittedFieldsTakeTheOldDefaults) {
    const LtGemmBatchedArgs a{.m = 1};
    EXPECT_EQ(a.batch, 1);
    EXPECT_EQ(a.stream, nullptr);
    EXPECT_EQ(a.a, nullptr);
}

}  // namespace
