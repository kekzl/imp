// C-API contract: hostile ImpConfig values through imp_context_create
// (AUDIT_arch_2026 G-3 / G-4). The public header is the ABI-stable boundary and
// neither shipping binary drives its config fields through it, so this file
// does. CPU-only by construction: every case is refused before a model or the
// GPU is looked at, so the model handle is an empty ImpModel_T and the process
// never touches CUDA.
#include "imp/imp.h"
#include "api/imp_internal.h"

#include <gtest/gtest.h>

namespace {

ImpModel empty_model() {
    static ImpModel_T dummy;
    return &dummy;
}

}  // namespace

TEST(CApiContract, NullArgumentsAreInvalidArg) {
    ImpConfig cfg = imp_config_default();
    ImpContext ctx = nullptr;
    EXPECT_EQ(imp_context_create(nullptr, &cfg, &ctx), IMP_ERROR_INVALID_ARG);
    EXPECT_EQ(imp_context_create(empty_model(), nullptr, &ctx), IMP_ERROR_INVALID_ARG);
    EXPECT_EQ(imp_context_create(empty_model(), &cfg, nullptr), IMP_ERROR_INVALID_ARG);
}

// Five named ImpDType values are not KV cache dtypes, and the enum has holes
// (9, 10) plus room above MXFP4_KV. All of them used to map to a QType the pool
// was sized for and the FP16 paged kernel then read.
TEST(CApiContract, KvCacheDtypeOutsideTheSixIsInvalidArgBeforeTheModelIsLookedAt) {
    const int hostile[] = {IMP_DTYPE_FP32, IMP_DTYPE_BF16, IMP_DTYPE_FP8_E5M2, IMP_DTYPE_INT32,
                           IMP_DTYPE_FP4_E2M1, 9, 10, 13, 15};
    for (int v : hostile) {
        ImpConfig cfg = imp_config_default();
        cfg.kv_cache_dtype = static_cast<ImpDType>(v);
        ImpContext ctx = nullptr;
        EXPECT_EQ(imp_context_create(empty_model(), &cfg, &ctx), IMP_ERROR_INVALID_ARG)
            << "kv_cache_dtype=" << v;
        EXPECT_EQ(ctx, nullptr) << "kv_cache_dtype=" << v;
    }
}

// The empty model handle is the next check in line, so INVALID_MODEL (not
// INVALID_ARG) is the proof that the dtype itself was accepted.
TEST(CApiContract, TheSixKvCacheDtypesPassValidationAndReachTheModelCheck) {
    const ImpDType valid[] = {IMP_DTYPE_FP16, IMP_DTYPE_FP8_E4M3, IMP_DTYPE_INT8,
                              IMP_DTYPE_INT4, IMP_DTYPE_NVFP4,    IMP_DTYPE_MXFP4_KV};
    for (ImpDType v : valid) {
        ImpConfig cfg = imp_config_default();
        cfg.kv_cache_dtype = v;
        ImpContext ctx = nullptr;
        EXPECT_EQ(imp_context_create(empty_model(), &cfg, &ctx), IMP_ERROR_INVALID_MODEL)
            << "kv_cache_dtype=" << static_cast<int>(v);
    }
    EXPECT_EQ(imp_config_default().kv_cache_dtype, IMP_DTYPE_FP16);
}
