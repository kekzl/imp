// Log-once latches re-arm on every model load (AUDIT_ARCH_2026_11_24 C-12, D-10). A process
// static bool kept model B silent about a fallback model A had tripped. CPU only: the C-API
// case refuses at the model check, after the re-arm.
#include "core/cuda_static_reset.h"
#include "core/log_once.h"
#include "imp/imp.h"
#include "api/imp_internal.h"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

namespace {

imp::LogOnce s_file_scope_latch;  // registered during static init

}  // namespace

TEST(LogOnce, FiresOnceThenAgainAfterRearmAll) {
    static imp::LogOnce latch;
    EXPECT_TRUE(latch.first());
    EXPECT_FALSE(latch.first());
    EXPECT_FALSE(latch.first());
    imp::log_once_rearm_all();
    EXPECT_TRUE(latch.first());
    EXPECT_FALSE(latch.first());
}

TEST(LogOnce, RearmAllReachesFileScopeLatchesAndEveryArrayElement) {
    static imp::LogOnce per_key[3];
    const int before = imp::log_once_count();
    static imp::LogOnce later[4];
    EXPECT_EQ(imp::log_once_count(), before + 4);
    for (auto& l : per_key)
        EXPECT_TRUE(l.first());
    (void)s_file_scope_latch.first();
    EXPECT_FALSE(s_file_scope_latch.first());
    imp::log_once_rearm_all();
    for (auto& l : per_key)
        EXPECT_TRUE(l.first());
    EXPECT_TRUE(s_file_scope_latch.first());
    EXPECT_TRUE(later[3].first());
}

TEST(LogOnce, ExactlyOneThreadWinsARace) {
    static imp::LogOnce latch;
    std::atomic<int> winners{0};
    std::vector<std::thread> ts;
    for (int i = 0; i < 8; ++i)
        ts.emplace_back([&] {
            for (int k = 0; k < 1000; ++k)
                winners += latch.first() ? 1 : 0;
        });
    for (auto& t : ts)
        t.join();
    EXPECT_EQ(winners.load(), 1);
}

// imp_context_create re-arms before it looks at the model: the empty handle returns
// INVALID_MODEL with the latch armed again. A refused call before that point does not.
TEST(LogOnce, ContextCreateRearmsBeforeTheModelCheck) {
    static imp::LogOnce latch;
    static ImpModel_T dummy;
    ASSERT_TRUE(latch.first());
    ImpConfig cfg = imp_config_default();
    ImpContext ctx = nullptr;
    cfg.device_id = 1;
    EXPECT_EQ(imp_context_create(&dummy, &cfg, &ctx), IMP_ERROR_INVALID_ARG);
    EXPECT_FALSE(latch.first()) << "a refused config must not re-arm";
    cfg.device_id = 0;
    EXPECT_EQ(imp_context_create(&dummy, &cfg, &ctx), IMP_ERROR_INVALID_MODEL);
    EXPECT_TRUE(latch.first()) << "imp_context_create did not re-arm the log-once latches";
}

// ~Engine and imp_gpu_release() run the pre-cudaDeviceReset registry; the re-arm is one hook.
TEST(LogOnce, RearmIsARegisteredStaticResetHook) {
    EXPECT_TRUE(imp::cuda_static_reset_has_hook(&imp::log_once_rearm_all));
}
