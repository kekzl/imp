// nvfp4_dequant_ws_take (audit D-9): gemm_nvfp4's fallback workspace is a per-call argument owned
// by the caller, not a process global. A deferred take runs once per workspace; two workspaces
// share nothing. CPU lane: the take is host-only.

#include <gtest/gtest.h>

#include <cstddef>

#include "quant/nvfp4_dequant_ws.h"

namespace {

using imp::nvfp4_dequant_ws_take;
using imp::Nvfp4DequantWorkspace;

struct Provider {
    void* buf = nullptr;
    int calls = 0;
    static void* provide(void* self) {
        auto* p = static_cast<Provider*>(self);
        ++p->calls;
        return p->buf;
    }
};

TEST(Nvfp4DequantWs, NoWorkspaceOrTooSmallGivesNone) {
    EXPECT_EQ(nvfp4_dequant_ws_take(nullptr, 16), nullptr);
    alignas(16) char storage[64];
    Nvfp4DequantWorkspace ws{.buf = storage, .size = sizeof(storage)};
    EXPECT_EQ(nvfp4_dequant_ws_take(&ws, sizeof(storage) + 1), nullptr);
    EXPECT_EQ(nvfp4_dequant_ws_take(&ws, sizeof(storage)), storage);
}

TEST(Nvfp4DequantWs, DeferredTakeRunsOncePerWorkspace) {
    alignas(16) char storage[64];
    Provider p{.buf = storage};
    Nvfp4DequantWorkspace ws{.size = sizeof(storage), .provide = &Provider::provide, .ctx = &p};
    EXPECT_EQ(nvfp4_dequant_ws_take(&ws, sizeof(storage) * 2), nullptr);
    EXPECT_EQ(p.calls, 0) << "a take the workspace cannot cover must not run the provider";
    EXPECT_EQ(nvfp4_dequant_ws_take(&ws, 32), storage);
    EXPECT_EQ(nvfp4_dequant_ws_take(&ws, 64), storage);
    EXPECT_EQ(p.calls, 1);
}

TEST(Nvfp4DequantWs, EmptyProviderLeavesNoWorkspaceAndIsNotAskedAgain) {
    Provider p;
    Nvfp4DequantWorkspace ws{.size = 64, .provide = &Provider::provide, .ctx = &p};
    EXPECT_EQ(nvfp4_dequant_ws_take(&ws, 32), nullptr);
    EXPECT_EQ(nvfp4_dequant_ws_take(&ws, 32), nullptr);
    EXPECT_EQ(p.calls, 1);
    EXPECT_EQ(ws.size, 0u);
}

TEST(Nvfp4DequantWs, TwoWorkspacesShareNothing) {
    alignas(16) char a_buf[64], b_buf[64];
    Provider pa{.buf = a_buf}, pb{.buf = b_buf};
    Nvfp4DequantWorkspace a{.size = 64, .provide = &Provider::provide, .ctx = &pa};
    Nvfp4DequantWorkspace b{.size = 64, .provide = &Provider::provide, .ctx = &pb};
    EXPECT_EQ(nvfp4_dequant_ws_take(&a, 32), a_buf);
    EXPECT_EQ(pb.calls, 0);
    EXPECT_EQ(nvfp4_dequant_ws_take(&b, 32), b_buf);
    EXPECT_EQ(nvfp4_dequant_ws_take(&a, 32), a_buf);
    EXPECT_EQ(pa.calls, 1);
    EXPECT_EQ(pb.calls, 1);
}

}  // namespace
