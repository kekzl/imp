// The 5 Inkling kernel shape guards throw (B-108): before, each logged and returned, the output
// buffer kept stale values and a captured step simply had no node. CPU lane: every case leaves
// before the first launch.

#include <gtest/gtest.h>

#include "compute/inkling_ops.h"

#include <functional>
#include <stdexcept>
#include <string>

namespace {

namespace ik = imp::inkling;

std::string thrown(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

bool has(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

}  // namespace

TEST(InklingGuardsFailLoud, AttentionHeadDimPastTheKernel) {
    ik::AttnArgs a;
    a.hd = 256;  // kAttnMaxHd 128
    EXPECT_TRUE(has(thrown([&] { ik::attention(a, nullptr); }), "inkling attention"));
    a.hd = 128;
    a.blocks = reinterpret_cast<const int*>(&a);  // a block table beside a ring
    a.ring = 64;
    EXPECT_TRUE(has(thrown([&] { ik::attention(a, nullptr); }), "inkling attention"));
}

TEST(InklingGuardsFailLoud, MixWithoutStates) {
    ik::MixArgs a;  // st.n 0
    EXPECT_TRUE(has(thrown([&] { ik::mix(a, 1, 64, nullptr); }), "inkling mix"));
}

TEST(InklingGuardsFailLoud, GateNormWithoutGates) {
    ik::GateNormArgs a;
    a.n = 0;
    EXPECT_TRUE(
        has(thrown([&] { ik::gate_norm(a, ik::GateOut{}, nullptr, 1, 64, nullptr); }), "inkling gate_norm"));
}

TEST(InklingGuardsFailLoud, KvSiteHeadDimOffTheWarp) {
    ik::KvSiteArgs a;
    a.hd = 48;  // % 32 != 0
    EXPECT_TRUE(has(thrown([&] { ik::kv_site(a, nullptr); }), "inkling kv_site"));
}

TEST(InklingGuardsFailLoud, RouterPastItsExperts) {
    EXPECT_TRUE(has(thrown([] {
                        ik::route_topk(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, 64, 129, 8,
                                       1.0f, nullptr);
                    }),
                    "inkling router"));
}
