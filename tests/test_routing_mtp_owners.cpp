// I5 last hand-written release sites: the allocating MoE routing result and the MTP workspace hold
// their buffers in DeviceOwned owners. CPU lane: the release hook replaces the driver, the buffers
// are host addresses that are never dereferenced.

#include <gtest/gtest.h>

#include "compute/moe_routing.h"
#include "compute/mtp_forward.h"
#include "memory/device_owned.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

using namespace imp;

namespace {

std::vector<void*>& log() {
    static std::vector<void*> l;
    return l;
}

void record(void* ptr, DeviceFamily /*family*/, VRAMAllocator* /*alloc*/, cudaStream_t /*stream*/) {
    log().push_back(ptr);
}

int released(const void* p) { return static_cast<int>(std::count(log().begin(), log().end(), p)); }

static_assert(!std::is_copy_constructible_v<MoeRoutingOwned>, "a copied owner would free twice");
static_assert(std::is_copy_constructible_v<MoeRoutingResult>, "the routing view stays a copyable view");

class RoutingMtpOwnersTest : public ::testing::Test {
protected:
    void SetUp() override {
        log().clear();
        prev_ = set_device_owned_release_hook(&record);
    }
    void TearDown() override { set_device_owned_release_hook(prev_); }

    char slots_[8] = {};
    void* at(int i) { return &slots_[i]; }

private:
    DeviceOwnedReleaseHook prev_ = nullptr;
};

TEST_F(RoutingMtpOwnersTest, MoeRoutingOwnedReleasesItsFourBuffersOnceAndAMoveLeavesNoView) {
    MoeRoutingOwned a;
    a.expert_indices.data = at(0);
    a.expert_weights.data = at(1);
    a.sorted_token_ids.data = at(2);
    a.expert_offsets.data = at(3);
    a.indices_owned.adopt(at(0), 8, nullptr);
    a.weights_owned.adopt(at(1), 8, nullptr);
    a.sorted_owned.adopt(at(2), 8, nullptr);
    a.offsets_owned.adopt(at(3), 8, nullptr);

    MoeRoutingOwned b(std::move(a));
    EXPECT_EQ(a.expert_indices.data, nullptr) << "a moved-from owner must not keep a view";
    EXPECT_EQ(a.expert_offsets.data, nullptr);
    EXPECT_FALSE(a.indices_owned);
    const MoeRoutingResult view = b;  // NOLINT(cppcoreguidelines-slicing): a sliced copy is a view
    EXPECT_EQ(view.sorted_token_ids.data, at(2));
    EXPECT_TRUE(log().empty());

    b = MoeRoutingOwned{};
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(released(at(i)), 1) << "buffer " << i;
    EXPECT_EQ(log().size(), 4u);
}

TEST_F(RoutingMtpOwnersTest, MtpWorkspaceFreeReleasesEveryOwnedBufferOnceAndClearsTheViews) {
    MtpDraftWorkspace ws;
    for (int i = 0; i < 3; ++i) {
        DeviceOwned<void> o;
        o.adopt(at(i), 8, nullptr);
        ws.owned.push_back(std::move(o));
    }
    ws.d_emb_norm = at(0);
    ws.d_chain_tokens = static_cast<int32_t*>(at(1));
    ws.d_feed_tokens = static_cast<int32_t*>(at(2));
    ws.d_row_slots = ws.d_feed_tokens;  // an alias, never released on its own

    mtp_workspace_free(ws);
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(released(at(i)), 1) << "buffer " << i;
    EXPECT_EQ(log().size(), 3u);
    EXPECT_TRUE(ws.owned.empty());
    EXPECT_EQ(ws.d_emb_norm, nullptr);
    EXPECT_EQ(ws.d_chain_tokens, nullptr);
    EXPECT_EQ(ws.d_feed_tokens, nullptr);
    EXPECT_EQ(ws.d_row_slots, nullptr);

    mtp_workspace_free(ws);  // a second free releases nothing
    EXPECT_EQ(log().size(), 3u);
}

}  // namespace
