// EngineCapabilities (audit C-6): a request feature hooked into the generic layer loop is a silent
// no-op on a forward that replaces the loop (Inkling-2-Mini). The table is a pure function of the
// load's facts; the server refuses what it says no to (tests/test_capability_refusal.cpp).

#include <gtest/gtest.h>

#include "runtime/engine_capabilities.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>

namespace {

using imp::derive_capabilities;
using imp::EngineCapabilities;
using imp::ForwardFacts;

constexpr ForwardFacts kDense{.model_loaded = true};
// Engine::init on an Inkling2-Loop checkpoint: the executor holds sequence slots, no ssm_state_.
constexpr ForwardFacts kInkling{.model_loaded = true, .layer_loop_replaced = true};

}  // namespace

TEST(EngineCapabilities, GenericDenseForwardServesEveryFeature) {
    const EngineCapabilities c = derive_capabilities(kDense);
    EXPECT_TRUE(c.logit_lens);
    EXPECT_TRUE(c.sae);
    EXPECT_TRUE(c.lora);
    EXPECT_TRUE(c.control_vector);
    EXPECT_TRUE(c.phrase_ban);
    EXPECT_TRUE(c.cfg);
}

TEST(EngineCapabilities, InklingForwardRefusesWhatHooksIntoTheLayerLoop) {
    const EngineCapabilities c = derive_capabilities(kInkling);
    EXPECT_FALSE(c.logit_lens);
    EXPECT_FALSE(c.sae);
    EXPECT_FALSE(c.lora);
    EXPECT_FALSE(c.control_vector);
    EXPECT_FALSE(c.phrase_ban);
    // Each guidance row decodes on its own sequence slot: the pair works.
    EXPECT_TRUE(c.cfg);
}

TEST(EngineCapabilities, GatedResidualRefusesResidualReaders) {
    ForwardFacts f = kDense;
    f.gated_residual = true;
    const EngineCapabilities c = derive_capabilities(f);
    EXPECT_FALSE(c.logit_lens);
    EXPECT_FALSE(c.sae);
    EXPECT_FALSE(c.control_vector);
    EXPECT_TRUE(c.lora);
    EXPECT_TRUE(c.phrase_ban);
}

TEST(EngineCapabilities, HybridGuidanceNeedsBatchedRecurrentDecode) {
    ForwardFacts f = kDense;
    f.recurrent_state = true;
    EXPECT_FALSE(derive_capabilities(f).cfg);
    f.batched_recurrent_decode = true;
    EXPECT_TRUE(derive_capabilities(f).cfg);
}

TEST(EngineCapabilities, NoModelServesNothing) {
    EXPECT_EQ(derive_capabilities({}), EngineCapabilities{});
    ForwardFacts f = kInkling;
    f.model_loaded = false;
    EXPECT_EQ(derive_capabilities(f), EngineCapabilities{});
}

TEST(EngineCapabilities, BitsRoundTripAndEveryFieldHasANameRow) {
    // One bit and one /v1/models name per bool field: a field without a row is never published.
    static_assert(sizeof(EngineCapabilities) == std::size(imp::kCapabilityNames));
    std::set<std::string> names;
    std::set<uint32_t> bits;
    for (const auto& [name, flag] : imp::kCapabilityNames) {
        EngineCapabilities only;
        only.*flag = true;
        EXPECT_EQ(EngineCapabilities::from_bits(only.bits()), only) << name;
        names.insert(name);
        bits.insert(only.bits());
    }
    EXPECT_EQ(names.size(), std::size(imp::kCapabilityNames));
    EXPECT_EQ(bits.size(), std::size(imp::kCapabilityNames));
    EXPECT_FALSE(bits.contains(0u));
    for (const ForwardFacts& f : {kDense, kInkling}) {
        const EngineCapabilities c = derive_capabilities(f);
        EXPECT_EQ(EngineCapabilities::from_bits(c.bits()), c);
    }
}
