// The "not one expert tensor was recognised" guard in WeightMap::apply_weights.
//
// It exists to stop a MoE checkpoint whose expert layout imp cannot read from
// loading and generating garbage through null experts (the #925 class). It had
// no test, and it was wrong: it asked only about `expert_w_gate`, so it fired on
// every up/down-only MoE — the entire Nemotron-H family, which has no gate
// projection. That went unnoticed because the caller discards the bool, so
// Nemotron-3-Nano logged the error and then generated correctly.
//
// These tests pin both directions: the guard must stay silent for a 2-projection
// MoE, and must still fire when the experts really are absent.

#include "model/model.h"
#include "model/weight_map.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

// A tensor is "present" iff data != nullptr — that is exactly what the guard
// asks, so the storage behind it never gets read.
Tensor fake_weight(void* backing, int64_t n, int64_t k) {
    Tensor t;
    t.data = backing;
    t.qtype = QType::F16;
    t.ndim = 2;
    t.shape[0] = n;
    t.shape[1] = k;
    return t;
}

struct Fixture {
    std::vector<uint16_t> backing{1024, 0};
    Model model;
    std::unordered_map<std::string, Tensor> tensors;

    Fixture(int n_experts, int n_layers = 1) {
        model.config_.arch = ModelArch::NEMOTRON_H_MOE;
        model.config_.n_experts = n_experts;
        model.config_.n_layers = n_layers;
        model.config_.d_model = 8;
        model.layers_.resize(static_cast<size_t>(n_layers));
    }

    void* mem() { return backing.data(); }

    // Nemotron-H spelling: the loader translates backbone.layers.N.mixer.* into
    // model.layers.N.mlp.* before matching, so use the checkpoint's own names.
    void add_expert(int layer, int expert, const char* proj) {
        const std::string n = "backbone.layers." + std::to_string(layer) + ".mixer.experts." +
                              std::to_string(expert) + "." + proj + ".weight";
        tensors[n] = fake_weight(mem(), 8, 8);
    }
};

// The regression: up_proj + down_proj and no gate_proj is a complete MoE for
// this family. Before the fix this returned false and logged "not one expert
// tensor was recognised".
TEST(WeightMapExpertGuard, AcceptsUpDownOnlyMoE) {
    Fixture f(/*n_experts=*/2);
    for (int e = 0; e < 2; e++) {
        f.add_expert(0, e, "up_proj");
        f.add_expert(0, e, "down_proj");
    }
    WeightMap wm(ModelArch::NEMOTRON_H_MOE);
    EXPECT_TRUE(wm.apply_weights(f.model, f.tensors));
}

// The guard's actual job: a config that declares experts, and not one expert
// weight in the checkpoint. Loading this would route through null experts.
TEST(WeightMapExpertGuard, StillRefusesWhenNoExpertArrives) {
    Fixture f(/*n_experts=*/2);
    // A non-expert tensor so the map is not simply empty.
    f.tensors["backbone.layers.0.norm.weight"] = fake_weight(f.mem(), 8, 1);
    WeightMap wm(ModelArch::NEMOTRON_H_MOE);
    EXPECT_FALSE(wm.apply_weights(f.model, f.tensors));
}

// A gate+up+down MoE must keep working — the fix widened the check, it did not
// move it.
TEST(WeightMapExpertGuard, AcceptsGateUpDownMoE) {
    Fixture f(/*n_experts=*/2);
    for (int e = 0; e < 2; e++) {
        f.add_expert(0, e, "gate_proj");
        f.add_expert(0, e, "up_proj");
        f.add_expert(0, e, "down_proj");
    }
    WeightMap wm(ModelArch::NEMOTRON_H_MOE);
    EXPECT_TRUE(wm.apply_weights(f.model, f.tensors));
}

// down_proj alone is still a recognised expert layout — the guard asks "did any
// expert weight arrive", not "is the expert complete". Pinned so a later
// tightening is a deliberate decision rather than a silent one.
TEST(WeightMapExpertGuard, DownProjAloneCounts) {
    Fixture f(/*n_experts=*/2);
    f.add_expert(0, 0, "down_proj");
    WeightMap wm(ModelArch::NEMOTRON_H_MOE);
    EXPECT_TRUE(wm.apply_weights(f.model, f.tensors));
}

// A dense model (no experts declared) must never reach the guard.
TEST(WeightMapExpertGuard, DenseModelUnaffected) {
    Fixture f(/*n_experts=*/0);
    f.tensors["backbone.layers.0.norm.weight"] = fake_weight(f.mem(), 8, 1);
    WeightMap wm(ModelArch::NEMOTRON_H_MOE);
    EXPECT_TRUE(wm.apply_weights(f.model, f.tensors));
}

}  // namespace
}  // namespace imp
