#pragma once

// What a request may ask of one load (audit C-6): one field per feature whose code path differs by
// forward, derived once at the end of Engine::init; engine load/admit paths and the server 400 read it.
// Header-only, CUDA-free: tests/test_engine_capabilities.cpp pins the derivation on the CPU lane.

#include <cstdint>

namespace imp {

// The facts of one load the capabilities follow from.
struct ForwardFacts {
    bool model_loaded = false;
    // GraphExecutor::run_inkling_ ran every layer and returns 0: the generic layer loop runs no
    // iteration, so nothing hooked into it (taps, deltas, per-layer adds) runs.
    bool layer_loop_replaced = false;
    // That forward applies an adapter's q / k / v / o deltas itself (InklingForward::delta_, B-111 S5).
    bool own_forward_lora = false;
    bool gated_residual = false;            // ModelProfile::gated_residual: hc_count x d_model stream
    bool recurrent_state = false;           // engine-owned GDN / SSM state (Engine::ssm_state_)
    bool batched_recurrent_decode = false;  // per-row recurrent slots (Engine::d_ssm_seq_slots_)
};

struct EngineCapabilities {
    bool logit_lens = false;      // "logit_lens": post_layer_taps_ inside the layer loop
    bool sae = false;             // "sae_features", "sae_clamp": sae_apply_ inside the layer loop
    bool lora = false;            // "lora": lora_delta_ in the loop, InklingForward::delta_
    bool control_vector = false;  // "control_vector": control_vector_add_ inside the layer loop
    bool phrase_ban = false;      // "banned_strings", "banned_regex": rewind of paged KV + recurrent slabs
    bool cfg = false;             // "cfg_scale": the guidance pair decodes as two rows of one step

    friend bool operator==(const EngineCapabilities&, const EngineCapabilities&) = default;

    // Lock-free publication to the server (one atomic word per load).
    [[nodiscard]] constexpr uint32_t bits() const {
        return (logit_lens ? 1u : 0u) | (sae ? 2u : 0u) | (lora ? 4u : 0u) | (control_vector ? 8u : 0u) |
               (phrase_ban ? 16u : 0u) | (cfg ? 32u : 0u);
    }
    [[nodiscard]] static constexpr EngineCapabilities from_bits(uint32_t b) {
        return {(b & 1u) != 0, (b & 2u) != 0, (b & 4u) != 0, (b & 8u) != 0, (b & 16u) != 0, (b & 32u) != 0};
    }
};

// Every field by name: /v1/models lists these, a new field without a row fails a test.
struct CapabilityName {
    const char* name;
    bool EngineCapabilities::* flag;
};
inline constexpr CapabilityName kCapabilityNames[] = {
    {"logit_lens", &EngineCapabilities::logit_lens},
    {"sae", &EngineCapabilities::sae},
    {"lora", &EngineCapabilities::lora},
    {"control_vector", &EngineCapabilities::control_vector},
    {"phrase_ban", &EngineCapabilities::phrase_ban},
    {"cfg", &EngineCapabilities::cfg},
};

[[nodiscard]] constexpr EngineCapabilities derive_capabilities(const ForwardFacts& f) {
    const bool loop = f.model_loaded && !f.layer_loop_replaced;
    // The gated residual is not the d_model stream the LM head, an SAE or a vector was made for.
    const bool residual = loop && !f.gated_residual;
    return {
        .logit_lens = residual,
        .sae = residual,
        .lora = f.model_loaded && (!f.layer_loop_replaced || f.own_forward_lora),
        .control_vector = residual,
        // Rewind = KVCacheManager::rollback + ssm_state_ slabs; an own forward keeps its own sliding K/V and
        // conv state, which InklingForward::run refuses to step back into (pos0 != pos_).
        .phrase_ban = loop,
        // Two rows of one step: each row needs its own recurrent slot; an own forward binds one slot
        // per sequence (run_inkling_ decodes each row on the slot its request id is bound to).
        .cfg = f.model_loaded && (f.layer_loop_replaced || !f.recurrent_state || f.batched_recurrent_decode),
    };
}

}  // namespace imp
