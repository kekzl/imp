#pragma once
// =============================================================================
// mtp_head.h — Multi-Token Predictor head storage
// =============================================================================
//
// Trained MTP head shipped alongside DeepSeek-V3-family models (Qwen3.6,
// DeepSeek V3, etc.) as `model_mtp.safetensors` in the model directory.
//
// Phase 1.A: detection metadata only.
// Phase 1.B (this expansion): named Tensor fields, weights loaded as BF16.
// Phase 2: forward kernel implementation.
// Phase 3+: verify-loop integration.
//
// Reference architecture (Qwen3.6-NVFP4 MTP, 1.6 GB BF16, 19 tensors):
//
//   FC + pre-FC norms (token-conditioning block):
//     mtp.fc                          [2048, 4096]   project concat(emb, h_prev)
//     mtp.pre_fc_norm_embedding       [2048]         RMSNorm on embedding input
//     mtp.pre_fc_norm_hidden          [2048]         RMSNorm on hidden_state input
//
//   Single transformer layer (mtp.layers.0.*):
//     input_layernorm                 [2048]
//     post_attention_layernorm        [2048]
//     self_attn.q_proj                [8192, 2048]   16 heads × 512 head_dim
//                                                    OR n_heads MQA-style variant
//     self_attn.k_proj                [512, 2048]    2 kv_heads × 256 head_dim
//     self_attn.v_proj                [512, 2048]
//     self_attn.o_proj                [2048, 4096]
//     self_attn.q_norm                [256]          per-head RMSNorm
//     self_attn.k_norm                [256]
//     mlp.gate                        [256, 2048]    256 experts router
//     mlp.experts.gate_up_proj        [256, 1024, 2048]  256 experts × (gate+up) packed
//     mlp.experts.down_proj           [256, 2048, 512]
//     mlp.shared_expert.gate_proj     [512, 2048]
//     mlp.shared_expert.up_proj       [512, 2048]
//     mlp.shared_expert.down_proj     [2048, 512]
//     mlp.shared_expert_gate          [1, 2048]      sigmoid-gated shared expert
//
//   Final norm:
//     mtp.norm                        [2048]
//
//   LM head: SHARED with main model's `model.lm_head.weight` (not stored here).
//
// Second layout (Nemotron-3.5-Lightning, 2.6 GB BF16, 270 tensors). Same idea,
// but it is a miniature Nemotron rather than a miniature Qwen: the head spans
// TWO blocks — attention in `layers.0`, MoE in `layers.1` — mirroring the
// hybrid main model, and every name differs:
//
//     mtp.layers.0.enorm / hnorm      [2688]         = pre_fc_norm_{embedding,hidden}
//     mtp.layers.0.eh_proj            [2688, 5376]   = fc (concat(emb,h) → hidden)
//     mtp.layers.0.norm               [2688]         = input_layernorm
//     mtp.layers.0.mixer.{q,k,v,o}_proj              = self_attn.* (no q/k norm)
//     mtp.layers.1.norm               [2688]         = post_attention_layernorm
//     mtp.layers.1.mixer.gate.weight  [128, 2688]    router
//     mtp.layers.1.mixer.gate.e_score_correction_bias [128]  DeepSeek-style bias
//     mtp.layers.1.mixer.experts.{e}.up_proj   [1856, 2688]  128 experts, PER-EXPERT
//     mtp.layers.1.mixer.experts.{e}.down_proj [2688, 1856]  2-D, not packed 3-D
//     mtp.layers.1.mixer.shared_experts.{up,down}_proj
//     mtp.layers.1.final_layernorm    [2688]         = final norm
//
// Two structural differences the forward pass has to honour, not just the
// names: the experts are NON-GATED (no `gate_proj` — squared-ReLU, like the
// main Nemotron FFN) and there is no sigmoid `shared_expert_gate`.
// =============================================================================

#include "core/tensor.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace imp {

// Is this checkpoint tensor part of an MTP head? Checkpoints write the head
// either as `mtp.*` or, with the outer prefix kept, as `model.mtp.*`.
//
// One rule, two callers: the divert decision in load_shard() and the presence
// probe that tells the operator about a head it is NOT loading. Two places
// asking the same question and answering differently is the defect class that
// produced #1384 and #1443, so both go through here.
inline bool name_is_mtp_tensor(std::string_view name) {
    return name.rfind("mtp.", 0) == 0 || name.rfind("model.mtp.", 0) == 0;
}

// Does this name make an `mtp.*` group a head imp can actually dispatch? It is
// the projection fusing the embedding with the hidden state, and dispatch_mtp()
// keys its two checkpoint shapes on exactly these two spellings. A probe that
// accepted any `mtp.*` name would advertise a head that enabling then rejects,
// which is a worse failure than saying nothing.
// The two spellings, as constants, because dispatch_mtp() branches on the same
// strings. Sharing them is what keeps probe and dispatch from drifting apart:
// a test could only notice the drift afterwards, a shared constant prevents it.
inline constexpr const char* kMtpHeadKeyEhProj = "mtp.layers.0.eh_proj.weight";
inline constexpr const char* kMtpHeadKeyFc = "mtp.fc.weight";

inline bool name_is_mtp_head_key(std::string_view name) {
    if (name.rfind("model.", 0) == 0)
        name.remove_prefix(6);
    return name == kMtpHeadKeyEhProj || name == kMtpHeadKeyFc;
}

// True when `model_dir` ships an MTP head, decided from tensor NAMES only: the
// sidecar file, the shard index, or the single-file header. Reads no weight
// byte, so it is cheap enough to run on a load that does not want the head.
bool probe_mtp_head(const std::string& model_dir);

// Phase 1.A leftover — kept as part of the new MtpHead struct for compatibility
// with the existing Model::mtp_info_ field.
struct MtpHeadInfo {
    std::string path;
    size_t      file_bytes = 0;
    int         n_tensors  = 0;
};

// Full MTP head storage. Populated by safetensors_loader when
// `model_mtp.safetensors` is present alongside the main weights and
// `runtime.mtp_spec_decode > 0` is enabled. Otherwise empty / .loaded=false.
struct MtpHead {
    MtpHeadInfo info;

    // Token-conditioning block:
    //   mtp_in = norm(emb(t)) || norm(h_prev)   then fc projects 4096 → 2048
    Tensor pre_fc_norm_embedding;
    Tensor pre_fc_norm_hidden;
    Tensor fc;

    // Single transformer layer (mtp.layers.0.*):
    Tensor input_layernorm;
    Tensor post_attention_layernorm;

    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;
    Tensor q_norm;
    Tensor k_norm;

    Tensor router;                          // mlp.gate.weight
    Tensor experts_gate_up_packed;          // [256, 1024, 2048] packed gate+up per expert
    Tensor experts_down_packed;             // [256, 2048, 512]

    Tensor shared_expert_gate_proj;
    Tensor shared_expert_up_proj;
    Tensor shared_expert_down_proj;
    Tensor shared_expert_gate;              // [1, hidden] sigmoid gate

    Tensor final_norm;                      // mtp.norm.weight

    // --- Nemotron-3.5 layout (see header comment) -----------------------------
    // Per-expert 2-D weights instead of the packed 3-D pair above. Empty on the
    // Qwen layout; when non-empty the forward pass indexes these directly and
    // ignores experts_*_packed.
    std::vector<Tensor> experts_up;    // [n_experts] each [d_ff_e, hidden]
    std::vector<Tensor> experts_down;  // [n_experts] each [hidden, d_ff_e]
    // The same weights restacked contiguously at upload, so the decode GEMV can
    // index them from a device-side expert id (gemv_f16_moe_decode) instead of
    // the host reading routing back and issuing one GEMM per chosen expert.
    // That host round trip is what kept the draft out of CUDA graph capture.
    // The per-expert Tensors above become views into these slabs once packing
    // has run. The slabs are a second copy in VRAM, not a re-pointing: the
    // original per-expert allocations stay tracked and are only released at
    // teardown, so the head costs file_bytes plus both slabs for the life of
    // the process. Measured on Nemotron-3.5: 6317 MiB of device free for a head
    // that is 2550 MiB on disk. Uploading straight into the slabs would remove
    // the second copy and is not done yet.
    Tensor experts_up_stacked;    // [n_experts, d_ff_e, hidden] FP16
    Tensor experts_down_stacked;  // [n_experts, hidden, d_ff_e] FP16
    // DeepSeek-style additive score bias on the router logits. Null when absent.
    Tensor router_score_bias;  // [n_experts] FP32
    // Experts have no gate_proj: activation is squared ReLU, not SwiGLU. This is
    // a property of the checkpoint, so it is recorded rather than re-derived
    // from which tensors happen to be present at each use site.
    bool experts_non_gated = false;
    // Qwen3.6's MTP attention is attn_output_gate=True: q_proj emits
    // [num_heads, 2*head_dim] and the second half gates the output. Nemotron's
    // does not, and its attention is NoPE — the hybrid's Mamba layers carry
    // position, so applying RoPE here would rotate against the main model.
    // Both default to the Qwen behaviour so that path is untouched.
    bool attn_output_gate = true;
    bool attn_rope = true;

    // Status flag set true when ALL of the above tensors are populated.
    bool loaded = false;
};

// Device VRAM the head's upload needs at its peak, in bytes.
//
// The head is BF16 on disk and FP16 on device, so file_bytes is its resident
// size. A per-expert layout additionally restacks both expert sets into one
// contiguous slab each while the per-expert allocations are still live, so a
// second copy of both sets is resident at the peak, and stays resident. See
// the note on experts_up_stacked.
//
// The first slab additionally costs twice its size, because it is the first
// large request the async pool serves and the pool rounds up when it grows.
// Probed on Nemotron-3.5, device free after each phase:
//
//   start                     12137 MiB
//   after 270 tensor uploads   9577      2560 used, against 2550 on disk
//   after slab up              7153      2424 used, for a 1218 MiB slab
//   after slab down            5937      1216 used, the slab exactly
//
// So the term is one extra slab, not a margin: this returns 6204 MiB for that
// head against 6200 measured. The allocator headroom the caller adds on top is
// left to cover allocator waste, which is what it is for. Relying on it to
// cover this would have held only on a large card: it is 5 % of the total, so
// 1630 MiB on 32 GB but 819 MiB on 16 GB, and only 409 MiB for a process
// running under an 8 GB --vram-budget.
//
// Measured over three runs per arm, comparing device free consumed by the load
// with speculative.mtp_k at 0 and 1:
//
//   Qwen3.6-35B-A3B-NVFP4   packed, 19 tensors    1608 MiB on disk, 1495 used
//   Nemotron-3.5-Lightning  per-expert, 270       2550 MiB on disk, 6317 used
//
// This returns 1608 for the packed head, a slight over-estimate, and 6204 for
// the per-expert one.
inline size_t mtp_upload_peak_bytes(const MtpHead& head) {
    constexpr size_t kFp16Bytes = 2;  // device dtype, independent of CUDA headers
    size_t bytes = head.info.file_bytes;
    size_t largest_slab = 0;
    for (const std::vector<Tensor>* parts : {&head.experts_up, &head.experts_down}) {
        if (parts->empty() || parts->front().data == nullptr)
            continue;
        const Tensor& t = parts->front();
        const size_t slab = static_cast<size_t>(t.shape[0]) * static_cast<size_t>(t.shape[1]) * kFp16Bytes *
                            parts->size();
        bytes += slab;
        largest_slab = slab > largest_slab ? slab : largest_slab;
    }
    return bytes + largest_slab;
}

}  // namespace imp
