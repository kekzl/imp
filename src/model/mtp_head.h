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
//
// Third layout (Qwen4Exp, Qwen3.8-Flash-Next, 3101 tensors). Spec:
// docs/plans/2026-09-28-qwen4exp-mtp.md. Hyper-connection head, hc_count=4:
//
//     mtp.fc_embedding / fc_hidden          [2560, 2560]  split fc, fc_hidden per hc stream
//     mtp.pre_fc_norm_hidden                [10240]       norm over the 4x2560 hc stream
//     mtp.hyper_connection_mixer.*          final mixer, no block_inject_weight
//     mtp.layers.0.{attn,mlp}_hyper_connection.*          hc read/inject
//     mtp.layers.0.self_attn.indexer.*      QSA indexer
//     mtp.layers.0.mlp.experts.{e}.{gate,up,down}_proj    512 experts, F8_E4M3
//         .weight_scale_inv                 BF16 128x128 block scales
//
// Draft forward: compute/mtp_forward_qwen4exp.cu; experts run through gemv_fp8_block_moe.
// =============================================================================

#include "core/tensor.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace imp {

// Checkpoint tensor is part of an MTP head if named mtp.* or model.mtp.* (outer prefix
// kept). One rule, two callers (load_shard's divert decision, the presence probe) - both
// must ask this shared question to avoid the #1384/#1443 defect class of disagreeing answers.
[[nodiscard]] inline bool name_is_mtp_tensor(std::string_view name) {
    return name.rfind("mtp.", 0) == 0 || name.rfind("model.mtp.", 0) == 0;
}

// Does this name make an mtp.* group a head imp can dispatch? It's the projection fusing
// embedding+hidden state; dispatch_mtp() keys its two checkpoint shapes on exactly these
// two spellings, shared as constants so probe and dispatch cannot drift apart.
inline constexpr const char* kMtpHeadKeyEhProj = "mtp.layers.0.eh_proj.weight";
inline constexpr const char* kMtpHeadKeyFc = "mtp.fc.weight";
inline constexpr const char* kMtpHeadKeyFcEmbedding = "mtp.fc_embedding.weight";

[[nodiscard]] inline bool name_is_mtp_head_key(std::string_view name) {
    if (name.rfind("model.", 0) == 0)
        name.remove_prefix(6);
    return name == kMtpHeadKeyEhProj || name == kMtpHeadKeyFc || name == kMtpHeadKeyFcEmbedding;
}

// Checkpoint layouts dispatch_mtp_head() recognises, keyed on the fusion projection name.
enum class MtpLayout { None, Qwen, Nemotron, Qwen4Exp };

// True when `model_dir` ships an MTP head, decided from tensor NAMES only: the
// sidecar file, the shard index, or the single-file header. Reads no weight
// byte, so it is cheap enough to run on a load that does not want the head.
[[nodiscard]] bool probe_mtp_head(const std::string& model_dir);

// Phase 1.A leftover — kept as part of the new MtpHead struct for compatibility
// with the existing Model::mtp_info_ field.
struct MtpHeadInfo {
    std::string path;
    size_t      file_bytes = 0;
    int         n_tensors  = 0;
    int         n_mapped   = 0;  // tensors dispatch_mtp_head() assigned to a field
};

// One hyper-connection module (GatedResidual): hc_norm [hc*d], input_mix_weight_down
// [lowrank, hc*d], input_mix_weight_up [hc*d, lowrank], block_inject_weight [hc, hc*d].
// block_inject is null on the final mixer (use_combine=False).
struct MtpHyperConnection {
    Tensor norm;
    Tensor mix_down;
    Tensor mix_up;
    Tensor block_inject;
};

// Qwen4Exp routed expert: F8_E4M3 weights, BF16 weight_scale_inv per 128x128 block.
struct MtpFp8Expert {
    Tensor gate_proj;  // [d_ff_e, hidden]
    Tensor up_proj;    // [d_ff_e, hidden]
    Tensor down_proj;  // [hidden, d_ff_e]
    Tensor gate_scale_inv;
    Tensor up_scale_inv;
    Tensor down_scale_inv;
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

    // Nemotron-3.5 layout: per-expert 2-D weights instead of the packed 3-D pair above. Empty
    // on the Qwen layout; when non-empty the forward pass indexes these directly and ignores
    // experts_*_packed.
    std::vector<Tensor> experts_up;    // [n_experts] each [d_ff_e, hidden]
    std::vector<Tensor> experts_down;  // [n_experts] each [hidden, d_ff_e]
    // Weights restacked contiguously at upload so the decode GEMV indexes by device-side
    // expert id (gemv_f16_moe_decode) instead of a host routing round trip, which had kept the
    // draft out of CUDA graph capture. Per-expert Tensors become views into these slabs; the
    // slabs are a SECOND copy in VRAM (original allocations stay tracked, released at teardown).
    Tensor experts_up_stacked;    // [n_experts, d_ff_e, hidden] FP16
    Tensor experts_down_stacked;  // [n_experts, hidden, d_ff_e] FP16
    // DeepSeek-style additive score bias on the router logits. Null when absent.
    Tensor router_score_bias;  // [n_experts] FP32
    // Experts have no gate_proj: activation is squared ReLU, not SwiGLU. This is
    // a property of the checkpoint, so it is recorded rather than re-derived
    // from which tensors happen to be present at each use site.
    bool experts_non_gated = false;
    // Qwen3.6 MTP attention is attn_output_gate=true: q_proj emits [num_heads, 2*head_dim],
    // the second half gates the output. Nemotron's is not gated and is NoPE (position lives
    // in the hybrid's Mamba layers; RoPE here would rotate against the main model).
    bool attn_output_gate = true;
    bool attn_rope = true;

    // Qwen4Exp layout only (MtpLayout::Qwen4Exp); empty on the other two. fc stays null:
    // the fusion is fc_embedding(norm(emb)) + fc_hidden applied per hc stream.
    Tensor fc_embedding;  // [hidden, hidden]
    Tensor fc_hidden;     // [hidden, hidden], shared across the hc_count streams
    MtpHyperConnection attn_hc;
    MtpHyperConnection mlp_hc;
    MtpHyperConnection final_mixer;  // hyper_connection_mixer, replaces final_norm
    Tensor indexer_qk_proj;          // QSA indexer: host only, the draft attends densely (spec OPEN 3)
    Tensor indexer_q_norm;
    Tensor indexer_k_norm;
    // Host-mapped checkpoint views; upload copies them into the device tables below.
    std::vector<MtpFp8Expert> experts_fp8;
    // Device copies for gemv_fp8_block_moe: per-expert E4M3 allocations addressed through
    // pointer tables (no multi-GiB slab), scales BF16 -> FP32 in one slab each.
    //   gate_up: table [n_experts * 2] (gate, up), scales [n_experts * 2, d_ff_e / 128, hidden / 128]
    //   down:    table [n_experts],     scales [n_experts, hidden / 128, d_ff_e / 128]
    const uint8_t* const* fp8_gate_up_tab = nullptr;
    const uint8_t* const* fp8_down_tab = nullptr;
    const float* fp8_gate_up_scales = nullptr;
    const float* fp8_down_scales = nullptr;
    int hc_count = 0;  // Qwen4Exp: width of the hc stream the head reads and writes (hc_count x hidden)

    MtpLayout layout = MtpLayout::None;

    // Status flag set true when ALL of the above tensors are populated.
    bool loaded = false;
};

// speculative.mtp_k=auto depth for this head: 0 = kMtpAutoK, < 0 = auto declines the head.
// Qwen4Exp declines: k=1 decodes 53.15 vs 65.31 tok/s k=0 (-18.6 %, 4 prompts x 3, #2272),
// head holds 2400 MiB VRAM.
inline constexpr int kMtpAutoDeclines = -1;
inline int mtp_auto_k_cap(const MtpHead& head) {
    return head.layout == MtpLayout::Qwen4Exp ? kMtpAutoDeclines : 0;
}

// Maps a raw mtp.* tensor map (outer "model." already stripped) onto MtpHead fields.
// Layout is keyed on the fusion projection name: eh_proj -> Nemotron, fc_embedding ->
// Qwen4Exp, else Qwen. `bytes` is the head's on-disk size, `src` is for logging.
MtpHead dispatch_mtp_head(const std::unordered_map<std::string, Tensor>& tm, const std::string& src,
                          size_t bytes);

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
// cover this would have held only on a large card: it is 4 % of the total, so
// 1304 MiB on 32 GB but 655 MiB on 16 GB, and only 327 MiB for a process
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
