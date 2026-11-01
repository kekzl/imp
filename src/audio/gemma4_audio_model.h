#pragma once
// Gemma-4 audio tower (HF Gemma4AudioModel, USM Conformer) + audio embedder, from the
// checkpoint's model.audio_tower.* / model.embed_audio.* tensors. Host tensors after load;
// upload converts linear weights to FP16, everything else (norms, convs, derived tables) to FP32.

#include "core/tensor.h"

#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {

struct JValue;

struct Gemma4AudioConfig {
    int hidden_size = 1024;
    int num_heads = 8;
    int num_layers = 12;
    int chunk_size = 12;      // attention_chunk_size
    int context_left = 13;    // attention_context_left: keys q-(left-2)..q attend
    int context_right = 0;    // attention_context_right
    float logit_cap = 50.0f;  // attention_logit_cap
    int conv_kernel = 5;      // conformer depthwise conv
    int output_proj_dims = 1536;
    float residual_weight = 0.5f;
    float rms_norm_eps = 1e-6f;
    int sub_channels0 = 128;  // subsampling_conv_channels
    int sub_channels1 = 32;
    bool clipped_linears = true;
    float gradient_clipping = 1e10f;
    int n_mels = 128;
    int lm_d_model = 0;  // text hidden size, embedder output
    int32_t audio_token_id = -1;
    int32_t boa_token_id = -1;
    int32_t eoa_token_id = -1;

    int head_dim() const { return hidden_size / num_heads; }
    // Relative-position rows: distances context_size/2 .. 0 (HF Gemma4AudioRelPositionalEncoding).
    int rel_positions() const { return ((chunk_size + context_left - 1 + context_right) / 2) + 1; }
};

// Gemma4ClippableLinear: clamp input, matmul, clamp output (scalars from the checkpoint).
struct Gemma4AudioLinear {
    Tensor w;  // [out, in]
    float in_min = -std::numeric_limits<float>::infinity();
    float in_max = std::numeric_limits<float>::infinity();
    float out_min = -std::numeric_limits<float>::infinity();
    float out_max = std::numeric_limits<float>::infinity();
};

struct Gemma4AudioFfw {
    Tensor pre_norm, post_norm;  // [hidden]
    Gemma4AudioLinear up, down;  // [4h, h], [h, 4h]
};

struct Gemma4AudioLayer {
    Gemma4AudioFfw ffw1, ffw2;
    Tensor norm_pre_attn, norm_post_attn, norm_out;
    Gemma4AudioLinear q, k, v, post;
    Tensor rel_k;    // FP32 [rel_positions, hidden] = relative_k_proj(sinusoid table), derived at load
    Tensor q_scale;  // FP32 [head_dim] = head_dim^-0.5 / ln2 * softplus(per_dim_scale), derived at load
    Tensor lconv_pre_norm, lconv_conv_norm;
    Tensor dw_conv;                        // [hidden, 1, kernel]
    Gemma4AudioLinear lin_start, lin_end;  // [2h, h], [h, h]
};

struct Gemma4AudioModel {
    Gemma4AudioConfig config;
    Tensor sub0_conv, sub0_norm;  // [c0, 1, 3, 3], [c0]
    Tensor sub1_conv, sub1_norm;  // [c1, c0, 3, 3], [c1]
    Tensor input_proj;            // [hidden, (c0/4)*c1]
    std::vector<Gemma4AudioLayer> layers;
    Tensor out_proj_w, out_proj_b;               // [out_dims, hidden], [out_dims]
    Tensor embed_proj;                           // [lm_d_model, out_dims]
    std::vector<std::vector<float>> host_owned;  // derived FP32 tables the Tensors point into
};

// audio_config is a gemma4_audio object: parse it with the token ids and the text hidden size.
// nullopt-like: unexpected("") when the checkpoint declares no Gemma-4 audio tower.
[[nodiscard]] std::expected<Gemma4AudioConfig, std::string> parse_gemma4_audio_config(const JValue& root);

// Fills every slot from the shard map and derives rel_k / q_scale. Refuses on a missing tensor
// or a shape that contradicts the config.
[[nodiscard]] std::expected<std::unique_ptr<Gemma4AudioModel>, std::string> load_gemma4_audio_tower(
    const Gemma4AudioConfig& cfg, const std::unordered_map<std::string, Tensor>& tensors);

// <model_dir>/config.json + shard map -> tower, or null (logged) when the checkpoint has none
// or it cannot be used; the text model loads either way.
std::unique_ptr<Gemma4AudioModel> load_gemma4_audio_from_checkpoint(
    const std::string& model_dir, const std::unordered_map<std::string, Tensor>& tensors);

// True for the names load_gemma4_audio_tower reads (the LM weight map skips them).
bool is_gemma4_audio_tensor_name(const std::string& name);

// Every uploaded slot once; fp16 = linear weight (FP16 on device), else FP32.
void gemma4_audio_visit_tensors(Gemma4AudioModel& m,
                                const std::function<void(Tensor&, const std::string&, bool fp16)>& fn);

// Device bytes of the uploaded tower (shapes only, 256-byte aligned per tensor).
size_t gemma4_audio_tower_device_bytes(Gemma4AudioModel& m);

// Upload to the engine arena. On failure every slot still points at host memory.
[[nodiscard]] std::expected<size_t, std::string> gemma4_audio_upload(Gemma4AudioModel& m);

}  // namespace imp
