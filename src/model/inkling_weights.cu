// Inkling2-Loop loader: vendor tensor names, BF16 -> FP16 / FP32, fused 3-D expert tensors kept quantized.

#include "model/inkling_weights.h"

#include "compute/inkling_ops.h"
#include "core/logging.h"
#include "memory/vram_allocator.h"
#include "model/json_util.h"
#include "model/safetensors_loader.h"

#include <cuda_runtime.h>

#include <sys/mman.h>

#include <cstring>
#include <filesystem>
#include <unordered_map>

namespace imp {

namespace {

float bf16_host(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

struct Loader {
    std::unordered_map<std::string, Tensor> t;
    InklingWeights& w;
    bool ok = true;

    const Tensor* find(const std::string& name, QType qt, int64_t numel) {
        const auto it = t.find(name);
        if (it == t.end()) {
            IMP_LOG_ERROR("inkling: tensor %s missing", name.c_str());
            ok = false;
            return nullptr;
        }
        const Tensor& x = it->second;
        if ((qt != QType::NONE && x.qtype != qt) || x.numel() != numel) {
            IMP_LOG_ERROR("inkling: tensor %s is %s with %lld elements, expected %s with %lld", name.c_str(),
                          qtype_name(x.qtype), static_cast<long long>(x.numel()), qtype_name(qt),
                          static_cast<long long>(numel));
            ok = false;
            return nullptr;
        }
        return &x;
    }
    void* dev(size_t bytes) {
        void* p = w.va ? w.va->allocate(bytes, "inkling_weights") : nullptr;
        if (!p && (w.va || cudaMalloc(&p, bytes) != cudaSuccess)) {
            IMP_LOG_ERROR("inkling: device allocation of %.1f MiB failed", bytes / 1048576.0);
            ok = false;
            return nullptr;
        }
        w.owned.push_back(p);
        w.device_bytes += bytes;
        return p;
    }
    void* raw(const std::string& name, QType qt, int64_t numel) {
        const Tensor* x = find(name, qt, numel);
        void* p = x ? dev(x->nbytes()) : nullptr;
        if (p && cudaMemcpy(p, x->data, x->nbytes(), cudaMemcpyHostToDevice) != cudaSuccess)
            ok = false;
        return p;
    }
    half* f16(const std::string& name, int64_t numel) {
        auto* p = static_cast<half*>(raw(name, QType::BF16, numel));
        if (p)
            inkling::bf16_to_f16(reinterpret_cast<const uint16_t*>(p), p, static_cast<size_t>(numel), nullptr);
        return p;
    }
    // BF16 or F32 on disk -> fp32 on the device.
    float* f32(const std::string& name, int64_t numel) {
        const Tensor* x = find(name, QType::NONE, numel);
        if (!x)
            return nullptr;
        std::vector<float> h(static_cast<size_t>(numel));
        if (x->qtype == QType::F32) {
            std::memcpy(h.data(), x->data, h.size() * sizeof(float));
        } else if (x->qtype == QType::BF16) {
            const auto* b = static_cast<const uint16_t*>(x->data);
            for (size_t i = 0; i < h.size(); ++i)
                h[i] = bf16_host(b[i]);
        } else {
            IMP_LOG_ERROR("inkling: tensor %s has dtype %s", name.c_str(), qtype_name(x->qtype));
            ok = false;
            return nullptr;
        }
        return upload(h);
    }
    float* upload(const std::vector<float>& h) {
        auto* p = static_cast<float*>(dev(h.size() * sizeof(float)));
        if (p && cudaMemcpy(p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            ok = false;
        return p;
    }
    std::vector<float> host_f32(const std::string& name, int64_t numel) {
        const Tensor* x = find(name, QType::F32, numel);
        std::vector<float> h(static_cast<size_t>(numel), 0.0f);
        if (x)
            std::memcpy(h.data(), x->data, h.size() * sizeof(float));
        return h;
    }

    // One fused expert tensor [E, rows, K]: NVFP4 (U8 nibbles + FP8 block scales + amax / denominator
    // per expert) or MXFP8 (FP8 values + one exponent byte per 32).
    void experts(const std::string& base, int64_t rows, int64_t K, bool fp8, NvFP4MoEQuantResult& nv,
                 uint8_t*& q, uint8_t*& s) {
        const int64_t E = w.cfg.n_experts;
        if (fp8) {
            q = static_cast<uint8_t*>(raw(base, QType::FP8_E4M3, E * rows * K));
            s = static_cast<uint8_t*>(raw(base + "_scale", QType::NONE, E * rows * (K / 32)));
            return;
        }
        nv.packed_data = raw(base, QType::INT8, E * rows * (K / 2));
        nv.micro_scales = raw(base + "_scale", QType::FP8_E4M3, E * rows * (K / 16));
        std::vector<float> amax = host_f32(base + "_amax", E);
        const std::vector<float> den = host_f32(base + "_outer_scale_denominator", 1);
        for (float& a : amax)
            a /= den[0];
        nv.tensor_scales = upload(amax);
        nv.n_experts = static_cast<int>(E);
        nv.N = rows;
        nv.K = K;
        nv.expert_stride_packed = static_cast<size_t>(rows * (K / 2));
        nv.expert_stride_ms = static_cast<size_t>(rows * (K / 16));
        nv.borrowed = true;
    }
};

}  // namespace

InklingWeights::~InklingWeights() {
    for (void* p : owned)
        va ? va->free(p) : static_cast<void>(cudaFree(p));
}

bool parse_inkling_config(const std::string& dir, InklingConfig& cfg) {
    JValue root;
    if (!parse_json_file(dir + "/config.json", root))
        return false;
    const JValue* tc = jobj_find(root, "text_config");
    if (!tc || tc->type != JType::OBJECT) {
        IMP_LOG_ERROR("inkling: config.json has no text_config");
        return false;
    }
    jobj_opt_int(*tc, "num_hidden_layers", cfg.n_layers);
    jobj_opt_int(*tc, "hidden_size", cfg.hidden);
    jobj_opt_int(*tc, "num_attention_heads", cfg.n_heads);
    jobj_opt_int(*tc, "num_key_value_heads", cfg.n_kv_heads);
    jobj_opt_int(*tc, "head_dim", cfg.head_dim);
    jobj_opt_int(*tc, "d_rel", cfg.d_rel);
    jobj_opt_int(*tc, "rel_extent", cfg.rel_extent);
    jobj_opt_int(*tc, "sliding_window_size", cfg.window);
    jobj_opt_int(*tc, "sconv_kernel_size", cfg.sconv_kernel);
    jobj_opt_int(*tc, "unpadded_vocab_size", cfg.vocab);
    jobj_opt_int(*tc, "vocab_size", cfg.vocab_padded);
    jobj_opt_int(*tc, "n_routed_experts", cfg.n_experts);
    jobj_opt_int(*tc, "num_experts_per_tok", cfg.top_k);
    jobj_opt_int(*tc, "intermediate_size", cfg.inter);
    jobj_opt_int(*tc, "n_layers_per_attn_res_block", cfg.block_layers);
    jobj_opt_float(*tc, "rms_norm_eps", cfg.eps);
    jobj_opt_float(*tc, "qk_l2_clip_threshold", cfg.qk_clip);
    jobj_opt_float(*tc, "rel_l2_clip_threshold", cfg.rel_clip);
    jobj_opt_float(*tc, "log_scaling_alpha", cfg.log_alpha);
    jobj_opt_float(*tc, "log_scaling_n_floor", cfg.log_floor);
    jobj_opt_float(*tc, "mlp_gate_softcap", cfg.gate_cap);
    jobj_opt_float(*tc, "mlp_up_softcap", cfg.up_cap);
    jobj_opt_float(*tc, "route_scale", cfg.route_scale);
    jobj_opt_float(*tc, "final_logit_softcapping", cfg.logit_cap);
    if (const JValue* lc = jobj_find(*tc, "loop_config"); lc && lc->type == JType::OBJECT) {
        jobj_opt_int(*lc, "core_start", cfg.loop_start);
        jobj_opt_int(*lc, "core_end", cfg.loop_end);
        jobj_opt_int(*lc, "num_passes", cfg.passes);
        jobj_opt_int(*lc, "gate_rank", cfg.gate_rank);
    }
    cfg.global.assign(static_cast<size_t>(cfg.n_layers), true);
    const JValue* local = jobj_find(*tc, "local_layer_ids");
    if (!local || local->type != JType::ARRAY) {
        IMP_LOG_ERROR("inkling: text_config has no local_layer_ids");
        return false;
    }
    for (const auto& v : local->arr)
        if (v.type == JType::NUMBER && v.as_int() >= 0 && v.as_int() < cfg.n_layers)
            cfg.global[static_cast<size_t>(v.as_int())] = false;
    const bool sane = cfg.sconv_kernel == 4 && cfg.n_experts <= 128 && cfg.top_k <= 16 && cfg.head_dim <= 128 &&
                      cfg.d_rel <= 32 && cfg.passes >= 1 && cfg.loop_start % cfg.block_layers == 0 &&
                      cfg.loop_end > cfg.loop_start && cfg.loop_end <= cfg.n_layers;
    if (!sane)
        IMP_LOG_ERROR("inkling: config outside what this build runs (sconv kernel, experts, head_dim, loop)");
    return sane;
}

bool load_inkling_weights(const std::string& dir, InklingWeights& w, bool with_vocab, VRAMAllocator* va) {
    w.va = va;
    if (!parse_inkling_config(dir, w.cfg))
        return false;
    const InklingConfig& c = w.cfg;
    std::string file;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.rfind("model-", 0) == 0 && n.size() > 12 && n.substr(n.size() - 12) == ".safetensors")
            file = e.path().string();
    }
    Loader L{{}, w};
    void* base = nullptr;
    size_t size = 0;
    if (file.empty() || !load_raw_safetensors_file(file, L.t, base, size)) {
        IMP_LOG_ERROR("inkling: no readable model-*.safetensors in %s", dir.c_str());
        return false;
    }
    const int64_t H = c.hidden, QD = static_cast<int64_t>(c.n_heads) * c.head_dim;
    const int64_t KD = static_cast<int64_t>(c.n_kv_heads) * c.head_dim, I = c.inter;
    const std::string llm = "model.llm.";
    if (with_vocab) {
        w.embed = L.f16(llm + "embed.weight", static_cast<int64_t>(c.vocab_padded) * H);
        w.unembed = L.f16(llm + "unembed.weight", static_cast<int64_t>(c.vocab_padded) * H);
    }
    w.embed_norm = L.f32(llm + "embed_norm.weight", H);
    w.norm = L.f32(llm + "norm.weight", H);
    w.out_query = L.f32(llm + "attn_res.output_query", H);
    {
        // Disk: [2 * loop layers, H] as (q, mlp) and (k, v) pairs -> [loop layers, (q, k, v, mlp), H].
        const int64_t nl = c.loop_end - c.loop_start;
        const std::vector<float> qm = L.host_f32(llm + "middle_cycle_cross_pass_queries", 2 * nl * H);
        const std::vector<float> kv = L.host_f32(llm + "middle_cycle_cross_pass_kv_queries", 2 * nl * H);
        std::vector<float> m(static_cast<size_t>(4 * nl * H));
        for (int64_t l = 0; l < nl; ++l) {
            std::memcpy(&m[(l * 4 + 0) * H], &qm[(l * 2 + 0) * H], H * sizeof(float));
            std::memcpy(&m[(l * 4 + 1) * H], &kv[(l * 2 + 0) * H], H * sizeof(float));
            std::memcpy(&m[(l * 4 + 2) * H], &kv[(l * 2 + 1) * H], H * sizeof(float));
            std::memcpy(&m[(l * 4 + 3) * H], &qm[(l * 2 + 1) * H], H * sizeof(float));
        }
        w.msg_query = L.upload(m);
    }
    w.layers.resize(static_cast<size_t>(c.n_layers));
    for (int i = 0; i < c.n_layers && L.ok; ++i) {
        InklingLayer& ly = w.layers[static_cast<size_t>(i)];
        const bool g = c.global[static_cast<size_t>(i)];
        const std::string p = llm + "layers." + std::to_string(i) + ".";
        ly.wq = L.f16(p + "attn.wq_du.weight", QD * H);
        ly.wk = L.f16(p + "attn.wk_dv.weight", KD * H);
        ly.wv = L.f16(p + "attn.wv_dv.weight", KD * H);
        ly.wo = L.f16(p + "attn.wo_ud.weight", H * QD);
        ly.gate_attn = L.f16(p + "attn.gate_attn.weight", QD * H);
        ly.wr = L.f16(p + "attn.wr_du.weight", static_cast<int64_t>(c.n_heads) * c.d_rel * H);
        ly.sink = L.f16(p + "attn.attn_sink_proj.weight", c.n_heads * H);
        ly.q_scale = L.f32(p + "attn.q_l2_clip.scale", QD);
        ly.rel_scale = L.f32(p + "attn.rel_l2_clip.scale", static_cast<int64_t>(c.n_heads) * c.d_rel);
        ly.rel_proj = L.f32(p + "attn.rel_logits_proj.proj", static_cast<int64_t>(c.d_rel) * c.rel_extent);
        if (g) {
            ly.norm_q = L.f32(p + "q_input_norm.weight", H);
            ly.norm_k = L.f32(p + "k_input_norm.weight", H);
            ly.norm_v = L.f32(p + "v_input_norm.weight", H);
            ly.k_conv.static_w = L.f32(p + "attn.k_sconv.static_w", 4 * KD);
            ly.v_conv.static_w = L.f32(p + "attn.v_sconv.static_w", 4 * KD);
            ly.k_conv.proj = L.f16(p + "attn.k_sconv.weight_proj", static_cast<int64_t>(c.n_kv_heads) * 4 * H);
            ly.v_conv.proj = L.f16(p + "attn.v_sconv.weight_proj", static_cast<int64_t>(c.n_kv_heads) * 4 * H);
        } else {
            ly.norm_q = ly.norm_k = ly.norm_v = L.f32(p + "attn_norm.weight", H);
            ly.k_conv.w = L.f32(p + "attn.k_sconv.weight", KD * 4);
            ly.v_conv.w = L.f32(p + "attn.v_sconv.weight", KD * 4);
        }
        ly.mlp_norm = L.f32(p + "mlp_norm.weight", H);
        ly.attn_conv = L.f32(p + "attn_sconv.weight", H * 4);
        ly.mlp_conv = L.f32(p + "mlp_sconv.weight", H * 4);
        ly.router_w = L.f32(p + "mlp.gate.weight", static_cast<int64_t>(c.n_experts) * H);
        ly.router_b = L.f32(p + "mlp.gate.bias", c.n_experts);
        if (i > 0)
            ly.attn_query = L.f32(llm + "attn_res.attn_queries." + std::to_string(i - 1), (g ? 3 : 1) * H);
        ly.mlp_query = L.f32(llm + "attn_res.mlp_queries." + std::to_string(i), H);
        if (c.is_loop(i)) {
            static const char* const kGate[4] = {"attn", "k", "v", "mlp"};
            for (int k = 0; k < 4; ++k) {
                const std::string gp = p + "controllers." + kGate[k] + ".";
                ly.gates[k].down = L.f16(gp + "down_weight", static_cast<int64_t>(c.gate_rank) * 2 * H);
                ly.gates[k].forget = L.f16(gp + "forget_weight", H * c.gate_rank);
                ly.gates[k].gain = L.f16(gp + "gain_weight", H * c.gate_rank);
            }
        }
        const auto it = L.t.find(p + "mlp.experts.w13_weight");
        ly.ex.fp8 = it != L.t.end() && it->second.qtype == QType::FP8_E4M3;
        L.experts(p + "mlp.experts.w13_weight", 2 * I, H, ly.ex.fp8, ly.ex.w13, ly.ex.w13_q, ly.ex.w13_s);
        L.experts(p + "mlp.experts.w2_weight", H, I, ly.ex.fp8, ly.ex.w2, ly.ex.w2_q, ly.ex.w2_s);
    }
    cudaDeviceSynchronize();
    munmap(base, size);
    if (L.ok)
        IMP_LOG_INFO("inkling: %d layers (%d applications), %.2f GiB on the device", c.n_layers, c.n_apps(),
                     w.device_bytes / 1073741824.0);
    return L.ok;
}

}  // namespace imp
