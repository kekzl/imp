// Qwen3-VL vision encoder forward, against an independent CPU reference.
//
// The end-to-end oracle for this encoder ("the model describes the picture")
// only exists once the LM side is wired, and it does not localise a fault. So
// this test builds a small synthetic tower, runs it through the GPU encoder and
// through a from-scratch double-precision reimplementation of the reference
// semantics, and compares. Mutation-checked — each of these fails this test:
// swapping the two RoPE axes, halving the RoPE frequency exponent, reading the
// fused QKV in the wrong order, flattening the merger-norm placement, dropping
// the position embedding, dropping the attention scale, dropping a residual.
// (The merge-block token order is NOT covered here: the grid is an input to both
// sides, so it cancels. `test_qwen3vl_vision_grid.cpp` owns that one.)
//
// One thing it does NOT cover, measured rather than assumed: the block MLP's
// tanh-GELU and the mergers' erf-GELU differ by at most 4.7e-4, which is below
// one FP16 ulp at magnitude 1 (9.8e-4). Swapping them is invisible at this
// precision, so the code follows upstream on the strength of the reference, not
// of a test.
//
// The reference is written from `modeling_qwen3_vl.py`, not from the kernels.

#include "memory/vram_allocator.h"
#include "vision/qwen3vl_encoder.h"
#include "vision/qwen3vl_vision_grid.h"
#include "vision/qwen3vl_vision_upload.h"
#include "scoped_engine_arena.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace imp {
namespace {

// A tower small enough to reference on the CPU but structurally identical to the
// real one: fused QKV, two-axis RoPE, 2x2 merge, a DeepStack tap.
constexpr int kHidden = 32;
constexpr int kHeads = 4;
constexpr int kHeadDim = kHidden / kHeads;  // 8 -> half_rot 4, quarter 2
constexpr int kInter = 64;
constexpr int kDepth = 3;
constexpr int kMerge = 2;
constexpr int kOutHidden = 16;
constexpr int kPosSide = 6;
constexpr int kPatch = 2;
constexpr int kTemporal = 1;
constexpr int kFeatures = 3 * kTemporal * kPatch * kPatch;  // 12
constexpr int kUnit = kMerge * kMerge;                      // 4
constexpr int kWide = kHidden * kUnit;                      // 128
constexpr float kEps = 1e-6f;
constexpr float kTheta = 10000.0f;

// Deterministic, small, and zero-mean so the FP16 path has headroom.
struct Rng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    // Uniform in [-1, 1).
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<float>((s >> 40) & 0xFFFF) / 32767.5f - 1.0f;
    }
};

std::vector<float> randoms(Rng& rng, size_t n, float scale = 1.0f, float centre = 0.0f) {
    std::vector<float> v(n);
    for (auto& x : v)
        x = centre + scale * rng.next();
    return v;
}

// Magnitudes matter here, and not for realism's sake. With small random gains
// the LayerNorms shrink every activation, the attention logits land within
// +-0.01, softmax comes out near-uniform, and the whole rotary embedding stops
// influencing the output — a mutation that swaps the two RoPE axes then passes.
// So norms get a gain near 1 and linears get Xavier-scaled weights, which puts
// the logits in a range where attention actually chooses.
enum class Init { NormWeight, NormBias, Linear, Bias, Table };

std::vector<float> init_values(Rng& rng, Init kind, size_t n, int64_t fan_in) {
    switch (kind) {
        case Init::NormWeight:
            return randoms(rng, n, 0.15f, 1.0f);
        case Init::NormBias:
            return randoms(rng, n, 0.05f);
        case Init::Linear:
            return randoms(rng, n, std::sqrt(3.0f / static_cast<float>(std::max<int64_t>(fan_in, 1))));
        case Init::Bias:
            return randoms(rng, n, 0.05f);
        default:
            return randoms(rng, n, 0.5f);
    }
}

Tensor host_tensor(std::vector<float>& storage, std::vector<int64_t> shape) {
    Tensor t;
    t.data = storage.data();
    t.qtype = QType::F32;
    t.ndim = static_cast<int>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
        t.shape[i] = shape[i];
    t.compute_strides();
    return t;
}

// Owns every host buffer so the Tensors in the VisionModel stay valid until the
// upload has copied them.
struct SyntheticTower {
    std::vector<std::vector<float>> storage;
    VisionModel model;

    void add(Rng& rng, Init kind, Tensor& slot, std::vector<int64_t> shape) {
        size_t n = 1;
        for (int64_t d : shape)
            n *= static_cast<size_t>(d);
        storage.push_back(init_values(rng, kind, n, shape.size() > 1 ? shape[1] : 1));
        slot = host_tensor(storage.back(), std::move(shape));
    }
};

// Host copies of the weights, kept as float so the reference never reads device
// memory or FP16.
struct RefWeights {
    std::vector<float> pe_w, pe_b, pos;
    struct Layer {
        std::vector<float> ln1_w, ln1_b, qkv_w, qkv_b, o_w, o_b, ln2_w, ln2_b, up_w, up_b, dn_w, dn_b;
    };
    std::vector<Layer> layers;
    struct Merger {
        std::vector<float> nw, nb, w1, b1, w2, b2;
        bool postshuffle = false;
    };
    Merger main;
    std::vector<Merger> deep;
};

// --- reference pieces -------------------------------------------------------

void layernorm(const double* x, const float* w, const float* b, double* out, int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        const double* xr = x + static_cast<size_t>(r) * dim;
        double* o = out + static_cast<size_t>(r) * dim;
        double mean = 0.0;
        for (int j = 0; j < dim; ++j)
            mean += xr[j];
        mean /= dim;
        double var = 0.0;
        for (int j = 0; j < dim; ++j)
            var += (xr[j] - mean) * (xr[j] - mean);
        var /= dim;
        const double inv = 1.0 / std::sqrt(var + kEps);
        for (int j = 0; j < dim; ++j)
            o[j] = (xr[j] - mean) * inv * w[j] + b[j];
    }
}

// out[rows, N] = x[rows, K] @ W[N, K]^T + bias[N]
void linear(const double* x, const float* w, const float* b, double* out, int rows, int N, int K) {
    for (int r = 0; r < rows; ++r) {
        for (int n = 0; n < N; ++n) {
            double acc = b ? b[n] : 0.0;
            for (int k = 0; k < K; ++k)
                acc += x[static_cast<size_t>(r) * K + k] * w[static_cast<size_t>(n) * K + k];
            out[static_cast<size_t>(r) * N + n] = acc;
        }
    }
}

double gelu_tanh(double v) {
    return 0.5 * v * (1.0 + std::tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
}
double gelu_erf(double v) { return 0.5 * v * (1.0 + std::erf(v * 0.7071067811865476)); }

void merger_ref(const RefWeights::Merger& m, const std::vector<double>& hidden, int tokens,
                std::vector<double>& out) {
    const int merged = tokens / kUnit;
    std::vector<double> normed(static_cast<size_t>(tokens) * kHidden);
    if (m.postshuffle)
        layernorm(hidden.data(), m.nw.data(), m.nb.data(), normed.data(), merged, kWide);
    else
        layernorm(hidden.data(), m.nw.data(), m.nb.data(), normed.data(), tokens, kHidden);

    std::vector<double> fc1(static_cast<size_t>(merged) * kWide);
    linear(normed.data(), m.w1.data(), m.b1.data(), fc1.data(), merged, kWide, kWide);
    for (auto& v : fc1)
        v = gelu_erf(v);
    out.assign(static_cast<size_t>(merged) * kOutHidden, 0.0);
    linear(fc1.data(), m.w2.data(), m.b2.data(), out.data(), merged, kOutHidden, kWide);
}

void reference_forward(const RefWeights& W, const std::vector<float>& patches, const QwenVisionGrid& grid,
                       std::vector<double>& out, std::vector<std::vector<double>>& deep_out) {
    const int n = grid.tokens;
    std::vector<double> x(static_cast<size_t>(n) * kFeatures);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = patches[i];

    std::vector<double> h(static_cast<size_t>(n) * kHidden);
    linear(x.data(), W.pe_w.data(), W.pe_b.data(), h.data(), n, kHidden, kFeatures);
    for (int i = 0; i < n; ++i)
        for (int t = 0; t < kQwenVisionPosTaps; ++t) {
            const double wt = grid.pos_weights[static_cast<size_t>(i) * kQwenVisionPosTaps + t];
            const int idx = grid.pos_taps[static_cast<size_t>(i) * kQwenVisionPosTaps + t];
            for (int j = 0; j < kHidden; ++j)
                h[static_cast<size_t>(i) * kHidden + j] += wt * W.pos[static_cast<size_t>(idx) * kHidden + j];
        }

    const int half_rot = kHeadDim / 2;
    const int quarter = half_rot / 2;
    size_t tap = 0;
    for (int l = 0; l < kDepth; ++l) {
        const auto& L = W.layers[static_cast<size_t>(l)];
        std::vector<double> normed(h.size());
        layernorm(h.data(), L.ln1_w.data(), L.ln1_b.data(), normed.data(), n, kHidden);
        std::vector<double> qkv(static_cast<size_t>(n) * 3 * kHidden);
        linear(normed.data(), L.qkv_w.data(), L.qkv_b.data(), qkv.data(), n, 3 * kHidden, kHidden);

        // [tokens, 3, heads, head_dim] -> per-head q/k/v, then the two-axis
        // rotation: first quarter of the head follows the row, second the column.
        std::vector<double> q(static_cast<size_t>(kHeads) * n * kHeadDim), k(q.size()), v(q.size());
        for (int i = 0; i < n; ++i) {
            for (int hd = 0; hd < kHeads; ++hd) {
                const size_t src = static_cast<size_t>(i) * 3 * kHidden + hd * kHeadDim;
                const size_t dst = (static_cast<size_t>(hd) * n + i) * kHeadDim;
                for (int d = 0; d < kHeadDim; ++d)
                    v[dst + d] = qkv[src + 2 * kHidden + d];
                for (int j = 0; j < half_rot; ++j) {
                    const int fi = (j < quarter) ? j : (j - quarter);
                    const double pos = (j < quarter) ? grid.row[i] : grid.col[i];
                    const double inv_freq = std::pow(kTheta, -static_cast<double>(2 * fi) / half_rot);
                    const double ang = pos * inv_freq;
                    const double cs = std::cos(ang), sn = std::sin(ang);
                    const double q0 = qkv[src + j], q1 = qkv[src + j + half_rot];
                    q[dst + j] = q0 * cs - q1 * sn;
                    q[dst + j + half_rot] = q1 * cs + q0 * sn;
                    const double k0 = qkv[src + kHidden + j], k1 = qkv[src + kHidden + j + half_rot];
                    k[dst + j] = k0 * cs - k1 * sn;
                    k[dst + j + half_rot] = k1 * cs + k0 * sn;
                }
            }
        }

        // Full bidirectional attention over the image's tokens.
        std::vector<double> attn(static_cast<size_t>(n) * kHidden);
        const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
        for (int hd = 0; hd < kHeads; ++hd) {
            for (int i = 0; i < n; ++i) {
                std::vector<double> s(static_cast<size_t>(n));
                double m = -1e300;
                for (int j = 0; j < n; ++j) {
                    double acc = 0.0;
                    for (int d = 0; d < kHeadDim; ++d)
                        acc += q[(static_cast<size_t>(hd) * n + i) * kHeadDim + d] *
                               k[(static_cast<size_t>(hd) * n + j) * kHeadDim + d];
                    s[static_cast<size_t>(j)] = acc * scale;
                    m = std::max(m, s[static_cast<size_t>(j)]);
                }
                double sum = 0.0;
                for (auto& e : s) {
                    e = std::exp(e - m);
                    sum += e;
                }
                for (int d = 0; d < kHeadDim; ++d) {
                    double acc = 0.0;
                    for (int j = 0; j < n; ++j)
                        acc += s[static_cast<size_t>(j)] *
                               v[(static_cast<size_t>(hd) * n + j) * kHeadDim + d];
                    attn[static_cast<size_t>(i) * kHidden + hd * kHeadDim + d] = acc / sum;
                }
            }
        }

        std::vector<double> proj(h.size());
        linear(attn.data(), L.o_w.data(), L.o_b.data(), proj.data(), n, kHidden, kHidden);
        for (size_t i = 0; i < h.size(); ++i)
            h[i] += proj[i];

        layernorm(h.data(), L.ln2_w.data(), L.ln2_b.data(), normed.data(), n, kHidden);
        std::vector<double> ff(static_cast<size_t>(n) * kInter);
        linear(normed.data(), L.up_w.data(), L.up_b.data(), ff.data(), n, kInter, kHidden);
        for (auto& e : ff)
            e = gelu_tanh(e);
        linear(ff.data(), L.dn_w.data(), L.dn_b.data(), proj.data(), n, kHidden, kInter);
        for (size_t i = 0; i < h.size(); ++i)
            h[i] += proj[i];

        if (tap < W.deep.size() && l == 1) {  // the tap this tower declares
            deep_out.emplace_back();
            merger_ref(W.deep[tap], h, n, deep_out.back());
            ++tap;
        }
    }
    merger_ref(W.main, h, n, out);
}

// --- fixture ----------------------------------------------------------------

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void fill_config(VisionConfig& c) {
    c.is_qwen3vl = true;
    c.num_layers = kDepth;
    c.hidden_size = kHidden;
    c.num_heads = kHeads;
    c.head_dim = kHeadDim;
    c.intermediate_size = kInter;
    c.patch_size = kPatch;
    c.merge_size = kMerge;
    c.temporal_patch_size = kTemporal;
    c.out_hidden_size = kOutHidden;
    c.pos_embed_grid = kPosSide;
    c.deepstack_indexes = {1};
}

// `grid_h` x `grid_w` patches through both implementations.
void run_case(int grid_h, int grid_w) {
    const int kTokens = grid_h * grid_w;
    const int kMerged = kTokens / kUnit;

    Rng rng;
    SyntheticTower tower;
    fill_config(tower.model.config);
    tower.model.layers.resize(kDepth);
    tower.model.deepstack_mergers.resize(1);

    RefWeights W;
    W.layers.resize(kDepth);

    // Every weight is generated once and handed to both sides: the tower (which
    // gets uploaded as FP16) and the reference (which keeps the float values).
    auto add = [&](Init kind, Tensor& slot, std::vector<int64_t> shape) -> std::vector<float> {
        tower.add(rng, kind, slot, std::move(shape));
        return tower.storage.back();
    };

    W.pe_w = add(Init::Linear, tower.model.patch_embd_w, {kHidden, kFeatures});
    W.pe_b = add(Init::Bias, tower.model.patch_embd_b, {kHidden});
    W.pos = add(Init::Table, tower.model.position_embd, {kPosSide * kPosSide, kHidden});

    for (int l = 0; l < kDepth; ++l) {
        VisionLayerWeights& L = tower.model.layers[static_cast<size_t>(l)];
        RefWeights::Layer& R = W.layers[static_cast<size_t>(l)];
        R.ln1_w = add(Init::NormWeight, L.ln1_w, {kHidden});
        R.ln1_b = add(Init::NormBias, L.ln1_b, {kHidden});
        R.qkv_w = add(Init::Linear, L.wq, {3 * kHidden, kHidden});
        R.qkv_b = add(Init::Bias, L.bq, {3 * kHidden});
        R.o_w = add(Init::Linear, L.wo, {kHidden, kHidden});
        R.o_b = add(Init::Bias, L.bo, {kHidden});
        R.ln2_w = add(Init::NormWeight, L.ln2_w, {kHidden});
        R.ln2_b = add(Init::NormBias, L.ln2_b, {kHidden});
        R.up_w = add(Init::Linear, L.ffn_up_w, {kInter, kHidden});
        R.up_b = add(Init::Bias, L.ffn_up_b, {kInter});
        R.dn_w = add(Init::Linear, L.ffn_down_w, {kHidden, kInter});
        R.dn_b = add(Init::Bias, L.ffn_down_b, {kHidden});
    }

    auto fill_merger = [&](VisionMergerWeights& m, RefWeights::Merger& r, bool postshuffle) {
        r.postshuffle = postshuffle;
        const int norm_w = postshuffle ? kWide : kHidden;
        r.nw = add(Init::NormWeight, m.norm_w, {norm_w});
        r.nb = add(Init::NormBias, m.norm_b, {norm_w});
        r.w1 = add(Init::Linear, m.fc1_w, {kWide, kWide});
        r.b1 = add(Init::Bias, m.fc1_b, {kWide});
        r.w2 = add(Init::Linear, m.fc2_w, {kOutHidden, kWide});
        r.b2 = add(Init::Bias, m.fc2_b, {kOutHidden});
    };
    fill_merger(tower.model.merger, W.main, false);
    W.deep.resize(1);
    fill_merger(tower.model.deepstack_mergers[0], W.deep[0], true);

    // The tower is a T2 arena tenant now, so this test has to open an arena the
    // way Engine::init does. Without one, take_bytes() returns empty and the
    // upload fails — which is the intended signal, not a fallback.
    ScopedEngineArena arena(64ull << 20);
    ASSERT_TRUE(arena.opened());
    // The test's own scratch stays on VRAMAllocator — only the tower moved.
    VRAMAllocator alloc;
    ASSERT_TRUE(alloc.init(0.10f));
    size_t bytes = 0;
    std::string err;
    ASSERT_TRUE(qwen3vl_upload_vision_tower(tower.model, bytes, err)) << err;
    EXPECT_GT(bytes, 0u);

    QwenVisionGrid grid;
    ASSERT_TRUE(qwen3vl_build_vision_grid(grid_h, grid_w, kMerge, kPosSide, grid, err)) << err;
    ASSERT_EQ(grid.tokens, kTokens);

    const std::vector<float> patches = randoms(rng, static_cast<size_t>(kTokens) * kFeatures, 1.0f);
    std::vector<half> patches_h(patches.size());
    for (size_t i = 0; i < patches.size(); ++i)
        patches_h[i] = __float2half(patches[i]);

    half* d_patches = static_cast<half*>(alloc.allocate(patches_h.size() * sizeof(half), "test_patches"));
    half* d_out = static_cast<half*>(
        alloc.allocate(static_cast<size_t>(kMerged) * kOutHidden * sizeof(half), "test_out"));
    half* d_deep = static_cast<half*>(
        alloc.allocate(static_cast<size_t>(kMerged) * kOutHidden * sizeof(half), "test_deep"));
    ASSERT_NE(d_patches, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_deep, nullptr);
    ASSERT_EQ(cudaMemcpy(d_patches, patches_h.data(), patches_h.size() * sizeof(half),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    Qwen3VLEncoder enc;
    ASSERT_TRUE(enc.init(tower.model, &alloc, kTokens));
    EXPECT_EQ(enc.merged_tokens(kTokens), kMerged);
    ASSERT_TRUE(enc.encode(d_patches, grid, d_out, {d_deep}, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<half> got(static_cast<size_t>(kMerged) * kOutHidden);
    std::vector<half> got_deep(got.size());
    ASSERT_EQ(cudaMemcpy(got.data(), d_out, got.size() * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(got_deep.data(), d_deep, got_deep.size() * sizeof(half), cudaMemcpyDeviceToHost),
              cudaSuccess);

    std::vector<double> want;
    std::vector<std::vector<double>> want_deep;
    reference_forward(W, patches, grid, want, want_deep);
    ASSERT_EQ(want.size(), got.size());
    ASSERT_EQ(want_deep.size(), 1u);

    auto compare = [](const std::vector<half>& g, const std::vector<double>& w, const char* what) {
        double scale = 1e-3;
        for (double v : w)
            scale = std::max(scale, std::fabs(v));
        for (size_t i = 0; i < w.size(); ++i)
            EXPECT_NEAR(__half2float(g[i]), w[i], 0.03 * scale) << what << " element " << i;
    };
    compare(got, want, "merged output");
    compare(got_deep, want_deep[0], "deepstack output");

    // A guard against a reference that is trivially satisfiable: the two mergers
    // must not agree, or "matches" would mean nothing.
    double spread = 0.0;
    for (size_t i = 0; i < want.size(); ++i)
        spread = std::max(spread, std::fabs(want[i] - want_deep[0][i]));
    EXPECT_GT(spread, 1e-3) << "main and DeepStack outputs are indistinguishable";

    enc.free_buffers();
    alloc.free(d_patches);
    alloc.free(d_out);
    alloc.free(d_deep);
    qwen3vl_release_vision_tower(tower.model);
}

TEST(Qwen3VLEncoder, MatchesAnIndependentCpuReference) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";
    run_case(4, 6);
}

// Attention is chunked over query rows, so a single-chunk image never exercises
// the strides that advance between chunks. This grid crosses the boundary.
TEST(Qwen3VLEncoder, MatchesTheReferenceAcrossAttentionChunks) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";
    run_case(34, 36);  // 1224 patches > one 512-row chunk
}

}  // namespace
}  // namespace imp
