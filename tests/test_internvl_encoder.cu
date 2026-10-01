// InternVL GPU encoder (InternViT + pixel shuffle + projector) per stage vs the pinned HF FP32
// modules on a tiny tower with realistic magnitudes (tests/fixtures/internvl). Bound: relL2 <= 10x
// the FP16 floor per stage; weights and pixels are FP16-representable (activation error only).

#include "internvl_tiny_config.h"
#include "memory/vram_allocator.h"
#include "safetensors_fixture.h"
#include "scoped_engine_arena.h"
#include "vision/internvl_encoder.h"
#include "vision/internvl_vision.h"
#include "vision/qwen3vl_vision_upload.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace imp {
namespace {

const std::string kDir = std::string(IMP_TEST_FIXTURES_DIR) + "/internvl";

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

double rel_l2(const std::vector<half>& got, const std::vector<float>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(__half2float(got[i])) - ref[i];
        num += d * d;
        den += static_cast<double>(ref[i]) * ref[i];
    }
    return std::sqrt(num / den);
}

std::vector<half> read(const half* d, size_t n) {
    std::vector<half> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}

TEST(InternVLEncoder, MatchesHfFp32PerStage) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";
    imp_test::SafetensorsFixture W, S;
    ASSERT_TRUE(W.load(kDir + "/tiny_tower.st"));
    ASSERT_TRUE(S.load(kDir + "/tiny_stages.st"));
    JsonParser p{std::string_view(imp_test::kInternVLTinyConfig)};
    const auto cfg = parse_internvl_vision_config(p.parse());
    ASSERT_TRUE(cfg) << cfg.error();
    VisionModel tower;
    tower.config = *cfg;
    ASSERT_TRUE(load_internvl_vision_tensors(W.tensors, tower));
    const VisionConfig& c = tower.config;
    const int np = c.num_patches, n = np + 1, H = c.hidden_size, side = c.pos_embed_grid, P = c.patch_size;
    const int m = c.num_image_tokens, D = c.out_hidden_size, features = 3 * P * P, img = side * P;

    ScopedEngineArena arena(64ull << 20);
    ASSERT_TRUE(arena.opened());
    VRAMAllocator alloc;
    ASSERT_TRUE(alloc.init(0.10f));
    const auto bytes = qwen3vl_upload_vision_tower(tower);
    ASSERT_TRUE(bytes) << bytes.error();

    // Patches in raster order, (channel, row, col) inside a patch.
    const std::vector<float> px = S.floats("pixel_values");
    std::vector<half> patches(static_cast<size_t>(np) * features);
    for (int pi = 0; pi < np; ++pi)
        for (int ch = 0; ch < 3; ++ch)
            for (int y = 0; y < P; ++y)
                for (int x = 0; x < P; ++x) {
                    const int py = (pi / side) * P + y, pxx = (pi % side) * P + x;
                    patches[static_cast<size_t>(pi) * features + (ch * P + y) * P + x] = __float2half(
                        px[(static_cast<size_t>(ch) * img + py) * img + pxx]);
                }

    auto dev = [&](size_t elems, const char* tag) {
        return static_cast<half*>(alloc.allocate(elems * sizeof(half), tag));
    };
    half* d_patches = dev(patches.size(), "internvl_test_patches");
    half* d_out = dev(static_cast<size_t>(m) * D, "internvl_test_out");
    InternVLStageTaps taps;
    taps.embeddings = dev(static_cast<size_t>(n) * H, "internvl_test_emb");
    for (int l = 0; l < c.num_layers; ++l)
        taps.layers.push_back(dev(static_cast<size_t>(n) * H, "internvl_test_layer"));
    taps.shuffled = dev(static_cast<size_t>(m) * 4 * H, "internvl_test_shuffled");
    ASSERT_EQ(cudaMemcpy(d_patches, patches.data(), patches.size() * sizeof(half), cudaMemcpyHostToDevice),
              cudaSuccess);

    InternVLEncoder enc;
    ASSERT_TRUE(enc.init(tower));
    EXPECT_EQ(enc.taken_bytes(), InternVLEncoder::demand_bytes(c));
    ASSERT_TRUE(enc.encode(d_patches, d_out, nullptr, &taps));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Bound per stage: 10x the FP16 floor, i.e. the relL2 of HF's own FP32 output rounded to FP16
    // (orchestrator, 2026-10-01: a fixed 1e-3 had no floor behind it; floor ~2e-4 per stage).
    struct Stage {
        const char* name;
        const half* dev;
        size_t elems;
    };
    const Stage stages[] = {{"embeddings", taps.embeddings, static_cast<size_t>(n) * H},
                            {"layer_0", taps.layers[0], static_cast<size_t>(n) * H},
                            {"layer_1", taps.layers[1], static_cast<size_t>(n) * H},
                            {"shuffled", taps.shuffled, static_cast<size_t>(m) * 4 * H},
                            {"projected", d_out, static_cast<size_t>(m) * D}};
    for (const Stage& s : stages) {
        const std::vector<float> ref = S.floats(s.name);
        std::vector<half> ref16(ref.size());
        for (size_t i = 0; i < ref.size(); ++i)
            ref16[i] = __float2half(ref[i]);
        const double got = rel_l2(read(s.dev, s.elems), ref), floor16 = rel_l2(ref16, ref);
        std::printf("[internvl-encoder] %-10s relL2 vs HF FP32 %.3e  fp16 floor %.3e  ratio %.2f\n", s.name,
                    got, floor16, got / floor16);
        EXPECT_LE(got, 10.0 * floor16) << s.name;
    }

    enc.free_buffers();
    for (half* d : {d_patches, d_out, taps.embeddings, taps.shuffled})
        alloc.free(d);
    for (half* d : taps.layers)
        alloc.free(d);
    qwen3vl_release_vision_tower(tower);
}

}  // namespace
}  // namespace imp
