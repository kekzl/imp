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
#include "model/model.h"
#include "model/safetensors_loader.h"
#include "vision/qwen3vl_pipeline.h"
#include "vision/qwen3vl_vision_load.h"
#include "vision/image_processor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

// The real InternVL3.5-2B tower (24 layers) on test_cat_pil.png (test_cat.jpg decoded by Pillow, lossless;
// a JPEG input carries the stb vs libjpeg-turbo decode gap, #2381) through the pipeline imp serves with
// (stb decode, Pillow-equivalent 448 resize, encoder, projector) vs HF FP32 on CPU
// (tools/internvl_fixture/run_real_ref.sh). Bound relL2 <= 1e-2 (orchestrator, 2026-10-01).
TEST(InternVLEncoder, RealTowerCatMatchesHfFp32) {
    const char* dir = std::getenv("IMP_TEST_MODEL_INTERNVL");
    const char* cat = std::getenv("IMP_TEST_IMAGE_CAT");
    if (!dir || !cat)
        GTEST_SKIP() << "IMP_TEST_MODEL_INTERNVL / IMP_TEST_IMAGE_CAT not set";
    // Reference files tests/fixtures/internvl/<stem>_*, default test_cat_pil (run_real_ref.sh names them by
    // stem).
    const std::string ref_stem = std::getenv("IMP_TEST_INTERNVL_REF_STEM")
                                     ? std::getenv("IMP_TEST_INTERNVL_REF_STEM")
                                     : "test_cat_pil";
    auto model = load_safetensors(dir, /*load_mtp_head=*/false);
    ASSERT_NE(model, nullptr);
    ASSERT_NE(model->vision_tower, nullptr);
    VisionModel& tower = *model->vision_tower;
    ASSERT_TRUE(tower.config.is_internvl);
    const size_t bytes = qwen3vl_vision_arena_bytes(tower, 0);
    ScopedEngineArena arena(bytes + bytes / 8);
    ASSERT_TRUE(arena.opened());
    VRAMAllocator alloc;
    ASSERT_TRUE(alloc.init(0.10f));
    Qwen3VLPipeline pipeline;
    ASSERT_TRUE(pipeline.init(tower, Qwen3VLPipeline::patch_budget(tower, 0)));

    std::ifstream f(cat, std::ios::binary);
    const std::vector<uint8_t> jpg((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    QwenPatches patches;
    ASSERT_TRUE(pipeline.preprocess(jpg, patches));
    const int tokens = pipeline.merged_tokens_of(patches), dim = pipeline.embedding_dim();
    ASSERT_EQ(tokens, 256);
    half* d_out = static_cast<half*>(
        alloc.allocate(static_cast<size_t>(tokens) * dim * sizeof(half), "internvl_cat"));
    ASSERT_NE(d_out, nullptr);
    Qwen3VLImage shape;
    ASSERT_TRUE(pipeline.encode_patches_to(patches, d_out, {}, shape, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto got = read(d_out, static_cast<size_t>(tokens) * dim);
    alloc.free(d_out);

    std::ifstream r(kDir + "/" + ref_stem + "_projector_fp32.f16", std::ios::binary);
    std::vector<uint16_t> raw(static_cast<size_t>(tokens) * dim);
    r.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * 2));
    ASSERT_TRUE(r.good()) << "missing fixture " << kDir;
    std::vector<float> ref(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        half h;
        std::memcpy(&h, &raw[i], sizeof(h));
        ref[i] = __half2float(h);
    }
    const double rl = rel_l2(got, ref);
    std::printf("[internvl-real] test_cat tokens=%d dim=%d relL2_vs_hf_fp32=%.4e (imp decode + resize)\n",
                tokens, dim, rl);

    // Split the gap: imp's 448 tile vs HF's (decode + resize), and the encoder fed HF's own tile.
    std::ifstream tf(kDir + "/" + ref_stem + "_pixels_448.u8", std::ios::binary);
    const std::vector<uint8_t> hf_tile((std::istreambuf_iterator<char>(tf)),
                                       std::istreambuf_iterator<char>());
    ASSERT_EQ(hf_tile.size(), 3u * 448 * 448);
    std::vector<uint8_t> rgb, tile;
    int w = 0, h = 0;
    ASSERT_TRUE(decode_rgb(jpg, rgb, w, h));
    InternVLPreprocessConfig pc;
    std::vector<half> unused;
    ASSERT_TRUE(internvl_preprocess(rgb.data(), w, h, pc, unused, &tile));
    int max_step = 0;
    double sum_step = 0;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 448; ++y)
            for (int x = 0; x < 448; ++x) {
                const int d = std::abs(int(tile[(static_cast<size_t>(y) * 448 + x) * 3 + c]) -
                                       int(hf_tile[(static_cast<size_t>(c) * 448 + y) * 448 + x]));
                max_step = std::max(max_step, d);
                sum_step += d;
            }
    std::printf("[internvl-real] tile imp vs hf: max %d/255, mean %.3f/255\n", max_step,
                sum_step / (3.0 * 448 * 448));

    QwenPatches hf_patches = patches;
    for (int pi = 0; pi < 1024; ++pi)
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < 14; ++y)
                for (int x = 0; x < 14; ++x) {
                    const int py = (pi / 32) * 14 + y, px = (pi % 32) * 14 + x;
                    const float v = (hf_tile[(static_cast<size_t>(c) * 448 + py) * 448 + px] / 255.0f -
                                     pc.mean[c]) /
                                    pc.std[c];
                    hf_patches.data[static_cast<size_t>(pi) * 588 + (c * 14 + y) * 14 + x] = __float2half(v);
                }
    half* d_out2 = static_cast<half*>(
        alloc.allocate(static_cast<size_t>(tokens) * dim * sizeof(half), "internvl_cat2"));
    ASSERT_TRUE(pipeline.encode_patches_to(hf_patches, d_out2, {}, shape, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const double rl_hf_tile = rel_l2(read(d_out2, static_cast<size_t>(tokens) * dim), ref);
    alloc.free(d_out2);
    std::printf("[internvl-real] test_cat relL2_vs_hf_fp32=%.4e (HF tile, imp encoder)\n", rl_hf_tile);
    EXPECT_LE(rl, 1e-2);
}

}  // namespace
}  // namespace imp
