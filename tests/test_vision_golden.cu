// =============================================================================
// Vision GPU encoder + projector frozen golden (R9 / issue #583).
//
// src/vision/ was the only fully GPU-blind area of the suite: since #564 only
// the CPU preprocessing (test_vision_preprocess.cpp) was covered; the SigLIP
// encoder, the gemma4v path (RMSNorm / per-head q/k/v norm / 2D axial RoPE /
// sandwich post-norms / GeGLU) and the fused-FP32 projector tail (#489) ran
// only in manual VL runs.
//
// This drives the full GPU pipeline directly — committed test image →
// preprocess → SigLIP/gemma4v encoder → projector tail → image embeddings —
// without an LM: load_vision_gguf() reads everything (incl. lm_d_model) from
// the mmproj GGUF, and VisionEncoder needs only the VisionModel + a
// VRAMAllocator. It compares projector-output spot values against a frozen
// stability golden (tests/refs/vision_encoder_golden.h) at the f16 class
// tolerance, and hard-guards against NaN/Inf over the whole embedding.
//
// The golden is a REGRESSION LOCK (output stability of a manually-validated
// build), not an independent oracle — there is no fp64 reference for the
// encoder. See the golden header + tests/refs/README.md.
//
// Skips cleanly when the model/image are absent (CI has no GPU/models):
//   IMP_TEST_MMPROJ          gemma-3 SigLIP mmproj GGUF
//   IMP_TEST_MMPROJ_GEMMA4   gemma4v mmproj GGUF (optional)
//   IMP_VISION_TEST_IMAGE    override the committed fixture path
//   IMP_VISION_GOLDEN_DUMP=1 print golden arrays for regeneration, assert only
//                            the NaN/Inf guard.
// =============================================================================

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "test_models.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "memory/vram_allocator.h"
#include "vision/image_processor.h"
#include "vision/vision_encoder.h"
#include "vision/vision_loader.h"
#include "vision/vision_model.h"

#include "refs/vision_encoder_golden.h"

namespace imp {
namespace {

using imp_refs::gemma3_golden;
using imp_refs::gemma4v_golden;
using imp_refs::VisionGoldenArch;

const char* fixture_image() {
    if (const char* p = std::getenv("IMP_VISION_TEST_IMAGE"))
        return p;
    // Relative to the repo root (where ctest runs). The committed fixture is a
    // 64x64 deterministic synthetic PNG (stb decodes PNG bit-exactly).
    return "tests/fixtures/vision_test_64.png";
}

bool dump_mode() {
    const char* v = std::getenv("IMP_VISION_GOLDEN_DUMP");
    // Treat only a non-empty value as "on" so `make test-vision` (which passes
    // an empty IMP_VISION_GOLDEN_DUMP= by default) asserts instead of dumping.
    return v != nullptr && v[0] != '\0';
}

bool file_exists(const std::string& p) {
    if (p.empty())
        return false;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f)
        return false;
    std::fclose(f);
    return true;
}

// Run the full pipeline for one mmproj GGUF and return the projector-output
// embeddings on the host ([num_tokens * d_model] row-major). Returns false on
// any setup failure (caller turns that into a hard test failure, not a skip —
// skips are decided before we get here, on file presence).
bool run_vision_pipeline(const std::string& mmproj_path, const std::string& image_path,
                         std::vector<float>& out, int& num_tokens, int& d_model, std::string& err) {
    std::unique_ptr<VisionModel> model = load_vision_gguf(mmproj_path);
    if (!model) {
        err = "load_vision_gguf failed";
        return false;
    }
    num_tokens = model->config.num_image_tokens;
    d_model = model->lm_d_model > 0 ? model->lm_d_model : 0;
    if (d_model <= 0) {
        err = "mm_proj output dim (lm_d_model) not resolved";
        return false;
    }

    VRAMAllocator alloc;
    if (!alloc.init()) {
        err = "VRAMAllocator init failed (no GPU?)";
        return false;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        err = "cudaStreamCreate failed";
        return false;
    }

    VisionEncoder encoder;
    if (!encoder.init(*model, d_model, stream, &alloc)) {
        err = "VisionEncoder init failed";
        cudaStreamDestroy(stream);
        return false;
    }

    // Preprocess exactly as VisionPipeline::set_image does: resize to the
    // model's square image_size, apply the model's mean/std, FP16 CHW.
    ImageData img;
    if (!load_and_preprocess_image(image_path, model->config.image_size, model->config.image_mean,
                                   model->config.image_std, img)) {
        err = "preprocess failed for " + image_path;
        cudaStreamDestroy(stream);
        return false;
    }

    const int n_pixels = 3 * img.width * img.height;
    half* d_pixels = static_cast<half*>(
        alloc.allocate(static_cast<size_t>(n_pixels) * sizeof(half), "vision_golden_pixels"));
    half* d_out = static_cast<half*>(
        alloc.allocate(static_cast<size_t>(num_tokens) * d_model * sizeof(half), "vision_golden_out"));
    if (!d_pixels || !d_out) {
        err = "device buffer alloc failed";
        if (d_pixels)
            alloc.free(d_pixels);
        if (d_out)
            alloc.free(d_out);
        cudaStreamDestroy(stream);
        return false;
    }
    // VRAMAllocator is a tracker, not an owner (its dtor is a no-op) — free both
    // buffers on every exit path below.
    struct BufGuard {
        VRAMAllocator& a;
        half* p;
        half* o;
        ~BufGuard() {
            a.free(p);
            a.free(o);
        }
    } guard{alloc, d_pixels, d_out};

    cudaMemcpyAsync(d_pixels, img.pixels.data(), static_cast<size_t>(n_pixels) * sizeof(half),
                    cudaMemcpyHostToDevice, stream);

    if (!encoder.encode(d_pixels, d_out, stream)) {
        err = "encode failed";
        cudaStreamDestroy(stream);
        return false;
    }
    cudaStreamSynchronize(stream);

    std::vector<half> h_out(static_cast<size_t>(num_tokens) * d_model);
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaStreamDestroy(stream);

    out.resize(h_out.size());
    for (size_t i = 0; i < h_out.size(); i++)
        out[i] = __half2float(h_out[i]);
    return true;
}

// ~64 strided flat indices spanning the whole [tokens, d_model] tensor,
// guaranteed to include the first and last elements.
std::vector<int> make_spot_indices(int total, int n) {
    std::vector<int> idx;
    if (total <= 0)
        return idx;
    n = std::min(n, total);
    for (int k = 0; k < n; k++) {
        long v = static_cast<long>(k) * (total - 1) / std::max(1, n - 1);
        idx.push_back(static_cast<int>(v));
    }
    return idx;
}

void emit_golden(const char* arch, const std::vector<float>& emb, int num_tokens, int d_model) {
    const int total = num_tokens * d_model;
    auto spot = make_spot_indices(total, 64);

    // per-token L2 for a few evenly spaced tokens
    std::vector<int> tok_idx;
    const int n_tok = std::min(8, num_tokens);
    for (int k = 0; k < n_tok; k++)
        tok_idx.push_back(static_cast<int>(static_cast<long>(k) * (num_tokens - 1) / std::max(1, n_tok - 1)));

    double sum = 0.0;
    for (float v : emb)
        sum += v;
    const double mean = sum / total;

    std::printf("\n// ---- PASTE BELOW for arch \"%s\" (num_tokens=%d d_model=%d) ----\n", arch, num_tokens,
                d_model);
    std::printf("inline constexpr int %s_spot_idx[%zu] = {", arch, spot.size());
    for (size_t i = 0; i < spot.size(); i++)
        std::printf("%s%d", i ? "," : "", spot[i]);
    std::printf("};\n");
    std::printf("inline constexpr float %s_spot_val[%zu] = {", arch, spot.size());
    for (size_t i = 0; i < spot.size(); i++)
        std::printf("%s%.7gf", i ? "," : "", emb[spot[i]]);
    std::printf("};\n");
    std::printf("inline constexpr int %s_tok_l2_idx[%zu] = {", arch, tok_idx.size());
    for (size_t i = 0; i < tok_idx.size(); i++)
        std::printf("%s%d", i ? "," : "", tok_idx[i]);
    std::printf("};\n");
    std::printf("inline constexpr float %s_tok_l2[%zu] = {", arch, tok_idx.size());
    for (size_t i = 0; i < tok_idx.size(); i++) {
        double l2 = 0.0;
        const float* row = &emb[static_cast<size_t>(tok_idx[i]) * d_model];
        for (int j = 0; j < d_model; j++)
            l2 += static_cast<double>(row[j]) * row[j];
        std::printf("%s%.7gf", i ? "," : "", std::sqrt(l2));
    }
    std::printf("};\n");
    std::printf("inline constexpr float %s_global_mean = %.7gf;\n", arch, mean);
    std::printf("// ---- END %s ----\n\n", arch);
}

float emb_mean(const std::vector<float>& emb) {
    double s = 0.0;
    for (float v : emb)
        s += v;
    return static_cast<float>(s / std::max<size_t>(1, emb.size()));
}

void guard_finite(const std::vector<float>& emb, const char* arch) {
    size_t bad = 0;
    for (float v : emb)
        if (!std::isfinite(v))
            bad++;
    EXPECT_EQ(bad, 0u) << arch << ": " << bad << " non-finite (NaN/Inf) embedding values";
}

// Assert the measured embeddings match the frozen golden for this arch.
void assert_golden(const VisionGoldenArch& g, const std::vector<float>& emb, int num_tokens, int d_model) {
    ASSERT_EQ(num_tokens, g.num_tokens) << g.name << ": token count drifted";
    ASSERT_EQ(d_model, g.d_model) << g.name << ": d_model drifted";

    // f16-class stability tolerance: 1e-2 rel with a small abs floor (the
    // embeddings span ~O(1..10); a fixed abs floor keeps near-zero spots from
    // demanding impossible relative precision). See tests/refs/README.md.
    const float rel_tol = 1e-2f;
    const float abs_floor = 5e-3f;
    for (int i = 0; i < g.n_spot; i++) {
        const float got = emb[g.spot_idx[i]];
        const float want = g.spot_val[i];
        const float tol = abs_floor + rel_tol * std::fabs(want);
        EXPECT_NEAR(got, want, tol) << g.name << ": spot[" << i << "] flat_idx=" << g.spot_idx[i];
    }
    for (int i = 0; i < g.n_tok_l2; i++) {
        const float* row = &emb[static_cast<size_t>(g.tok_l2_idx[i]) * d_model];
        double l2 = 0.0;
        for (int j = 0; j < d_model; j++)
            l2 += static_cast<double>(row[j]) * row[j];
        const float got = static_cast<float>(std::sqrt(l2));
        const float tol = abs_floor + rel_tol * std::fabs(g.tok_l2[i]);
        EXPECT_NEAR(got, g.tok_l2[i], tol) << g.name << ": token " << g.tok_l2_idx[i] << " L2";
    }
    const float mean_tol = abs_floor + rel_tol * std::fabs(g.global_mean);
    EXPECT_NEAR(emb_mean(emb), g.global_mean, mean_tol) << g.name << ": global mean";
}

void run_arch(const char* env_var, const char* arch, const VisionGoldenArch* golden) {
    const char* mmproj = std::getenv(env_var);
    if (!mmproj || !file_exists(mmproj))
        GTEST_SKIP() << "Set " << env_var << " to an mmproj GGUF to run (" << arch << ")";
    const std::string image = fixture_image();
    if (!file_exists(image))
        GTEST_SKIP() << "Test image not found: " << image
                     << " (set IMP_VISION_TEST_IMAGE or run from repo root)";

    std::vector<float> emb;
    int num_tokens = 0, d_model = 0;
    std::string err;
    ASSERT_TRUE(run_vision_pipeline(mmproj, image, emb, num_tokens, d_model, err)) << err;
    ASSERT_FALSE(emb.empty());

    // The real Gemma-class assert: no NaN/Inf anywhere (catches the #514/#489
    // overflow-class failures regardless of golden state).
    guard_finite(emb, arch);

    if (dump_mode()) {
        emit_golden(arch, emb, num_tokens, d_model);
        GTEST_SKIP() << arch
                     << ": dump mode — golden arrays printed, paste into "
                        "tests/refs/vision_encoder_golden.h";
    }

    if (!golden) {
        GTEST_SKIP() << arch
                     << ": no frozen golden committed yet — run with "
                        "IMP_VISION_GOLDEN_DUMP=1 to generate";
    }
    assert_golden(*golden, emb, num_tokens, d_model);
}

TEST(VisionGolden, Gemma3SigLIP) { run_arch(imp_test::kEnvMmproj, "gemma3", gemma3_golden()); }

TEST(VisionGolden, Gemma4v) { run_arch(imp_test::kEnvMmprojGemma4, "gemma4v", gemma4v_golden()); }

}  // namespace
}  // namespace imp
