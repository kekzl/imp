// InternVL vision tower, CPU half: config parse, checkpoint-name loader, and a double-precision
// reference forward checked per stage against the pinned HF FP32 modules
// (tests/fixtures/internvl, tools/internvl_fixture/run_encoder.sh). The GPU encoder is checked
// against the same HF stages in test_internvl_encoder.cu.

#include "internvl_tiny_config.h"
#include "safetensors_fixture.h"
#include "vision/internvl_vision.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace imp {
namespace {

const std::string kDir = std::string(IMP_TEST_FIXTURES_DIR) + "/internvl";

JValue parse(const char* s) {
    JsonParser p{std::string_view(s)};
    return p.parse();
}

using Mat = std::vector<double>;  // row-major

Mat to_d(const std::vector<float>& v) { return Mat(v.begin(), v.end()); }

double rel_l2(const Mat& a, const std::vector<float>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += (a[i] - ref[i]) * (a[i] - ref[i]);
        den += static_cast<double>(ref[i]) * ref[i];
    }
    return std::sqrt(num / den);
}

// y[r, o] = sum_i x[r, i] * w[o, i] + b[o]
Mat linear(const Mat& x, int rows, int in, const Mat& w, const Mat& b, int out) {
    Mat y(static_cast<size_t>(rows) * out);
    for (int r = 0; r < rows; ++r)
        for (int o = 0; o < out; ++o) {
            double s = b.empty() ? 0.0 : b[o];
            for (int i = 0; i < in; ++i)
                s += x[static_cast<size_t>(r) * in + i] * w[static_cast<size_t>(o) * in + i];
            y[static_cast<size_t>(r) * out + o] = s;
        }
    return y;
}

Mat layernorm(const Mat& x, int rows, int dim, const Mat& g, const Mat& b, double eps) {
    Mat y(x.size());
    for (int r = 0; r < rows; ++r) {
        double mean = 0, var = 0;
        for (int i = 0; i < dim; ++i)
            mean += x[static_cast<size_t>(r) * dim + i];
        mean /= dim;
        for (int i = 0; i < dim; ++i) {
            const double d = x[static_cast<size_t>(r) * dim + i] - mean;
            var += d * d;
        }
        var /= dim;
        for (int i = 0; i < dim; ++i)
            y[static_cast<size_t>(r) * dim + i] = (x[static_cast<size_t>(r) * dim + i] - mean) /
                                                      std::sqrt(var + eps) * g[i] +
                                                  b[i];
    }
    return y;
}

void gelu(Mat& x) {
    for (double& v : x)
        v = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
}

struct Stages {
    Mat embeddings;
    std::vector<Mat> layers;
    Mat shuffled, projected;
};

// The reference, written from modeling_internvl.py (transformers 5.17.0), not from the encoder.
Stages reference(const imp_test::SafetensorsFixture& W, const std::vector<float>& pixels, int H, int heads,
                 int inter, int depth, int side, int P, int D) {
    auto w = [&](const std::string& n) { return to_d(W.floats(n)); };
    const int np = side * side, n = np + 1, hd = H / heads, img = side * P;
    Stages s;
    // Patch embedding: Conv2d(kernel == stride == P) in (channel, row, col) weight order.
    const Mat cw = w("vision_tower.embeddings.patch_embeddings.projection.weight");
    const Mat cb = w("vision_tower.embeddings.patch_embeddings.projection.bias");
    const Mat cls = w("vision_tower.embeddings.cls_token"),
              pos = w("vision_tower.embeddings.position_embeddings");
    Mat h(static_cast<size_t>(n) * H);
    for (int i = 0; i < H; ++i)
        h[i] = cls[i] + pos[i];
    for (int p = 0; p < np; ++p)
        for (int o = 0; o < H; ++o) {
            double acc = cb[o];
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < P; ++y)
                    for (int x = 0; x < P; ++x) {
                        const int py = (p / side) * P + y, px = (p % side) * P + x;
                        acc += cw[((static_cast<size_t>(o) * 3 + c) * P + y) * P + x] *
                               pixels[(static_cast<size_t>(c) * img + py) * img + px];
                    }
            h[static_cast<size_t>(p + 1) * H + o] = acc + pos[static_cast<size_t>(p + 1) * H + o];
        }
    s.embeddings = h;
    for (int l = 0; l < depth; ++l) {
        const std::string pre = "vision_tower.encoder.layer." + std::to_string(l) + ".";
        const Mat x1 = layernorm(h, n, H, w(pre + "layernorm_before.weight"),
                                 w(pre + "layernorm_before.bias"), 1e-6);
        const Mat q = linear(x1, n, H, w(pre + "attention.q_proj.weight"), w(pre + "attention.q_proj.bias"),
                             H);
        const Mat k = linear(x1, n, H, w(pre + "attention.k_proj.weight"), w(pre + "attention.k_proj.bias"),
                             H);
        const Mat v = linear(x1, n, H, w(pre + "attention.v_proj.weight"), w(pre + "attention.v_proj.bias"),
                             H);
        Mat att(static_cast<size_t>(n) * H, 0.0);
        for (int hh = 0; hh < heads; ++hh)
            for (int i = 0; i < n; ++i) {
                std::vector<double> sc(static_cast<size_t>(n));
                double mx = -1e300;
                for (int j = 0; j < n; ++j) {
                    double d = 0;
                    for (int e = 0; e < hd; ++e)
                        d += q[static_cast<size_t>(i) * H + hh * hd + e] *
                             k[static_cast<size_t>(j) * H + hh * hd + e];
                    sc[j] = d / std::sqrt(static_cast<double>(hd));
                    mx = std::max(mx, sc[j]);
                }
                double sum = 0;
                for (double& e : sc)
                    sum += (e = std::exp(e - mx));
                for (int j = 0; j < n; ++j)
                    for (int e = 0; e < hd; ++e)
                        att[static_cast<size_t>(i) * H + hh * hd + e] +=
                            sc[j] / sum * v[static_cast<size_t>(j) * H + hh * hd + e];
            }
        const Mat o = linear(att, n, H, w(pre + "attention.projection_layer.weight"),
                             w(pre + "attention.projection_layer.bias"), H);
        const Mat l1 = w(pre + "lambda_1"), l2 = w(pre + "lambda_2");
        for (size_t i = 0; i < h.size(); ++i)
            h[i] += l1[i % H] * o[i];
        const Mat x2 = layernorm(h, n, H, w(pre + "layernorm_after.weight"), w(pre + "layernorm_after.bias"),
                                 1e-6);
        Mat f = linear(x2, n, H, w(pre + "mlp.fc1.weight"), w(pre + "mlp.fc1.bias"), inter);
        gelu(f);
        const Mat f2 = linear(f, n, inter, w(pre + "mlp.fc2.weight"), w(pre + "mlp.fc2.bias"), H);
        for (size_t i = 0; i < h.size(); ++i)
            h[i] += l2[i % H] * f2[i];
        s.layers.push_back(h);
    }
    // Drop CLS, 2x2 shuffle (InternVLModel.pixel_shuffle), projector.
    const int hs = side / 2, W4 = 4 * H, m = hs * hs;
    s.shuffled.assign(static_cast<size_t>(m) * W4, 0.0);
    for (int i = 0; i < hs; ++i)
        for (int j = 0; j < hs; ++j)
            for (int qd = 0; qd < 4; ++qd)
                for (int c = 0; c < H; ++c) {
                    const int r = 2 * i + qd / 2, col = 2 * j + qd % 2;
                    s.shuffled[(static_cast<size_t>(i) * hs + j) * W4 + qd * H + c] =
                        h[(1 + static_cast<size_t>(r) * side + col) * H + c];
                }
    Mat y = layernorm(s.shuffled, m, W4, w("multi_modal_projector.layer_norm.weight"),
                      w("multi_modal_projector.layer_norm.bias"), 1e-5);
    y = linear(y, m, W4, w("multi_modal_projector.linear_1.weight"), w("multi_modal_projector.linear_1.bias"),
               D);
    gelu(y);
    s.projected = linear(y, m, D, w("multi_modal_projector.linear_2.weight"),
                         w("multi_modal_projector.linear_2.bias"), D);
    return s;
}

TEST(InternVLVision, ParsesTheConfig) {
    const auto c = parse_internvl_vision_config(parse(imp_test::kInternVLTinyConfig));
    ASSERT_TRUE(c) << c.error();
    EXPECT_TRUE(c->is_internvl);
    EXPECT_EQ(c->hidden_size, 64);
    EXPECT_EQ(c->head_dim, 16);
    EXPECT_EQ(c->num_patches, 16);
    EXPECT_EQ(c->num_image_tokens, 4);
    EXPECT_EQ(c->pos_embed_grid, 4);
    EXPECT_EQ(c->out_hidden_size, 32);
    EXPECT_FLOAT_EQ(c->image_mean[0], 0.485f);
    EXPECT_FLOAT_EQ(c->image_std[2], 0.225f);
}

TEST(InternVLVision, RefusesWhatTheEncoderDoesNotImplement) {
    std::string s = imp_test::kInternVLTinyConfig;
    std::string qk = s;
    qk.replace(qk.find("\"use_qk_norm\": false"), 20, "\"use_qk_norm\": true ");
    EXPECT_FALSE(parse_internvl_vision_config(parse(qk.c_str())));
    std::string ds = s;
    ds.replace(ds.find("\"downsample_ratio\": 0.5"), 23, "\"downsample_ratio\": 0.25");
    EXPECT_FALSE(parse_internvl_vision_config(parse(ds.c_str())));
    std::string rms = s;
    rms.replace(rms.find("\"layer_norm\""), 12, "\"rms_norm\"");
    EXPECT_FALSE(parse_internvl_vision_config(parse(rms.c_str())));
}

TEST(InternVLVision, LoaderReadsEveryCheckpointTensor) {
    imp_test::SafetensorsFixture W;
    ASSERT_TRUE(W.load(kDir + "/tiny_tower.st"));
    VisionModel m;
    m.config = *parse_internvl_vision_config(parse(imp_test::kInternVLTinyConfig));
    const auto used = load_internvl_vision_tensors(W.tensors, m);
    ASSERT_TRUE(used) << used.error();
    EXPECT_EQ(*used, static_cast<int>(W.tensors.size()));
    EXPECT_EQ(m.layers[0].wq.shape[0], 3 * 64) << "q|k|v fused";
    int slots = 0;
    internvl_visit_vision_tensors(m, [&](Tensor& t, const std::string& what) {
        EXPECT_NE(t.data, nullptr) << what;
        ++slots;
    });
    EXPECT_EQ(slots, 4 + 2 * 14 + 6);

    W.tensors["vision_tower.encoder.layer.0.attention.q_norm.weight"] =
        W.tensors["vision_tower.embeddings.cls_token"];
    VisionModel m2;
    m2.config = m.config;
    EXPECT_FALSE(load_internvl_vision_tensors(W.tensors, m2)) << "an unknown vision tensor must refuse";
}

TEST(InternVLVision, DoubleReferenceMatchesHfPerStage) {
    imp_test::SafetensorsFixture W, S;
    ASSERT_TRUE(W.load(kDir + "/tiny_tower.st"));
    ASSERT_TRUE(S.load(kDir + "/tiny_stages.st"));
    const Stages r = reference(W, S.floats("pixel_values"), 64, 4, 128, 2, 4, 14, 32);
    const double e = rel_l2(r.embeddings, S.floats("embeddings"));
    const double l0 = rel_l2(r.layers[0], S.floats("layer_0"));
    const double l1 = rel_l2(r.layers[1], S.floats("layer_1"));
    const double sh = rel_l2(r.shuffled, S.floats("shuffled"));
    const double pr = rel_l2(r.projected, S.floats("projected"));
    std::printf(
        "[internvl-ref] relL2 vs HF FP32: embeddings %.3e layer_0 %.3e layer_1 %.3e shuffled %.3e "
        "projected %.3e\n",
        e, l0, l1, sh, pr);
    for (double v : {e, l0, l1, sh, pr})
        EXPECT_LE(v, 1e-5);
}

}  // namespace
}  // namespace imp
