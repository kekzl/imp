#include "model/gguf_loader.h"
#include "model/tensor_kind_matcher.h"
#include "core/tensor_kind.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include <unordered_set>

#include "test_models.h"

using namespace imp;

namespace {

const char* kTestModelPath = std::getenv(imp_test::kEnvGguf);

}  // namespace

TEST(TensorKindCoverage, NoUnknownKindsInSmallQwen) {
    if (!kTestModelPath) {
        GTEST_SKIP() << "Set IMP_TEST_GGUF=/path/to/model.gguf to run this test";
    }
    if (!std::filesystem::exists(kTestModelPath)) {
        GTEST_SKIP() << "Model not found: " << kTestModelPath;
    }

    auto model = load_gguf(kTestModelPath);
    ASSERT_NE(model, nullptr) << "Failed to load model: " << kTestModelPath;

    std::unordered_set<std::string> unknown_names;
    auto check = [&](const Tensor& t, const char* debug_name) {
        if (t.data == nullptr)
            return;
        if (t.kind == TensorKind::UNKNOWN) {
            unknown_names.insert(debug_name);
        }
    };

    for (int i = 0; i < model->n_layers(); ++i) {
        const auto& L = model->layer(i);
        check(L.wq, "wq");
        check(L.wk, "wk");
        check(L.wv, "wv");
        check(L.wo, "wo");
        check(L.w_gate, "w_gate");
        check(L.w_up, "w_up");
        check(L.w_down, "w_down");
        check(L.attn_norm, "attn_norm");
        check(L.ffn_norm, "ffn_norm");
    }

    if (!unknown_names.empty()) {
        std::string msg = "Tensors with UNKNOWN kind:";
        for (const auto& n : unknown_names) {
            msg += " " + n;
        }
        FAIL() << msg;
    }
}
