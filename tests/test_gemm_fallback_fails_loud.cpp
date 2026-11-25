// gemm_dispatch_uncached_fallback: an exit that cannot write the output throws (B-108). Before, the
// four exits below logged and returned, and the forward pass read whatever the buffer held.
// CPU lane: every case leaves before the first kernel launch.

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/logging.h"
#include "core/tensor.h"
#include "exec/gemm_dispatch_fallback.h"
#include "exec/quant_scratch.h"
#include "exec/weight_caches.h"

namespace {

using imp::GemmContext;
using imp::QType;
using imp::Tensor;

constexpr int64_t kN = 8, kK = 32;

// F32 rows: no in-place dp4a / IMMA route and no generic dequant route takes them.
struct Case {
    float x[kK] = {};
    float y[kN] = {};
    uint8_t w[64] = {};
    imp::WeightCaches wc;
    imp::QuantScratch qs;  // dequant == nullptr: no whole-weight dequant fits
    Tensor in, out, weight;
    GemmContext ctx;

    explicit Case(QType wq, float beta = 0.0f) {
        const int64_t xs[2] = {1, kK}, ys[2] = {1, kN}, ws[2] = {kN, kK};
        in = Tensor(x, QType::F32, 2, xs, false);
        out = Tensor(y, QType::F32, 2, ys, false);
        weight = Tensor(w, wq, 2, ws, false);
        ctx.beta = beta;
        ctx.wcache = &wc;
        ctx.qscratch = &qs;
    }

    std::string thrown() {
        try {
            imp::gemm_dispatch_uncached_fallback(in, weight, out, ctx);
        } catch (const std::runtime_error& e) {
            return e.what();
        }
        return {};
    }
};

bool has(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

}  // namespace

TEST(GemmFallbackFailsLoud, BlockQuantWeightWithoutDequantScratchThrows) {
    Case c(QType::Q6_K);
    const std::string what = c.thrown();
    EXPECT_TRUE(has(what, "no route for a 8 x 32 Q6_K weight at M=1")) << what;
}

TEST(GemmFallbackFailsLoud, ResidualAddWithoutFp16PathThrows) {
    Case c(QType::Q6_K, 1.0f);
    const std::string what = c.thrown();
    EXPECT_TRUE(has(what, "beta=1.000 but no FP16 path for a 8 x 32 Q6_K weight at M=1")) << what;
}

TEST(GemmFallbackFailsLoud, DroppedSourceAtTheFinalFallbackThrows) {
    Case c(QType::Q6_K);
    c.weight.dropped_source = true;
    c.weight.kind = imp::TensorKind::UNKNOWN;
    const std::string what = c.thrown();
    EXPECT_TRUE(has(what, "dropped")) << what;
    EXPECT_TRUE(has(what, "8 x 32 Q6_K")) << what;
}

TEST(GemmFallbackFailsLoud, MissingCacheOrScratchThrows) {
    Case no_cache(QType::Q6_K);
    no_cache.ctx.wcache = nullptr;
    EXPECT_TRUE(has(no_cache.thrown(), "no weight cache")) << no_cache.thrown();
    Case no_scratch(QType::Q6_K);
    no_scratch.ctx.qscratch = nullptr;
    EXPECT_TRUE(has(no_scratch.thrown(), "no quant scratch")) << no_scratch.thrown();
}

TEST(GemmFallbackFailsLoud, ThrowHelperFormatsAndTruncates) {
    try {
        imp::throw_runtime_error("M=%d %s", 7, std::string(600, 'x').c_str());
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_EQ(what.rfind("M=7 xxx", 0), 0u);
        EXPECT_EQ(what.size(), 383u);
        return;
    }
    FAIL() << "throw_runtime_error returned";
}
