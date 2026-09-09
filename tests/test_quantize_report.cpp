// imp-quantize's export report: the per-tensor error, the summary line, the
// JSON an operator diffs between two exports, and the provenance banner.
//
// The numbers this file checks are the ones the tool had already computed and
// discarded (awq.cu's err_rtn / err_best) plus the one it never computed at
// all: what a round-to-nearest export actually cost per tensor. A report that
// silently reads zero everywhere would look exactly like a perfect export, so
// the decode is checked against hand-built format bytes rather than against
// imp's own dequant kernel.

#include "../tools/imp-quantize/quant_report.h"

#include "core/fp_bits.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace imp;
using namespace imp::quantize;

namespace {

// FP8 E4M3 byte for 1.0: sign 0, exp 0111 (bias 7), mantissa 000.
constexpr unsigned char kFp8One = 0x38;

// The eight E2M1 magnitudes, in nibble order.
constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// One row of 16 values, packed the way the writer packs them: low nibble is
// the even index, high nibble the odd one.
std::vector<unsigned char> pack_row(const std::vector<uint8_t>& nibbles) {
    std::vector<unsigned char> out(nibbles.size() / 2, 0);
    for (size_t i = 0; i < nibbles.size(); i++)
        out[i / 2] |= static_cast<unsigned char>((nibbles[i] & 0x0F) << (4 * (i % 2)));
    return out;
}

}  // namespace

// Values the grid can represent exactly must report exactly zero error. If this
// ever reads non-zero the decode disagrees with the writer and every number in
// the report is off by that disagreement.
TEST(QuantReport, ExactValuesReportZeroError) {
    std::vector<uint8_t> nibbles(16);
    std::vector<uint16_t> src(16);
    for (int i = 0; i < 16; i++) {
        const int mag = i % 8;
        const bool neg = i >= 8;
        nibbles[static_cast<size_t>(i)] = static_cast<uint8_t>(mag | (neg ? 0x08 : 0));
        src[static_cast<size_t>(i)] = float_to_half(neg ? -kE2M1[mag] : kE2M1[mag]);
    }
    const auto packed = pack_row(nibbles);
    const std::vector<unsigned char> micro = {kFp8One};

    const TensorError e = nvfp4_tensor_error("w", src.data(), packed.data(), micro.data(), 1.0f, 1, 16);
    EXPECT_EQ(e.name, "w");
    EXPECT_DOUBLE_EQ(e.max_rel, 0.0);
    EXPECT_DOUBLE_EQ(e.mse, 0.0);
}

// A value between two grid points must show up, and the size must be the
// distance to the grid point the writer chose, not a number of its own.
TEST(QuantReport, OffGridValuesReportTheDistanceToTheGrid) {
    // Source 5.0 stored as the nibble for 4.0: the grid step there is 2.0.
    std::vector<uint8_t> nibbles(16, 0);
    std::vector<uint16_t> src(16, float_to_half(0.0f));
    nibbles[0] = 6;  // 4.0
    src[0] = float_to_half(5.0f);
    const auto packed = pack_row(nibbles);
    const std::vector<unsigned char> micro = {kFp8One};

    const TensorError e = nvfp4_tensor_error("w", src.data(), packed.data(), micro.data(), 1.0f, 1, 16);
    // Normalised by the tensor absmax (5.0), so 1.0 of absolute error is 0.2.
    EXPECT_NEAR(e.max_rel, 0.2, 1e-6);
    EXPECT_NEAR(e.mse, 1.0 / 16.0, 1e-6);

    // The tensor scale multiplies: doubling it doubles the reconstruction and
    // therefore the error. A report blind to it would call a wrong-scale export
    // perfect, which is the #1 failure mode of the two NVFP4 layouts.
    const TensorError scaled = nvfp4_tensor_error("w", src.data(), packed.data(), micro.data(), 2.0f, 1, 16);
    EXPECT_GT(scaled.max_rel, e.max_rel);
}

TEST(QuantReport, WorstTensorsAreOrderedAndTruncated) {
    std::vector<TensorError> rows = {
        {"a", 0.10, 0.001}, {"b", 0.50, 0.02}, {"c", 0.25, 0.004}, {"d", 0.75, 0.03}, {"e", 0.05, 0.0001},
    };
    const auto worst = worst_tensors(rows, 3);
    ASSERT_EQ(worst.size(), 3u);
    EXPECT_EQ(worst[0].name, "d");
    EXPECT_EQ(worst[1].name, "b");
    EXPECT_EQ(worst[2].name, "c");
    // A limit past the end returns everything, not a truncation to zero.
    EXPECT_EQ(worst_tensors(rows, 99).size(), 5u);
    EXPECT_TRUE(worst_tensors({}, 5).empty());
}

TEST(QuantReport, SummaryNamesTheWorstTensorsAndTheirNumbers) {
    const std::vector<TensorError> rows = {{"model.layers.0.mlp.down_proj.weight", 0.42, 0.01},
                                           {"model.layers.0.self_attn.q_proj.weight", 0.11, 0.002}};
    const std::string s = format_error_summary(rows, 5);
    EXPECT_NE(s.find("2 tensor"), std::string::npos) << s;
    EXPECT_NE(s.find("down_proj"), std::string::npos) << s;
    EXPECT_NE(s.find("0.42"), std::string::npos) << s;
    // Worst first, so the line an operator reads first is the one that decides.
    EXPECT_LT(s.find("down_proj"), s.find("q_proj")) << s;
    // Nothing quantized is a statement, not an empty table.
    EXPECT_FALSE(format_error_summary({}, 5).empty());
}

TEST(QuantReport, JsonCarriesTheTensorsAndTheGroupObjective) {
    const std::vector<TensorError> tensors = {{"w0", 0.5, 0.02}, {"w1", 0.25, 0.01}};
    const std::vector<GroupError> groups = {{3, "WQ", 8.0, 2.0}};
    const std::string j = error_report_json(tensors, groups, /*calibrated=*/true);
    EXPECT_EQ(j.front(), '{');
    EXPECT_EQ(j.back(), '\n');
    EXPECT_NE(j.find("\"calibrated\": true"), std::string::npos) << j;
    EXPECT_NE(j.find("\"tensors\""), std::string::npos) << j;
    EXPECT_NE(j.find("\"w0\""), std::string::npos) << j;
    EXPECT_NE(j.find("\"max_rel\""), std::string::npos) << j;
    EXPECT_NE(j.find("\"groups\""), std::string::npos) << j;
    EXPECT_NE(j.find("\"err_rtn\""), std::string::npos) << j;
    EXPECT_NE(j.find("\"WQ\""), std::string::npos) << j;
    // An uncalibrated export has no groups and must say so rather than omit
    // the key: a reader diffing two reports would see a missing field.
    const std::string rtn = error_report_json(tensors, {}, /*calibrated=*/false);
    EXPECT_NE(rtn.find("\"calibrated\": false"), std::string::npos) << rtn;
    EXPECT_NE(rtn.find("\"groups\": []"), std::string::npos) << rtn;
}

// The tool has been EXPERIMENTAL in a source comment and a doc heading since it
// shipped, and printed neither. A checkpoint outlives the terminal it was made
// in, but the operator who has to judge it reads this line first.
TEST(QuantReport, BannerStatesWhatTheExportIs) {
    const std::string rtn = experimental_banner(/*calibrated=*/false, 0);
    EXPECT_NE(rtn.find("experimental"), std::string::npos) << rtn;
    EXPECT_NE(rtn.find("RTN"), std::string::npos) << rtn;
    EXPECT_NE(rtn.find("no calibration"), std::string::npos) << rtn;

    const std::string awq = experimental_banner(/*calibrated=*/true, 4096);
    EXPECT_NE(awq.find("experimental"), std::string::npos) << awq;
    EXPECT_NE(awq.find("AWQ"), std::string::npos) << awq;
    EXPECT_NE(awq.find("4096"), std::string::npos) << awq;
    EXPECT_NE(awq.find("no vendor export"), std::string::npos) << awq;
    // One line, both arms: this is printed before anything else and must not
    // push the rest of the output off a screen.
    EXPECT_EQ(rtn.find('\n'), std::string::npos);
    EXPECT_EQ(awq.find('\n'), std::string::npos);
}
