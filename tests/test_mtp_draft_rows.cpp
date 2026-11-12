// compute/mtp_draft_rows.h: rows the NVFP4 draft copy holds and rows one MTP draft pass scores. A pass
// that scores more rows than the copy holds reads past its buffer. CPU-only.

#include <gtest/gtest.h>

#include "compute/mtp_draft_rows.h"

namespace imp {
namespace {

constexpr int kVocab = 248320;  // Qwen3.8-27B
constexpr int kPrefix = 98304;

TEST(MtpDraftRows, CopyHoldsThePrefixOrEveryRow) {
    EXPECT_EQ(mtp_draft_copy_rows(kVocab, kPrefix), kPrefix);
    EXPECT_EQ(mtp_draft_copy_rows(kVocab, 0), kVocab);
    EXPECT_EQ(mtp_draft_copy_rows(kVocab, -1), kVocab);
    EXPECT_EQ(mtp_draft_copy_rows(kVocab, kVocab), kVocab);
    EXPECT_EQ(mtp_draft_copy_rows(kVocab, kVocab + 1), kVocab) << "a prefix past the head is the head";
}

TEST(MtpDraftRows, PassNeverReadsPastThePrefixCopy) {
    EXPECT_EQ(mtp_draft_pass_rows(kVocab, kPrefix, /*fp8=*/false, kPrefix), kPrefix);
    EXPECT_EQ(mtp_draft_pass_rows(kVocab, kPrefix, /*fp8=*/false, 0), kPrefix)
        << "the lone chain passes no draft_vocab: the copy's rows bound the pass";
    EXPECT_EQ(mtp_draft_pass_rows(kVocab, kPrefix, /*fp8=*/false, kVocab), kPrefix);
    EXPECT_EQ(mtp_draft_pass_rows(kVocab, kPrefix, /*fp8=*/false, 65536), 65536)
        << "draft_vocab below the copy scores fewer rows";
}

TEST(MtpDraftRows, FullCopyAndFp8HeadFollowDraftVocab) {
    EXPECT_EQ(mtp_draft_pass_rows(kVocab, kVocab, /*fp8=*/false, 0), kVocab);
    EXPECT_EQ(mtp_draft_pass_rows(kVocab, kVocab, /*fp8=*/false, kPrefix), kPrefix);
    EXPECT_EQ(mtp_draft_pass_rows(kVocab, 0, /*fp8=*/true, kPrefix), kPrefix);
    EXPECT_EQ(mtp_draft_pass_rows(kVocab, 0, /*fp8=*/true, 0), kVocab);
}

TEST(MtpDraftRows, CutlassHeadScoresEveryRow) {
    EXPECT_EQ(mtp_draft_pass_rows(kVocab, 0, /*fp8=*/false, kPrefix), kVocab)
        << "the W4A4 head GEMM has no row prefix";
}

}  // namespace
}  // namespace imp
