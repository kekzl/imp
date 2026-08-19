#include <gtest/gtest.h>

#include "model/mtp_head.h"

#include <cstdint>
#include <vector>

namespace imp {
namespace {

// mtp_upload_peak_bytes decides whether the MTP head is uploaded at all. The
// upload is 272 allocations on a per-expert checkpoint and the allocator
// decides per allocation, so running out partway leaves everything already
// uploaded stranded until the process exits. Under-estimating here is what
// lets that happen.

// A stand-in expert weight. The estimator reads shape and the non-null data
// pointer only, so no device memory is involved.
Tensor fake_expert(int64_t rows, int64_t cols) {
    static int placeholder = 0;
    const int64_t shape[4] = {rows, cols, 0, 0};
    return Tensor(&placeholder, QType::F16, 2, shape, /*on_device=*/true);
}

TEST(MtpUploadBudgetTest, PackedLayoutCostsTheFileSize) {
    MtpHead head;
    head.info.file_bytes = 1608ull * 1024 * 1024;

    // No per-expert tensors: nothing is restacked, so nothing is copied twice.
    EXPECT_TRUE(head.experts_up.empty());
    EXPECT_EQ(mtp_upload_peak_bytes(head), head.info.file_bytes);
}

TEST(MtpUploadBudgetTest, PerExpertLayoutAddsBothSlabsAndPoolGrowth) {
    // Nemotron-3.5-Lightning-30B: 128 experts, hidden 2688, d_ff_e 1856.
    constexpr int64_t kExperts = 128, kHidden = 2688, kDffE = 1856;
    MtpHead head;
    head.info.file_bytes = 2550ull * 1024 * 1024;
    for (int64_t e = 0; e < kExperts; e++) {
        head.experts_up.push_back(fake_expert(kDffE, kHidden));
        head.experts_down.push_back(fake_expert(kHidden, kDffE));
    }

    // Both slabs, plus one more for the pool growth the first large request
    // triggers. Probed phase by phase on this checkpoint: 2424 MiB went for the
    // first 1218 MiB slab and 1216 MiB for the second.
    const size_t slab = static_cast<size_t>(kExperts) * kDffE * kHidden * 2;
    EXPECT_EQ(mtp_upload_peak_bytes(head), head.info.file_bytes + 3 * slab);

    // Which lands within 4 MiB of the 6200 MiB the load was measured taking.
    const size_t measured = 6200ull * 1024 * 1024;
    const size_t est = mtp_upload_peak_bytes(head);
    const size_t off = est > measured ? est - measured : measured - est;
    EXPECT_LT(off, 16ull * 1024 * 1024)
        << "estimate " << est / (1024 * 1024) << " MiB against 6200 MiB measured";

    // And it must not be reachable by the file size alone, or a per-expert head
    // gets waved through and strands its allocations partway.
    EXPECT_GT(est, 2 * head.info.file_bytes);
}

TEST(MtpUploadBudgetTest, UnuploadedExpertsAreNotCounted) {
    // Before the upload runs, the per-expert tensors carry host pointers or
    // none at all. Counting a slab for those would refuse heads that fit.
    MtpHead head;
    head.info.file_bytes = 1000;
    const int64_t shape[4] = {16, 16, 0, 0};
    head.experts_up.push_back(Tensor(nullptr, QType::F16, 2, shape, /*on_device=*/false));
    head.experts_down.push_back(Tensor(nullptr, QType::F16, 2, shape, /*on_device=*/false));

    EXPECT_EQ(mtp_upload_peak_bytes(head), 1000u);
}

}  // namespace
}  // namespace imp
