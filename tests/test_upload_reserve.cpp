// Weight-upload reserve and the batch it fits (memory/upload_reserve.h, #2393). CPU-only.

#include <gtest/gtest.h>

#include "memory/upload_reserve.h"

#include <cmath>
#include <string>

using namespace imp;

namespace {

constexpr size_t kMiB = 1024ull * 1024;

size_t mib(double v) { return static_cast<size_t>(std::llround(v * static_cast<double>(kMiB))); }

// Qwen3.8-27B-NVFP4, max_seq_len 131072, NVFP4 KV, BF16 state (the #2393 load log):
// workspace 948.62 + safety 256 + snapshot 256; KV capped at total/5 = 6515.89; 79.5 MiB per slot.
UploadReserve qwen38_reserve() {
    UploadReserve r;
    r.fixed_bytes = mib(948.62) + mib(256) + mib(256);
    r.kv_bytes_per_seq = mib(2304);  // 8192 blocks x 16 attn layers x 18432 B
    r.kv_cap_bytes = mib(6515.89);
    r.state_bytes_per_slot = mib(79.5);
    r.reserved_state_slots = 0;
    return r;
}

// Budget-view free at upload start: 30691 MiB raw minus the arena's uncommitted 1999.5 MiB.
constexpr double kFreeMiB = 30691.0 - (2095.5 - 96.0);
// Pass-1 weights: 15612.8 MiB at the refusal in layer 59, plus layers 59-63 (estimate).
constexpr double kWeightsMiB = 16640.0;

}  // namespace

TEST(UploadReserve, ReproducesTheLoggedReserveAt64Slots) {
    // "Expert upload reserve: 13064.51 MiB (workspace=948.62, kv=6515.89, ...)"
    const UploadReserve r = qwen38_reserve();
    EXPECT_NEAR(upload_reserve_bytes(r, 64) / double(kMiB), 13064.51, 0.01);
    EXPECT_NEAR(upload_reserve_kv_bytes(r, 64) / double(kMiB), 6515.89, 0.01);
    // Below the KV cap a slot also costs its KV share.
    EXPECT_EQ(upload_reserve_kv_bytes(r, 1), mib(2304));
}

TEST(UploadReserve, Qwen38At64SlotsIsClampedNotAborted) {
    const UploadReserve r = qwen38_reserve();
    const size_t free_bytes = mib(kFreeMiB), weights = mib(kWeightsMiB);
    ASSERT_GT(weights + upload_reserve_bytes(r, 64), free_bytes) << "the #2393 abort: 64 slots do not fit";
    const int fit = upload_fitting_batch(r, 64, free_bytes, weights);
    EXPECT_EQ(fit, 51);
    EXPECT_LE(weights + upload_reserve_bytes(r, fit), free_bytes);
    EXPECT_GT(weights + upload_reserve_bytes(r, fit + 1), free_bytes) << "the largest fitting batch";
}

TEST(UploadReserve, AFittingBatchIsKept) {
    const UploadReserve r = qwen38_reserve();
    const size_t free_bytes = mib(kFreeMiB), weights = mib(kWeightsMiB);
    EXPECT_EQ(upload_fitting_batch(r, 32, free_bytes, weights), 32);
    EXPECT_EQ(upload_fitting_batch(r, 51, free_bytes, weights), 51);
    EXPECT_EQ(upload_fitting_batch(r, 1, free_bytes, weights), 1);
}

TEST(UploadReserve, ZeroWhenOneSlotDoesNotFit) {
    const UploadReserve r = qwen38_reserve();
    const size_t weights = mib(kWeightsMiB);
    EXPECT_EQ(upload_fitting_batch(r, 64, weights + upload_reserve_bytes(r, 1) - 1, weights), 0);
    EXPECT_EQ(upload_fitting_batch(r, 64, weights + upload_reserve_bytes(r, 1), weights), 1);
}

TEST(UploadReserve, ReservedStateSlotsAreChargedPastTheBatch) {
    UploadReserve r = qwen38_reserve();
    const size_t at64 = upload_reserve_bytes(r, 64);
    r.reserved_state_slots = 3;
    EXPECT_EQ(upload_reserve_bytes(r, 64), at64 + 3 * mib(79.5));
}

TEST(UploadReserve, ClampMessageNamesConfiguredFittedAndPerSlotCost) {
    const UploadReserve r = qwen38_reserve();
    const std::string m = upload_batch_clamp_message(r, 64, 51, mib(kFreeMiB), mib(kWeightsMiB));
    EXPECT_NE(m.find("clamped 64 -> 51"), std::string::npos) << m;
    EXPECT_NE(m.find("79.5 MiB SSM/GDN state"), std::string::npos) << m;
    EXPECT_NE(m.find("13065 MiB reserve at 64 slots"), std::string::npos) << m;
    EXPECT_NE(m.find("runtime.max_batch_size=51"), std::string::npos) << m;
}
