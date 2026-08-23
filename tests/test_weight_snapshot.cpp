// Unit tests for the suspend-to-RAM weight snapshot plumbing (CPU-only):
// key canonicalization, /proc/meminfo parsing, WeightUploadLog record/evict
// semantics, and the pending-arm slot. The D2H capture + warm restore paths
// need a GPU + real model and live in tests/test_suspend_resume.cu.

#include <gtest/gtest.h>

#include "core/tensor.h"
#include "memory/weight_snapshot.h"

#include <cstdint>

namespace {

using imp::make_weight_key;
using imp::parse_meminfo_available;
using imp::Tensor;
using imp::QType;
using imp::WeightUploadLog;

TEST(WeightKeyTest, CanonicalFormats) {
    char buf[96];
    EXPECT_STREQ(make_weight_key(buf, "wq", 5), "L5.wq");
    EXPECT_STREQ(make_weight_key(buf, "expert_w_gate.7", 12), "L12.expert_w_gate.7");
    EXPECT_STREQ(make_weight_key(buf, "tok_emb", -1), "tok_emb");
    EXPECT_STREQ(make_weight_key(buf, "q_norm", imp::kMtpKeyLayer), "mtp.q_norm");
}

TEST(MeminfoParseTest, ParsesMemAvailable) {
    EXPECT_EQ(parse_meminfo_available("MemTotal:       65536000 kB\n"
                                      "MemFree:         1234 kB\n"
                                      "MemAvailable:   32768000 kB\n"),
              32768000ull * 1024);
    // Leading lines missing / label absent → 0 (gate degrades to "no check").
    EXPECT_EQ(parse_meminfo_available("MemTotal: 1 kB\nMemFree: 1 kB\n"), 0u);
    // Malformed digits → 0.
    EXPECT_EQ(parse_meminfo_available("MemAvailable:   garbage kB\n"), 0u);
    EXPECT_EQ(parse_meminfo_available(""), 0u);
}

// Fake "device" allocations — the log never dereferences them.
struct FakeAllocs {
    alignas(16) uint8_t a1[256];
    alignas(16) uint8_t a2[64];
};

TEST(WeightUploadLogTest, RecordResolvesSizesAndOffsets) {
    FakeAllocs f;
    WeightUploadLog log;
    log.note_alloc(f.a1, sizeof(f.a1));
    log.note_alloc(f.a2, sizeof(f.a2));

    int64_t shape[4] = {8, 16, 0, 0};
    Tensor post(f.a1 + 32, QType::F16, 2, shape, /*on_device=*/true);  // interior data pointer
    post.scales = f.a2;                                                // scales in the second alloc

    const void* const allocs[] = {f.a1, f.a2};
    log.record("L0.wq", allocs, 2, post, QType::BF16, 128, 256);

    ASSERT_EQ(log.records().size(), 1u);
    const auto& rec = log.records()[0];
    EXPECT_EQ(rec.key, "L0.wq");
    ASSERT_EQ(rec.allocs.size(), 2u);
    EXPECT_EQ(rec.allocs[0].bytes, sizeof(f.a1));
    EXPECT_EQ(rec.allocs[1].bytes, sizeof(f.a2));
    EXPECT_EQ(rec.data_alloc, 0);
    EXPECT_EQ(rec.data_off, 32u);
    EXPECT_EQ(rec.scales_alloc, 1);
    EXPECT_EQ(rec.scales_off, 0u);
    EXPECT_EQ(rec.src_qtype, QType::BF16);
    EXPECT_EQ(rec.src_numel, 128);
    EXPECT_FALSE(rec.dead);
    EXPECT_EQ(log.live_bytes(), sizeof(f.a1) + sizeof(f.a2));
}

TEST(WeightUploadLogTest, UnknownAllocSizeSkipsRecord) {
    FakeAllocs f;
    WeightUploadLog log;  // note_alloc never called → size unknown
    int64_t shape[4] = {4, 4, 0, 0};
    Tensor post(f.a1, QType::F16, 2, shape, true);
    const void* const allocs[] = {f.a1};
    log.record("L0.wq", allocs, 1, post, QType::F16, 16, 32);
    EXPECT_TRUE(log.records().empty());
}

TEST(WeightUploadLogTest, DataOutsideAllocationsSkipsRecord) {
    FakeAllocs f;
    WeightUploadLog log;
    log.note_alloc(f.a1, sizeof(f.a1));
    int64_t shape[4] = {4, 4, 0, 0};
    Tensor post(f.a2, QType::F16, 2, shape, true);  // data not backed by a1
    const void* const allocs[] = {f.a1};
    log.record("L0.wq", allocs, 1, post, QType::F16, 16, 32);
    EXPECT_TRUE(log.records().empty());
}

TEST(WeightUploadLogTest, SameKeyReplaces) {
    FakeAllocs f;
    WeightUploadLog log;
    log.note_alloc(f.a1, sizeof(f.a1));
    log.note_alloc(f.a2, sizeof(f.a2));
    int64_t shape[4] = {4, 4, 0, 0};
    Tensor post1(f.a1, QType::F16, 2, shape, true);
    Tensor post2(f.a2, QType::F16, 2, shape, true);
    const void* const allocs1[] = {f.a1};
    const void* const allocs2[] = {f.a2};
    log.record("L0.wq", allocs1, 1, post1, QType::F16, 16, 32);
    log.record("L0.wq", allocs2, 1, post2, QType::F16, 16, 32);
    ASSERT_EQ(log.records().size(), 1u);
    EXPECT_EQ(log.records()[0].allocs[0].ptr, static_cast<void*>(f.a2));
}

TEST(WeightUploadLogTest, EvictMarksDeadAndDropsLiveBytes) {
    FakeAllocs f;
    WeightUploadLog log;
    log.note_alloc(f.a1, sizeof(f.a1));
    log.note_alloc(f.a2, sizeof(f.a2));
    int64_t shape[4] = {4, 4, 0, 0};
    Tensor p1(f.a1, QType::F16, 2, shape, true);
    Tensor p2(f.a2, QType::F16, 2, shape, true);
    const void* const allocs1[] = {f.a1};
    const void* const allocs2[] = {f.a2};
    log.record("L0.wq", allocs1, 1, p1, QType::F16, 16, 32);
    log.record("L0.wk", allocs2, 1, p2, QType::F16, 16, 32);
    EXPECT_EQ(log.live_bytes(), sizeof(f.a1) + sizeof(f.a2));

    log.evict_ptr(f.a1);
    ASSERT_EQ(log.records().size(), 2u);
    EXPECT_TRUE(log.records()[0].dead);
    EXPECT_FALSE(log.records()[1].dead);
    EXPECT_EQ(log.live_bytes(), sizeof(f.a2));

    log.evict_ptr(nullptr);  // no-op
    EXPECT_EQ(log.live_bytes(), sizeof(f.a2));
}

TEST(WeightUploadLogTest, RawFromSourceHeuristic) {
    FakeAllocs f;
    WeightUploadLog log;
    log.note_alloc(f.a1, sizeof(f.a1));
    log.note_alloc(f.a2, sizeof(f.a2));
    int64_t shape[4] = {4, 4, 0, 0};

    // Verbatim h2d: one alloc, same qtype, byte count matches the source.
    Tensor raw_post(f.a1, QType::Q8_0, 2, shape, true);
    const void* const a1[] = {f.a1};
    log.record("L0.raw", a1, 1, raw_post, QType::Q8_0, 16, sizeof(f.a1));
    EXPECT_TRUE(log.records().back().raw_from_source);

    // Converted: qtype changed (BF16 -> F16 upload).
    Tensor conv_post(f.a2, QType::F16, 2, shape, true);
    const void* const a2[] = {f.a2};
    log.record("L0.conv", a2, 1, conv_post, QType::BF16, 16, sizeof(f.a2));
    EXPECT_FALSE(log.records().back().raw_from_source);

    // MXFP4 keeps qtype + byte count but reorders — excluded from "raw".
    Tensor mx_post(f.a1, QType::MXFP4, 2, shape, true);
    log.record("L0.mx", a1, 1, mx_post, QType::MXFP4, 16, sizeof(f.a1));
    EXPECT_FALSE(log.records().back().raw_from_source);

    // Split upload (two allocs, scales sidecar) is never raw.
    Tensor split_post(f.a1, QType::Q4_0, 2, shape, true);
    split_post.scales = f.a2;
    const void* const both[] = {f.a1, f.a2};
    log.record("L0.split", both, 2, split_post, QType::Q4_0, 16, sizeof(f.a1) + sizeof(f.a2));
    EXPECT_FALSE(log.records().back().raw_from_source);
}

TEST(WeightSnapshotArmSlotTest, ArmTakeDisarm) {
    // The slot stores a non-owning pointer; a dummy address is fine for the
    // slot mechanics (nothing dereferences it here).
    auto* fake = reinterpret_cast<imp::WeightSnapshot*>(0x1234);

    EXPECT_EQ(imp::weight_snapshot_take_armed(), nullptr);
    imp::weight_snapshot_arm(fake);
    EXPECT_EQ(imp::weight_snapshot_take_armed(), fake);
    EXPECT_EQ(imp::weight_snapshot_take_armed(), nullptr);  // consumed

    imp::weight_snapshot_arm(fake);
    imp::weight_snapshot_disarm(fake);
    EXPECT_EQ(imp::weight_snapshot_take_armed(), nullptr);

    // Disarm of a non-armed pointer leaves an armed one alone.
    imp::weight_snapshot_arm(fake);
    imp::weight_snapshot_disarm(reinterpret_cast<imp::WeightSnapshot*>(0x5678));
    EXPECT_EQ(imp::weight_snapshot_take_armed(), fake);
}

}  // namespace
