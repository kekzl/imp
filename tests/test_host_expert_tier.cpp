// HostExpertTier (memory/host_expert_tier.h): LRU over expert units, loads from the mapped file
// with pread, falls back to memcpy for anonymous sources. CPU lane, no CUDA.

#include <gtest/gtest.h>

#include "memory/host_expert_tier.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace imp;

namespace {

struct Arena {
    explicit Arena(size_t n) : bytes(n), p(static_cast<char*>(std::aligned_alloc(4096, n))) {}
    ~Arena() { std::free(p); }
    size_t bytes;
    char* p;
};

std::vector<char> pattern(size_t n, int seed) {
    std::vector<char> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = static_cast<char>(
            ((i + 1 + static_cast<uint64_t>(seed) * 1000003ull) * 0x9E3779B97F4A7C15ull) >> 56);
    return v;
}

}  // namespace

TEST(HostExpertTier, LruEvictsLeastRecentAndKeepsCurrentCall) {
    constexpr size_t pb = 4096, mb = 256;
    const size_t unit = HostExpertTier::unit_bytes(pb, mb);
    Arena arena(unit * 3);
    constexpr int kKeys = 6;
    std::vector<std::vector<char>> pk, ms;
    HostExpertTier tier(arena.p, arena.p, arena.bytes, unit, kKeys, 2);
    ASSERT_EQ(tier.units(), 3);
    for (int k = 0; k < kKeys; ++k) {
        pk.push_back(pattern(pb, k));
        ms.push_back(pattern(mb, 100 + k));
    }
    for (int k = 0; k < kKeys; ++k)
        tier.set_source(k, {pk[k].data(), pb, ms[k].data(), mb});

    auto check = [&](int key, const HostExpertTier::UnitView& v) {
        EXPECT_EQ(0, std::memcmp(v.packed, pk[key].data(), pb)) << "key " << key;
        EXPECT_EQ(0, std::memcmp(v.ms, ms[key].data(), mb)) << "key " << key;
    };
    HostExpertTier::UnitView v[4];
    const int a[] = {0, 1, 2};
    ASSERT_TRUE(tier.acquire(a, 3, v));
    for (int i = 0; i < 3; ++i)
        check(a[i], v[i]);
    const int b[] = {0};  // 0 most recent, 1 least recent
    ASSERT_TRUE(tier.acquire(b, 1, v));
    const int c[] = {3};  // evicts 1
    ASSERT_TRUE(tier.acquire(c, 1, v));
    check(3, v[0]);
    const auto s0 = tier.stats();
    const int d[] = {0, 2, 3};  // all resident
    ASSERT_TRUE(tier.acquire(d, 3, v));
    for (int i = 0; i < 3; ++i)
        check(d[i], v[i]);
    EXPECT_EQ(tier.stats().misses, s0.misses);
    EXPECT_EQ(tier.stats().hits, s0.hits + 3);

    // Duplicates count once; four distinct keys do not fit three units.
    const int dup[] = {4, 4, 5, 5};
    ASSERT_TRUE(tier.acquire(dup, 4, v));
    for (int i = 0; i < 4; ++i)
        check(dup[i], v[i]);
    const int many[] = {0, 1, 2, 3};
    EXPECT_FALSE(tier.acquire(many, 4, v));
    EXPECT_EQ(tier.stats().copies, tier.stats().misses * 2);  // anonymous sources: memcpy path
}

TEST(HostExpertTier, TransientAcquireUsesScratchAndKeepsTheLru) {
    constexpr size_t pb = 4096, mb = 256;
    const size_t unit = HostExpertTier::unit_bytes(pb, mb);
    Arena arena(unit * 5);
    constexpr int kKeys = 8;
    std::vector<std::vector<char>> pk, ms;
    for (int k = 0; k < kKeys; ++k) {
        pk.push_back(pattern(pb, k));
        ms.push_back(pattern(mb, 50 + k));
    }
    HostExpertTier tier(arena.p, arena.p, arena.bytes, unit, kKeys, 2, /*scratch_units=*/2);
    ASSERT_EQ(tier.units(), 3);
    for (int k = 0; k < kKeys; ++k)
        tier.set_source(k, {pk[k].data(), pb, ms[k].data(), mb});
    HostExpertTier::UnitView v[4];
    const int lru[] = {0, 1, 2};
    ASSERT_TRUE(tier.acquire(lru, 3, v));
    const auto s0 = tier.stats();

    // 0 and 1 resident (hits), 5 and 6 load into scratch; four keys exceed three LRU units.
    const int scan[] = {5, 0, 6, 1};
    ASSERT_TRUE(tier.acquire(scan, 4, v, /*transient=*/true));
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(0, std::memcmp(v[i].packed, pk[scan[i]].data(), pb)) << "key " << scan[i];
        EXPECT_EQ(0, std::memcmp(v[i].ms, ms[scan[i]].data(), mb)) << "key " << scan[i];
    }
    EXPECT_EQ(tier.stats().hits, s0.hits + 2);
    EXPECT_EQ(tier.stats().misses, s0.misses + 2);
    // Three missing keys do not fit two scratch units.
    const int wide[] = {5, 6, 7};
    EXPECT_FALSE(tier.acquire(wide, 3, v, true));

    // The LRU still holds 0, 1, 2: a plain acquire of them reads nothing.
    ASSERT_TRUE(tier.acquire(lru, 3, v));
    EXPECT_EQ(tier.stats().misses, s0.misses + 2);
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(0, std::memcmp(v[i].packed, pk[lru[i]].data(), pb)) << "key " << lru[i];
}

TEST(HostExpertTier, ExchangeMovesHitsOutAndTakesVictimsIn) {
    constexpr size_t pb = 4096, mb = 256;
    const size_t unit = HostExpertTier::unit_bytes(pb, mb);
    Arena arena(unit * 7);
    constexpr int kKeys = 10;
    std::vector<std::vector<char>> pk, ms;
    for (int k = 0; k < kKeys; ++k) {
        pk.push_back(pattern(pb, k));
        ms.push_back(pattern(mb, 30 + k));
    }
    HostExpertTier tier(arena.p, arena.p, arena.bytes, unit, kKeys, 2, /*scratch_units=*/3);
    ASSERT_EQ(tier.units(), 4);
    for (int k = 0; k < kKeys; ++k)
        tier.set_source(k, {pk[k].data(), pb, ms[k].data(), mb});
    HostExpertTier::UnitView v[3];
    const int warm[] = {0, 1};
    ASSERT_TRUE(tier.acquire(warm, 2, v));
    const auto s0 = tier.stats();

    // VRAM misses 0 (here) and 5 (not here); 5 replaces VRAM victim 7, 0 an empty slot.
    const int in[] = {0, 5};
    const int out[] = {-1, 7};
    HostExpertTier::MutView wb[2];
    ASSERT_TRUE(tier.exchange(in, 2, v, out, wb));
    EXPECT_EQ(0, std::memcmp(v[0].packed, pk[0].data(), pb));
    EXPECT_EQ(0, std::memcmp(v[1].packed, pk[5].data(), pb));
    EXPECT_EQ(0, std::memcmp(v[1].ms, ms[5].data(), mb));
    EXPECT_EQ(wb[0].packed, nullptr);
    ASSERT_NE(wb[1].packed, nullptr);
    std::memcpy(wb[1].packed, pk[7].data(), pb);  // what the GPU writes back
    std::memcpy(wb[1].ms, ms[7].data(), mb);
    EXPECT_EQ(tier.stats().hits, s0.hits + 1);
    EXPECT_EQ(tier.stats().misses, s0.misses + 1);
    EXPECT_EQ(tier.stats().writebacks, 1u);

    // 7 is resident from the write-back, 0 left with the step, 1 stayed.
    const int k7[] = {7};
    ASSERT_TRUE(tier.acquire(k7, 1, v));
    EXPECT_EQ(0, std::memcmp(v[0].packed, pk[7].data(), pb));
    EXPECT_EQ(0, std::memcmp(v[0].ms, ms[7].data(), mb));
    EXPECT_EQ(tier.stats().misses, s0.misses + 1);
    const int k1[] = {1};
    ASSERT_TRUE(tier.acquire(k1, 1, v));
    EXPECT_EQ(tier.stats().misses, s0.misses + 1);
    const int k0[] = {0};
    ASSERT_TRUE(tier.acquire(k0, 1, v));
    EXPECT_EQ(tier.stats().misses, s0.misses + 2);

    // Four misses exceed three scratch units.
    const int many[] = {2, 3, 4, 6};
    const int none[] = {-1, -1, -1, -1};
    HostExpertTier::UnitView v4[4];
    HostExpertTier::MutView wb4[4];
    EXPECT_FALSE(tier.exchange(many, 4, v4, none, wb4));
}

TEST(HostExpertTier, LoadsFromMappedFileAtAlignedAndUnalignedOffsets) {
    char path[] = "/tmp/imp_host_expert_tier_XXXXXX";
    const int fd = mkstemp(path);
    ASSERT_GE(fd, 0);
    const size_t file_bytes = 1 << 20;
    const std::vector<char> data = pattern(file_bytes, 7);
    ASSERT_EQ(write(fd, data.data(), file_bytes), static_cast<ssize_t>(file_bytes));
    void* map = mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    ASSERT_NE(map, MAP_FAILED);
    const char* m = static_cast<const char*>(map);

    constexpr size_t pb = 70000, mb = 3000;
    const size_t unit = HostExpertTier::unit_bytes(pb, mb);
    Arena arena(unit * 4);
    // key 0: 16-aligned offsets (O_DIRECT where the filesystem allows), key 1: odd offsets
    // (buffered pread), key 2: offsets near EOF (short O_DIRECT superset).
    const size_t offs[3][2] = {{4096 + 32, 200000 + 48},
                               {12345, 300001},
                               {file_bytes - pb - mb - 16, file_bytes - mb}};
    HostExpertTier tier(arena.p, arena.p, arena.bytes, unit, 3, 4);
    for (int k = 0; k < 3; ++k)
        tier.set_source(k, {m + offs[k][0], pb, m + offs[k][1], mb});
    HostExpertTier::UnitView v[3];
    const int keys[] = {0, 1, 2};
    ASSERT_TRUE(tier.acquire(keys, 3, v));
    for (int k = 0; k < 3; ++k) {
        EXPECT_EQ(0, std::memcmp(v[k].packed, data.data() + offs[k][0], pb)) << "key " << k;
        EXPECT_EQ(0, std::memcmp(v[k].ms, data.data() + offs[k][1], mb)) << "key " << k;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(v[k].packed) % 16, 0u) << "key " << k;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(v[k].ms) % 16, 0u) << "key " << k;
    }
    const auto s = tier.stats();
    EXPECT_EQ(s.copies, 0u);
    EXPECT_EQ(s.direct_reads + s.buffered_reads, 6u);
    munmap(map, file_bytes);
    unlink(path);
}

// Units of ~0.9 MiB, as on Flash-Next, at a 16-aligned and an odd file offset.
TEST(HostExpertTier, LoadsLargeUnits) {
    char path[] = "/tmp/imp_host_expert_tier_mc_XXXXXX";
    const int fd = mkstemp(path);
    ASSERT_GE(fd, 0);
    const size_t file_bytes = 4u << 20;
    const std::vector<char> data = pattern(file_bytes, 3);
    ASSERT_EQ(write(fd, data.data(), file_bytes), static_cast<ssize_t>(file_bytes));
    void* map = mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    ASSERT_NE(map, MAP_FAILED);
    const char* m = static_cast<const char*>(map);

    constexpr size_t pb = 900000, mb = 5000;
    const size_t unit = HostExpertTier::unit_bytes(pb, mb);
    Arena arena(unit * 2);
    const size_t offs[2][2] = {{8192 + 16, 3000000}, {1500001, 3100003}};
    HostExpertTier tier(arena.p, arena.p, arena.bytes, unit, 2, 8);
    for (int k = 0; k < 2; ++k)
        tier.set_source(k, {m + offs[k][0], pb, m + offs[k][1], mb});
    HostExpertTier::UnitView v[2];
    const int keys[] = {0, 1};
    ASSERT_TRUE(tier.acquire(keys, 2, v));
    for (int k = 0; k < 2; ++k) {
        EXPECT_EQ(0, std::memcmp(v[k].packed, data.data() + offs[k][0], pb)) << "key " << k;
        EXPECT_EQ(0, std::memcmp(v[k].ms, data.data() + offs[k][1], mb)) << "key " << k;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(v[k].packed) % 16, 0u) << "key " << k;
    }
    EXPECT_EQ(tier.stats().copies, 0u);
    munmap(map, file_bytes);
    unlink(path);
}

TEST(HostFileMap, ResolvesOffsetsFromMapsText) {
    const std::string maps =
        "7f0000000000-7f0000100000 r--p 00200000 08:01 42 /models/a.safetensors\n"
        "7f0000200000-7f0000201000 rw-p 00000000 00:00 0 \n"
        "7f0000300000-7f0000400000 r--p 00000000 08:01 43 /tmp/gone (deleted)\n";
    HostFileMap fm;
    fm.load(&maps);
    // Paths do not exist here: resolve opens nothing and reports not file-backed.
    EXPECT_LT(fm.resolve(reinterpret_cast<const void*>(0x7f0000000010ull)).fd_buffered, 0);
    EXPECT_LT(fm.resolve(reinterpret_cast<const void*>(0x7f0000200010ull)).fd_buffered, 0);
}
