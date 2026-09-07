// Dropping the weight-file pages must not change what reads back.
//
// The loaders map the checkpoint with MAP_POPULATE + MADV_WILLNEED and nothing
// dropped it again, so a serving process held the whole file resident for its
// lifetime: measured 18.48 GiB of file-backed RSS out of 21.53 GiB total on
// Qwen3.8-27B-NVFP4-vllm, while `docker stats` reported 3.2 GiB because the
// cgroup does not account it. It cost another session two OOM-killed jobs.
//
// The release is MADV_DONTNEED, not munmap, and that choice is the whole safety
// argument: a host-resident expert or an offloaded layer still holds a pointer
// INTO the mapping (`executor_forward_moe_nvfp4_host.cu` passes `w.data` to the
// expert cache), and there is no tensor iterator that could prove otherwise
// field by field. Dropping pages keeps every such pointer valid and correct;
// unmapping would turn one missed field into a use-after-free.
//
// This pins that property on a real mapping, with no GPU and no checkpoint.

#include "model/model.h"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace imp {
namespace {

// A file mapped exactly the way the loaders map a checkpoint.
struct MappedFile {
    std::string path;
    void* base = nullptr;
    size_t size = 0;

    explicit MappedFile(size_t bytes) : size(bytes) {
        char tmpl[] = "/tmp/imp_pagerelease_XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd < 0)
            return;
        path = tmpl;
        std::vector<uint8_t> pattern(size);
        for (size_t i = 0; i < size; ++i)
            pattern[i] = static_cast<uint8_t>((i * 31u + 7u) & 0xFF);
        ssize_t w = write(fd, pattern.data(), pattern.size());
        (void)w;
        base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        if (base == MAP_FAILED)
            base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base == MAP_FAILED)
            base = nullptr;
        close(fd);
    }
    ~MappedFile() {
        if (!path.empty())
            ::remove(path.c_str());
    }
    static uint8_t expected(size_t i) { return static_cast<uint8_t>((i * 31u + 7u) & 0xFF); }
};

constexpr size_t kMapBytes = 2u * 1024u * 1024u;  // 2 MiB, several pages

// The bytes must survive the release: MADV_DONTNEED on a read-only private file
// mapping discards resident pages, and the next read refaults them from the
// page cache. If this ever became munmap, this test segfaults, which is the
// point.
TEST(WeightPageRelease, DataStillReadsBackAfterRelease) {
    MappedFile f(kMapBytes);
    ASSERT_NE(f.base, nullptr) << "could not map the fixture file";

    Model model;
    model.mmap_base_ = f.base;
    model.mmap_size_ = f.size;

    const auto* bytes = static_cast<const uint8_t*>(f.base);
    ASSERT_EQ(bytes[0], MappedFile::expected(0)) << "fixture did not map its own contents";

    const size_t advised = model.release_weight_pages();
    EXPECT_EQ(advised, f.size) << "the whole mapping should have been advised away";

    // Every page, not just the first: a partial or misaligned advise would show
    // up here and nowhere else.
    for (size_t i = 0; i < f.size; i += 4096)
        ASSERT_EQ(bytes[i], MappedFile::expected(i)) << "byte " << i << " changed across the release";
    EXPECT_EQ(bytes[f.size - 1], MappedFile::expected(f.size - 1));

    // The Model destructor owns the unmap; hand it over rather than double-free.
}

// Additional shard mappings are released too. The SafeTensors loader puts every
// shard past the first into split_mmaps_, and on a sharded checkpoint that is
// where nearly all of the bytes are.
TEST(WeightPageRelease, SplitShardMappingsAreReleasedToo) {
    MappedFile a(kMapBytes);
    MappedFile b(kMapBytes);
    ASSERT_NE(a.base, nullptr);
    ASSERT_NE(b.base, nullptr);

    Model model;
    model.mmap_base_ = a.base;
    model.mmap_size_ = a.size;
    model.split_mmaps_.emplace_back(b.base, b.size);

    EXPECT_EQ(model.release_weight_pages(), a.size + b.size);

    const auto* bb = static_cast<const uint8_t*>(b.base);
    for (size_t i = 0; i < b.size; i += 4096)
        ASSERT_EQ(bb[i], MappedFile::expected(i)) << "shard byte " << i << " changed";
}

// A model that was never mapped (GGUF loaded from a warm cache, a stub model in
// a test) must be a no-op rather than an madvise on a null pointer.
TEST(WeightPageRelease, NoMappingIsANoOp) {
    Model model;
    EXPECT_EQ(model.release_weight_pages(), 0u);
}

// Calling it twice is harmless: the second call re-advises pages that may have
// refaulted in between. Serving does not do this, but a future caller might.
TEST(WeightPageRelease, ReleaseIsIdempotent) {
    MappedFile f(kMapBytes);
    ASSERT_NE(f.base, nullptr);
    Model model;
    model.mmap_base_ = f.base;
    model.mmap_size_ = f.size;

    EXPECT_EQ(model.release_weight_pages(), f.size);
    EXPECT_EQ(model.release_weight_pages(), f.size);
    const auto* bytes = static_cast<const uint8_t*>(f.base);
    EXPECT_EQ(bytes[1234], MappedFile::expected(1234));
}

}  // namespace
}  // namespace imp
