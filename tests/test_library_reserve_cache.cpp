// The persisted library-reserve measurement (AUDIT B41/B42/B49).
//
// CPU lane: it is a keyed text file and nothing else, which is deliberate — the
// number it carries decides how much VRAM the plan hands out, so the thing that
// stores it should be boring and testable without a GPU.
//
// What matters here is the failure behaviour. A cache miss, a truncated file or
// an unwritable directory must all read as "no entry, charge the constant and
// measure again"; refusing to serve a model over a cache file would be absurd.

#include <gtest/gtest.h>

#include "memory/library_reserve_cache.h"

#include <filesystem>
#include <fstream>
#include <string>

using namespace imp;

namespace {

LibraryReserveKey key_a() {
    LibraryReserveKey k;
    k.model_fingerprint = 0xdeadbeefcafe1234ull;
    k.nvfp4_decode_mode = 1;
    k.fp8_prefill = false;
    k.cuda_runtime_version = 13030;
    return k;
}

std::string tmp_path(const char* name) {
    auto p = std::filesystem::temp_directory_path() / "imp_lrc_test" / name;
    std::filesystem::create_directories(p.parent_path());
    std::filesystem::remove(p);
    return p.string();
}

}  // namespace

TEST(LibraryReserveCache, KeyIsStableAndSeparatesWhatTheChargeVariesWith) {
    const LibraryReserveKey a = key_a();
    EXPECT_EQ(a.str(), key_a().str()) << "the same inputs must produce the same key";

    LibraryReserveKey other_model = a;
    other_model.model_fingerprint ^= 1;
    LibraryReserveKey other_path = a;
    other_path.nvfp4_decode_mode = 2;
    LibraryReserveKey other_fp8 = a;
    other_fp8.fp8_prefill = true;
    LibraryReserveKey other_cuda = a;
    other_cuda.cuda_runtime_version = 13040;

    // Each of these selects a different execution path or library stack, which
    // AUDIT B41 measured as the thing the charge actually varies with.
    EXPECT_NE(a.str(), other_model.str());
    EXPECT_NE(a.str(), other_path.str());
    EXPECT_NE(a.str(), other_fp8.str());
    EXPECT_NE(a.str(), other_cuda.str());

    // And a key can never contain the separators the file format relies on.
    EXPECT_EQ(a.str().find('\t'), std::string::npos);
    EXPECT_EQ(a.str().find('\n'), std::string::npos);
}

TEST(LibraryReserveCache, RoundTrips) {
    const std::string p = tmp_path("roundtrip");
    ASSERT_TRUE(library_reserve_cache_store(p, key_a(), 7822376960ull));
    EXPECT_EQ(library_reserve_cache_load(p, key_a()), 7822376960ull);
}

TEST(LibraryReserveCache, MissingFileAndUnknownKeyBothReadAsNoEntry) {
    EXPECT_EQ(library_reserve_cache_load(tmp_path("absent"), key_a()), 0u);

    const std::string p = tmp_path("other_key");
    ASSERT_TRUE(library_reserve_cache_store(p, key_a(), 4096));
    LibraryReserveKey missing = key_a();
    missing.model_fingerprint ^= 0xffull;
    EXPECT_EQ(library_reserve_cache_load(p, missing), 0u)
        << "a different model must not inherit another model's measurement";
}

TEST(LibraryReserveCache, ARemeasurementReplacesTheOldValue) {
    const std::string p = tmp_path("replace");
    ASSERT_TRUE(library_reserve_cache_store(p, key_a(), 1000));
    ASSERT_TRUE(library_reserve_cache_store(p, key_a(), 2000));
    EXPECT_EQ(library_reserve_cache_load(p, key_a()), 2000u);

    // And it must not have appended a second line for the same key: the loader
    // takes the last, but a file that grows per boot is still a defect.
    std::ifstream in(p);
    std::string line;
    int matches = 0;
    while (std::getline(in, line))
        if (line.rfind(key_a().str(), 0) == 0)
            ++matches;
    EXPECT_EQ(matches, 1);
}

TEST(LibraryReserveCache, EntriesForDifferentKeysCoexist) {
    const std::string p = tmp_path("multi");
    LibraryReserveKey b = key_a();
    b.nvfp4_decode_mode = 2;
    ASSERT_TRUE(library_reserve_cache_store(p, key_a(), 1111));
    ASSERT_TRUE(library_reserve_cache_store(p, b, 2222));
    EXPECT_EQ(library_reserve_cache_load(p, key_a()), 1111u);
    EXPECT_EQ(library_reserve_cache_load(p, b), 2222u)
        << "storing a second key must not drop the first — one host serves many models";
}

// A half-written or hand-edited file must degrade to "no entry", never to a
// wrong number: a bogus reserve is worse than no reserve, because the plan would
// hand out VRAM the libraries then take.
TEST(LibraryReserveCache, MalformedLinesAreSkippedNotGuessedAt) {
    const std::string p = tmp_path("corrupt");
    {
        std::ofstream out(p);
        out << "# a comment line\n"
            << "no-tab-here 1234\n"
            << "\t5678\n"
            << key_a().str() << "\tnot-a-number\n";
    }
    EXPECT_EQ(library_reserve_cache_load(p, key_a()), 0u);

    // …and a subsequent store still succeeds over the damaged file.
    ASSERT_TRUE(library_reserve_cache_store(p, key_a(), 4242));
    EXPECT_EQ(library_reserve_cache_load(p, key_a()), 4242u);
}

TEST(LibraryReserveCache, AnEmptyPathIsAWorkingNoOp) {
    EXPECT_EQ(library_reserve_cache_load("", key_a()), 0u);
    EXPECT_FALSE(library_reserve_cache_store("", key_a(), 1234));
}

TEST(LibraryReserveCache, AnUnwritablePathFailsWithoutThrowing) {
    // /proc is present and not writable in every environment this runs in.
    EXPECT_FALSE(library_reserve_cache_store("/proc/imp_lrc_should_fail/x", key_a(), 1));
}
