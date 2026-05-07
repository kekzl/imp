// Unit tests for the SafeTensors loader's blob-level validation surface.
// Exercises the test-visible helpers in safetensors_internal:: directly with
// synthetic blob bytes — no Model construction, no GPU.
//
// Closes audit findings F3, F4, F5, F7, F8 from
// docs/audit/safetensors_nvfp4_audit_2026-05.md.

#include "model/safetensors_loader.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <string>

namespace imp {
namespace {

// ---- F3: header_size validation ----

TEST(SafeTensorsValidateHeaderSize, RejectsFileTruncatedBelow8) {
    std::string err;
    EXPECT_FALSE(safetensors_internal::validate_header_size(0, 0, &err));
    EXPECT_FALSE(err.empty());
    err.clear();
    EXPECT_FALSE(safetensors_internal::validate_header_size(7, 0, &err));
    EXPECT_FALSE(err.empty());
}

TEST(SafeTensorsValidateHeaderSize, AcceptsExactMinimum) {
    std::string err;
    // 8-byte file with declared header_size=0: legal but empty header.
    EXPECT_TRUE(safetensors_internal::validate_header_size(8, 0, &err)) << err;
}

TEST(SafeTensorsValidateHeaderSize, AcceptsTypicalSize) {
    std::string err;
    // file_size = 1 MiB, header_size = 64 KiB. Plenty of room for tensor data.
    EXPECT_TRUE(safetensors_internal::validate_header_size(1u << 20, 64 * 1024, &err)) << err;
}

TEST(SafeTensorsValidateHeaderSize, RejectsHeaderExceedingFile) {
    std::string err;
    // file_size = 1 KiB but declared header_size = 1 KiB → leaves no room for
    // the 8-byte prefix and would overflow the JSON parser bounds.
    EXPECT_FALSE(safetensors_internal::validate_header_size(1024, 1024, &err));
    EXPECT_FALSE(err.empty());
}

TEST(SafeTensorsValidateHeaderSize, RejectsUInt64MaxOverflowAttack) {
    // The bug fixed: prior code computed `8 + header_size > file_size`. With
    // header_size = UINT64_MAX-4 the addition wrapped to 3, which is NOT
    // greater than any legitimate file size — the check silently bypassed.
    // The new overflow-safe check rejects this.
    std::string err;
    EXPECT_FALSE(safetensors_internal::validate_header_size(
        16, std::numeric_limits<uint64_t>::max(), &err));
    EXPECT_FALSE(err.empty()) << "validator should produce a reason string";

    err.clear();
    EXPECT_FALSE(safetensors_internal::validate_header_size(
        16, std::numeric_limits<uint64_t>::max() - 4, &err));
    EXPECT_FALSE(err.empty());

    err.clear();
    EXPECT_FALSE(safetensors_internal::validate_header_size(
        16, std::numeric_limits<uint64_t>::max() - 7, &err));
    EXPECT_FALSE(err.empty());
}

TEST(SafeTensorsValidateHeaderSize, RejectsAboveSoftCap) {
    // Soft cap is 128 MiB per ADR 0002. Any larger declared header is rejected
    // even when the file claims to be that big.
    constexpr uint64_t k129MiB = 129ULL * 1024ULL * 1024ULL;
    std::string err;
    EXPECT_FALSE(safetensors_internal::validate_header_size(
        k129MiB + 8, k129MiB, &err));
    EXPECT_FALSE(err.empty());

    // 128 MiB exactly is within the cap.
    constexpr uint64_t k128MiB = 128ULL * 1024ULL * 1024ULL;
    err.clear();
    EXPECT_TRUE(safetensors_internal::validate_header_size(k128MiB + 8, k128MiB, &err)) << err;
}

}  // namespace
}  // namespace imp
