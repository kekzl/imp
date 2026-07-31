// Unit tests for the SafeTensors loader's blob-level validation surface.
// Exercises the test-visible helpers in safetensors_internal:: directly with
// synthetic blob bytes — no Model construction, no GPU.
//
// Closes audit findings F3, F4, F5, F7, F8 from
// docs/audit/safetensors_nvfp4_audit_2026-05.md.

#include "model/safetensors_loader.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace imp {
namespace {

// Helper: write a synthetic single-shard SafeTensors blob to a temp file.
// header_json is the inner JSON header text; tensor_payload_size is how
// much trailing tensor data to allocate (zeroed).
std::string write_temp_blob(const std::string& header_json, size_t tensor_payload_size) {
    char tmpl[] = "/tmp/imp_test_st_XXXXXX";
    int fd = ::mkstemp(tmpl);
    if (fd < 0)
        return "";
    std::string path = tmpl;
    // Add ".safetensors" suffix so load_safetensors path-detection works.
    std::string final_path = path + ".safetensors";
    ::close(fd);
    ::rename(path.c_str(), final_path.c_str());

    std::ofstream out(final_path, std::ios::binary);
    if (!out)
        return "";
    uint64_t hsize = static_cast<uint64_t>(header_json.size());
    out.write(reinterpret_cast<const char*>(&hsize), sizeof(hsize));
    out.write(header_json.data(), header_json.size());
    if (tensor_payload_size > 0) {
        std::vector<char> zeros(tensor_payload_size, 0);
        out.write(zeros.data(), zeros.size());
    }
    out.close();
    return final_path;
}

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

// ---- F4: per-tensor offset validation ----

TEST(SafeTensorsValidateTensorOffsets, AcceptsTypicalValid) {
    // file_size = 1 KiB, tensor_data_offset = 256 (header occupies first 256 B),
    // tensor at [0, 64) inside data block — i.e. 32 FP16 elements.
    std::string err;
    EXPECT_TRUE(safetensors_internal::validate_tensor_offsets(
        /*start=*/0, /*end=*/64, /*expected=*/64, /*tdo=*/256, /*file=*/1024, &err))
        << err;
}

TEST(SafeTensorsValidateTensorOffsets, RejectsStartAfterEnd) {
    std::string err;
    EXPECT_FALSE(safetensors_internal::validate_tensor_offsets(100, 64, 64, 256, 1024, &err));
    EXPECT_FALSE(err.empty());
}

TEST(SafeTensorsValidateTensorOffsets, RejectsEndPastFile) {
    std::string err;
    // file_size = 1024, tdo = 256 → max offset_end is 768. Setting end=1024
    // would put real bytes at file offset 1280 — past EOF.
    EXPECT_FALSE(safetensors_internal::validate_tensor_offsets(0, 1024, 1024, 256, 1024, &err));
    EXPECT_FALSE(err.empty());
}

TEST(SafeTensorsValidateTensorOffsets, RejectsByteCountMismatch) {
    // 32 FP16 elements declared but only 32 bytes (= 16 FP16) of data on disk.
    std::string err;
    EXPECT_FALSE(safetensors_internal::validate_tensor_offsets(0, 32, 64, 256, 1024, &err));
    EXPECT_FALSE(err.empty());
}

TEST(SafeTensorsValidateTensorOffsets, ZeroSizeTensorIsValid) {
    // Some checkpoints emit metadata-only entries with size 0.
    std::string err;
    EXPECT_TRUE(safetensors_internal::validate_tensor_offsets(0, 0, 0, 256, 1024, &err)) << err;
}

TEST(SafeTensorsValidateTensorOffsets, RejectsHeaderSizeInvariantViolation) {
    // tdo > file_size should never reach this validator (header_size check
    // upstream prevents it), but defend in depth.
    std::string err;
    EXPECT_FALSE(safetensors_internal::validate_tensor_offsets(0, 64, 64, /*tdo=*/2000, /*file=*/1024, &err));
    EXPECT_FALSE(err.empty());
}

TEST(SafeTensorsValidateTensorOffsets, EndExactlyAtFileBoundary) {
    // tdo = 8, file_size = 1024 → max usable end is 1016. Exactly that should pass.
    std::string err;
    EXPECT_TRUE(safetensors_internal::validate_tensor_offsets(0, 1016, 1016, 8, 1024, &err)) << err;
}

// ---- F5: malformed-tensor-entry warnings ----

// Loads a synthetic blob with one tensor missing 'dtype' and one with malformed
// 'shape'. load_safetensors returns nullptr (no config.json → cannot build a
// Model) but the per-shard load must drop both bad tensors with a WARN naming
// each, AND log an end-of-shard summary. We capture stderr and check.
TEST(SafeTensorsMalformedEntryWarnings, MissingDtypeAndShapeWarn) {
    // Two malformed tensors; offsets are valid in case they get past the dtype/shape checks.
    const std::string header =
        "{\"bad_no_dtype\": {\"shape\": [4], \"data_offsets\": [0, 16]},"
        " \"bad_no_shape\": {\"dtype\": \"F32\", \"data_offsets\": [0, 16]}}";
    std::string path = write_temp_blob(header, 16);
    ASSERT_FALSE(path.empty());

    testing::internal::CaptureStderr();
    auto model = load_safetensors(path);
    std::string captured = testing::internal::GetCapturedStderr();
    std::remove(path.c_str());

    // Model build fails (no config.json) — but the per-shard scan must have run
    // and emitted the WARN lines.
    EXPECT_EQ(model.get(), nullptr);
    EXPECT_NE(captured.find("bad_no_dtype"), std::string::npos)
        << "Expected WARN naming the dtype-less tensor. Captured: " << captured;
    EXPECT_NE(captured.find("bad_no_shape"), std::string::npos)
        << "Expected WARN naming the shape-less tensor. Captured: " << captured;
    EXPECT_NE(captured.find("dropped"), std::string::npos)
        << "Expected end-of-shard summary line. Captured: " << captured;
}

TEST(SafeTensorsMalformedEntryWarnings, OffsetByteCountMismatchWarns) {
    // 4 FP32 elements declared, but only 8 bytes of tensor data (= 2 FP32).
    const std::string header =
        "{\"size_mismatch\": {\"dtype\": \"F32\", \"shape\": [4], \"data_offsets\": [0, 8]}}";
    std::string path = write_temp_blob(header, 16);
    ASSERT_FALSE(path.empty());

    testing::internal::CaptureStderr();
    auto model = load_safetensors(path);
    std::string captured = testing::internal::GetCapturedStderr();
    std::remove(path.c_str());

    EXPECT_EQ(model.get(), nullptr);
    EXPECT_NE(captured.find("size_mismatch"), std::string::npos) << captured;
    // The WARN includes the validate_tensor_offsets reason text.
    EXPECT_NE(captured.find("byte count"), std::string::npos) << captured;
}

// A tensor with more dims than the engine's kMaxDims used to be DROPPED with a
// WARN — which silently loses a weight. Qwen3-VL's patch embed is
// [1024, 3, 2, 16, 16] and vanished exactly that way. It is now flattened to
// [d0, d1*..*dn] instead: element order is row-major and untouched, so this is
// a pure reinterpretation, and it is the only shape the GEMM path could consume.
TEST(SafeTensorsHighDimTensors, FlattenedInsteadOfDropped) {
    // [2, 3, 2, 2, 2] F32 = 48 elements = 192 bytes. Flattens to [2, 24].
    const std::string header =
        "{\"conv5d\": {\"dtype\": \"F32\", \"shape\": [2, 3, 2, 2, 2], \"data_offsets\": [0, 192]}}";
    std::string path = write_temp_blob(header, 192);
    ASSERT_FALSE(path.empty());

    testing::internal::CaptureStderr();
    auto model = load_safetensors(path);
    std::string captured = testing::internal::GetCapturedStderr();
    std::remove(path.c_str());

    // Whether a Model is built depends on what else sits next to the temp file,
    // so that is not asserted here — the shard scan is what this test is about.
    (void)model;
    EXPECT_NE(captured.find("conv5d"), std::string::npos) << captured;
    EXPECT_NE(captured.find("flattening"), std::string::npos)
        << "Expected the reinterpretation to be logged, not silent. Captured: " << captured;
    EXPECT_NE(captured.find("[2, 24]"), std::string::npos)
        << "Expected the flattened shape in the log. Captured: " << captured;
    // The old behaviour must be gone: nothing dropped for dimensionality.
    EXPECT_EQ(captured.find("ndim exceeds"), std::string::npos)
        << "High-dim tensors must no longer be dropped. Captured: " << captured;
}

}  // namespace
}  // namespace imp
