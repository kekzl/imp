// Unit tests for the SafeTensors loader's blob-level validation surface.
// Exercises the test-visible helpers in safetensors_internal:: directly with
// synthetic blob bytes — no Model construction, no GPU.

#include "model/safetensors_loader.h"
#include "core/logging.h"
#include "core/qtype.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
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
    // Bug: `8 + header_size > file_size` with header_size = UINT64_MAX-4 wrapped to 3, which is
    // not greater than any legitimate file size, silently bypassing the check. Overflow-safe
    // check rejects this.
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

// load_safetensors returns nullptr on a synthetic blob with bad tensors (no config.json), but
// the per-shard load must drop both bad tensors with a WARN naming each, plus an end-of-shard
// summary; captured via stderr.
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

// A tensor with more dims than kMaxDims used to be DROPPED with a WARN, silently losing a
// weight (Qwen3-VL's patch embed [1024,3,2,16,16] vanished this way). Now flattened to
// [d0, d1*...*dn]: row-major order is untouched, the only shape the GEMM path can consume.
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

// INFO goes to stdout and WARN/ERROR to stderr (src/core/logging.cpp:71), so
// a test that only captures stderr cannot see "Parsed N tensors" - which is the
// line that says whether the tensor actually survived. Capture both.
static std::string load_and_capture(const std::string& path) {
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    auto model = load_safetensors(path);
    (void)model;
    std::string out = testing::internal::GetCapturedStdout();
    std::string err = testing::internal::GetCapturedStderr();
    return out + err;
}

// #1604: the loader kept two private dtype tables, one for the validated byte count, one for
// the consumer's QType; I16 was 2 vs 4, so a file passing every check was read at twice its
// own size. Fails the moment a row's two widths disagree.
TEST(SafeTensorsDtypeTable, ServableRowsHaveMatchingWidths) {
    const size_t n = safetensors_internal::dtype_table_size();
    ASSERT_GT(n, 0u);
    size_t n_servable = 0;
    for (size_t i = 0; i < n; i++) {
        auto row = safetensors_internal::dtype_table_row(i);
        if (!row.servable)
            continue;
        n_servable++;
        EXPECT_EQ(qtype_elem_bytes(row.qtype), row.wire_bytes)
            << "dtype '" << row.name << "' is served as a QType " << qtype_elem_bytes(row.qtype)
            << " bytes wide but is " << row.wire_bytes
            << " bytes wide on disk. Either map it to an equal-width QType or mark it unservable.";
    }
    EXPECT_GT(n_servable, 0u);
}

TEST(SafeTensorsDtypeTable, WidthMismatchedWireTypesAreRefused) {
    // The five that have no equal-width engine type. Pinning them by name so a
    // future "closest proxy" re-typing has to argue with a test.
    const char* refused[] = {"F64", "I64", "U64", "I16", "U16"};
    for (const char* name : refused) {
        bool found = false;
        for (size_t i = 0; i < safetensors_internal::dtype_table_size(); i++) {
            auto row = safetensors_internal::dtype_table_row(i);
            if (row.name == name) {
                found = true;
                EXPECT_FALSE(row.servable) << name << " must not be re-typed to a different width";
            }
        }
        EXPECT_TRUE(found) << name << " missing from the wire dtype table";
    }
}

TEST(SafeTensorsHostileHeader, I16TensorIsDroppedNotRetyped) {
    // Internally consistent file: 4 elements x 2 bytes = 8 bytes, exactly the
    // declared window. Nothing here is malformed - the defect was that the
    // consumer then read 16 bytes, running off the end of the mapping.
    const std::string header = "{\"w\": {\"dtype\": \"I16\", \"shape\": [4], \"data_offsets\": [0, 8]}}";
    std::string path = write_temp_blob(header, 8);
    ASSERT_FALSE(path.empty());

    std::string captured = load_and_capture(path);
    std::remove(path.c_str());

    EXPECT_NE(captured.find("equal-width"), std::string::npos)
        << "Expected an I16 tensor to be refused, not mapped to a 4-byte QType. Captured: " << captured;
    EXPECT_NE(captured.find("dtype_unsupported=1"), std::string::npos) << captured;
    EXPECT_EQ(captured.find("Parsed 1 tensors"), std::string::npos)
        << "The tensor must not survive into the map. Captured: " << captured;
}

// ---- #1603: no lenient branch for an unknown dtype ----

TEST(SafeTensorsHostileHeader, UnknownDtypeWithUnboundedOffsetStartIsDropped) {
    // Reachable half of #1603: offset_start was never validated on the unknown-dtype branch
    // (validate_tensor_offsets skipped it; the branch's own check looked only at offset_end), so
    // it went straight into mmap_base+tensor_data_offset+offset_start with the pointer 2^63 bytes
    // outside a 16-byte mapping.
    const std::string header =
        "{\"w\": {\"dtype\": \"F8_E8M0\", \"shape\": [4, 4], "
        "\"data_offsets\": [18446744073709551000, 0]}}";
    std::string path = write_temp_blob(header, 64);
    ASSERT_FALSE(path.empty());

    std::string captured = load_and_capture(path);
    std::remove(path.c_str());

    EXPECT_NE(captured.find("dtype_unsupported=1"), std::string::npos) << captured;
    // The old lenient path emitted this text; it must be gone entirely.
    EXPECT_EQ(captured.find("lenient check"), std::string::npos)
        << "The lenient unknown-dtype branch must not exist. Captured: " << captured;
    EXPECT_EQ(captured.find("Parsed 1 tensors"), std::string::npos)
        << "The tensor must not survive into the map. Captured: " << captured;
}

TEST(SafeTensorsHostileHeader, HugeOffsetEndIsRejected) {
    // Not a wrap test: data_offsets is read through JsonValue::as_int(), narrowing a double to
    // int64_t, so anything above 2^63 lands on INT64_MIN and reads back as 2^63 (measured:
    // static_cast<int64_t>(18446744073709551608.0) == INT64_MIN). tensor_data_offset+offset_end
    // can't reach 2^64 from this path; what IS reachable is an offset past the file, refused on
    // every dtype.
    for (const char* dtype : {"F32", "F8_E8M0"}) {
        const std::string header = std::string("{\"w\": {\"dtype\": \"") + dtype +
                                   "\", \"shape\": [4], "
                                   "\"data_offsets\": [0, 9223372036854775807]}}";
        std::string path = write_temp_blob(header, 16);
        ASSERT_FALSE(path.empty());
        std::string captured = load_and_capture(path);
        std::remove(path.c_str());
        EXPECT_EQ(captured.find("Parsed 1 tensors"), std::string::npos)
            << "dtype " << dtype << " must not produce a tensor. Captured: " << captured;
    }
}

TEST(SafeTensorsHostileHeader, NegativeShapeDimIsDropped) {
    const std::string header = "{\"w\": {\"dtype\": \"F32\", \"shape\": [-4], \"data_offsets\": [0, 16]}}";
    std::string path = write_temp_blob(header, 16);
    ASSERT_FALSE(path.empty());

    std::string captured = load_and_capture(path);
    std::remove(path.c_str());

    EXPECT_NE(captured.find("bad_shape=1"), std::string::npos)
        << "A negative dim must be dropped, not multiplied into nelem. Captured: " << captured;
    EXPECT_EQ(captured.find("Parsed 1 tensors"), std::string::npos)
        << "The tensor must not survive into the map. Captured: " << captured;
}

TEST(SafeTensorsHostileHeader, OverflowingShapeProductIsDropped) {
    // 2^32 x 2^32 x 4 bytes wraps int64 to 0, which would make expected_nbytes
    // 0 and let a zero-length window pass the size check for a tensor whose
    // numel() the consumer recomputes as something else entirely.
    const std::string header =
        "{\"w\": {\"dtype\": \"F32\", \"shape\": [4294967296, 4294967296], "
        "\"data_offsets\": [0, 0]}}";
    std::string path = write_temp_blob(header, 16);
    ASSERT_FALSE(path.empty());

    std::string captured = load_and_capture(path);
    std::remove(path.c_str());

    EXPECT_NE(captured.find("bad_shape=1"), std::string::npos)
        << "An overflowing shape product must be dropped. Captured: " << captured;
    EXPECT_EQ(captured.find("Parsed 1 tensors"), std::string::npos)
        << "The tensor must not survive into the map. Captured: " << captured;
}

TEST(SafeTensorsHostileHeader, WellFormedF32TensorStillLoads) {
    // Negative control: the hardening above must not reject a valid tensor.
    // Nothing is dropped, so no drop summary is emitted at all.
    const std::string header = "{\"w\": {\"dtype\": \"F32\", \"shape\": [2, 2], \"data_offsets\": [0, 16]}}";
    std::string path = write_temp_blob(header, 16);
    ASSERT_FALSE(path.empty());

    std::string captured = load_and_capture(path);
    std::remove(path.c_str());

    EXPECT_EQ(captured.find("dropped"), std::string::npos)
        << "A well-formed F32 tensor must survive the hardening. Captured: " << captured;
    EXPECT_NE(captured.find("Parsed 1 tensors"), std::string::npos)
        << "The tensor must reach the map. Captured: " << captured;
}

// AWQ checkpoints (#2196 refusal, #2205 dequant): one llama layer, q_proj AWQ-packed at group 32.
struct AwqTensor {
    std::string name, dtype;
    std::vector<int64_t> shape;
    size_t bytes;
};

std::vector<AwqTensor> awq_q_proj(int64_t K, int64_t N, int64_t groups) {
    return {{"model.embed_tokens.weight", "F16", {8, 32}, 8 * 32 * 2},
            {"model.layers.0.self_attn.q_proj.qweight", "I32", {K, N / 8}, size_t(K * N / 8 * 4)},
            {"model.layers.0.self_attn.q_proj.qzeros", "I32", {groups, N / 8}, size_t(groups * N / 8 * 4)},
            {"model.layers.0.self_attn.q_proj.scales", "F16", {groups, N}, size_t(groups * N * 2)}};
}

std::filesystem::path write_awq_checkpoint(const std::string& tag, const std::string& quant_config,
                                           const std::vector<AwqTensor>& tensors) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("imp_awq_test_" + tag);
    fs::remove_all(root);
    fs::create_directories(root);
    std::string header = "{";
    size_t off = 0;
    for (const auto& t : tensors) {
        std::string shape;
        for (size_t i = 0; i < t.shape.size(); ++i)
            shape += (i ? "," : "") + std::to_string(t.shape[i]);
        header += (off || header.size() > 1 ? "," : "") + std::string("\"") + t.name + "\":{\"dtype\":\"" + t.dtype +
                  "\",\"shape\":[" + shape + "],\"data_offsets\":[" + std::to_string(off) + "," +
                  std::to_string(off + t.bytes) + "]}";
        off += t.bytes;
    }
    header += "}";
    std::ofstream st(root / "model.safetensors", std::ios::binary);
    const uint64_t hdr = header.size();
    st.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    st << header;
    const std::vector<char> data(off, 0);
    st.write(data.data(), static_cast<std::streamsize>(data.size()));
    std::ofstream cfg(root / "config.json");
    cfg << R"({"model_type": "llama", "num_hidden_layers": 1, "hidden_size": 32,
               "num_attention_heads": 1, "vocab_size": 8, "quantization_config": )"
        << quant_config << "}";
    return root;
}

std::pair<std::unique_ptr<Model>, std::string> load_capturing(const std::filesystem::path& root) {
    const LogLevel saved = log_get_level();
    log_set_level(LogLevel::INFO);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    auto model = load_safetensors(root.string());
    std::string log = testing::internal::GetCapturedStderr();
    log += testing::internal::GetCapturedStdout();
    log_set_level(saved);
    return {std::move(model), std::move(log)};
}

// #2196: every AWQ variant without a dequant stays refused, with the detected config in the log.
TEST(SafeTensorsAwq, UnsupportedVariantsAreRefusedWithTheDetectedConfig) {
    const struct {
        const char* qc;
        const char* detected;
    } cases[] = {
        {R"({"quant_method": "awq", "bits": 4, "group_size": 32, "zero_point": true, "version": "gemv"})",
         "bits=4 group_size=32 zero_point=true version=gemv"},
        {R"({"quant_method": "awq", "bits": 3, "group_size": 32, "zero_point": true, "version": "gemm"})",
         "bits=3 group_size=32 zero_point=true version=gemm"},
        {R"({"quant_method": "awq", "bits": 4, "group_size": 32, "zero_point": false, "version": "gemm"})",
         "bits=4 group_size=32 zero_point=false version=gemm"},
        {R"({"quant_method": "awq", "bits": 4, "group_size": 32, "zero_point": true, "version": "marlin"})",
         "bits=4 group_size=32 zero_point=true version=marlin"},
    };
    for (const auto& c : cases) {
        const auto root = write_awq_checkpoint("refuse", c.qc, awq_q_proj(32, 32, 1));
        auto [model, log] = load_capturing(root);
        EXPECT_EQ(model, nullptr) << c.detected << "\n" << log;
        EXPECT_NE(log.find(c.detected), std::string::npos) << log;
        EXPECT_NE(log.find("AWQ variant not supported"), std::string::npos) << log;
        EXPECT_NE(log.find("#2205"), std::string::npos) << log;
        std::filesystem::remove_all(root);
    }
}

// #2205: 4-bit GEMM with zero points loads; q_proj is marked for dequant_awq4 at upload.
TEST(SafeTensorsAwq, Gemm4BitZeroPointLoadsAndMarksTheProjection) {
    for (const char* version : {"\"gemm\"", "\"GEMM\""}) {
        const std::string qc = std::string(R"({"quant_method": "awq", "bits": 4, "group_size": 32, )") +
                               R"("zero_point": true, "version": )" + version + "}";
        const auto root = write_awq_checkpoint("accept", qc, awq_q_proj(64, 32, 2));
        auto [model, log] = load_capturing(root);
        ASSERT_NE(model, nullptr) << log;
        EXPECT_TRUE(model->config().is_awq_prequant);
        const auto& q = model->layers_.at(0).gptq_q;
        EXPECT_TRUE(q.awq_gemm);
        EXPECT_EQ(q.bits, 4);
        EXPECT_EQ(q.group_size, 32);
        EXPECT_FALSE(model->layers_.at(0).gptq_k.awq_gemm);
        EXPECT_NE(log.find("AWQ GEMM 4-bit: 1 projections, group_size=32"), std::string::npos) << log;
        std::filesystem::remove_all(root);
    }
}

// A malformed or unmapped AWQ tensor would dequantize garbage or be dropped: refuse instead.
TEST(SafeTensorsAwq, BadShapeOrUnmappedQweightIsRefused) {
    const std::string qc = R"({"quant_method": "awq", "bits": 4, "group_size": 32, "zero_point": true, "version": "gemm"})";
    auto bad = awq_q_proj(64, 32, 2);
    bad[3].shape = {2, 16};
    bad[3].bytes = 2 * 16 * 2;
    auto root = write_awq_checkpoint("badshape", qc, bad);
    auto [m1, log1] = load_capturing(root);
    EXPECT_EQ(m1, nullptr) << log1;
    EXPECT_NE(log1.find("layer 0: scales is not [K/group_size, N]"), std::string::npos) << log1;
    std::filesystem::remove_all(root);

    auto extra = awq_q_proj(64, 32, 2);
    extra.push_back({"model.layers.0.mlp.experts.0.gate_proj.qweight", "I32", {64, 4}, 64 * 4 * 4});
    root = write_awq_checkpoint("unmapped", qc, extra);
    auto [m2, log2] = load_capturing(root);
    EXPECT_EQ(m2, nullptr) << log2;
    EXPECT_NE(log2.find("2 .qweight tensors, 1 on a q/k/v/o/gate/up/down projection slot"), std::string::npos)
        << log2;
    std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace imp
