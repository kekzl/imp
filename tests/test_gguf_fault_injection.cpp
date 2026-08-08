// GGUF loader fault-injection tests — TEST_AUDIT.md Phase 2, risk #10.
//
// Charter (risk #10): "header fault injection → clean error". The contract
// under test is NOT that a malformed file loads correctly — it is that the
// loader NEVER crashes, hangs, or allocates unbounded memory on adversarial
// input. Internal errors may throw or return nullptr (both are "clean"); UB,
// OOM, and infinite loops are the failures we hunt.
//
// Method: build a minimal VALID GGUF v3 byte buffer in-memory (one F32 tensor,
// minimal llama metadata) that load_gguf() accepts, then inject ONE targeted
// corruption per test and assert the loader rejects it gracefully. Each case
// states, by hand, why the expected outcome is what it is. We compare against
// the real loader, not a mock (audit §4).
//
// Several of these tests are REGRESSION tests for real weaknesses found while
// writing them (length/count overflow, unknown-array-type spin, unbounded
// reserve, unchecked tensor offsets). See the commit message / report.

#include <gtest/gtest.h>
#include "model/gguf_loader.h"
#include "model/model.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

namespace imp {
namespace {

// ---- Minimal GGUF byte-buffer builder ----
//
// Produces a valid single-tensor GGUF v3 file. Field offsets of interest are
// recorded so a test can patch exactly one value without rebuilding the rest.

struct GgufBytes {
    std::vector<uint8_t> buf;

    // Recorded byte offsets into buf (set during build()).
    size_t off_magic = 0;
    size_t off_version = 0;
    size_t off_tensor_count = 0;
    size_t off_kv_count = 0;
    size_t off_tensor_name_len = 0;   // u64 length prefix of the tensor's name
    size_t off_tensor_dim0 = 0;       // u64 ne[0] of the tensor
    size_t off_tensor_type = 0;       // u32 ggml type of the tensor
    size_t off_tensor_data_offset = 0;  // u64 data-section-relative offset
    size_t off_block_count = 0;         // u32 value of llama.block_count
};

class Writer {
public:
    explicit Writer(std::vector<uint8_t>& b) : b_(b) {}
    size_t pos() const { return b_.size(); }
    void u32(uint32_t v) { raw(&v, 4); }
    void i32(int32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void f32(float v) { raw(&v, 4); }
    void str(const std::string& s) {
        u64(s.size());
        raw(s.data(), s.size());
    }
    void raw(const void* p, size_t n) {
        size_t at = b_.size();
        b_.resize(at + n);
        std::memcpy(b_.data() + at, p, n);
    }
    void pad_to(size_t align) {
        while (b_.size() % align)
            b_.push_back(0);
    }

private:
    std::vector<uint8_t>& b_;
};

// GGUF metadata value types (subset)
constexpr uint32_t T_UINT32 = 4;
constexpr uint32_t T_STRING = 8;
// GGML tensor type
constexpr uint32_t GGML_F32 = 0;

// One F32 tensor [4 x 4] = 16 elements = 64 bytes. Metadata is the minimum a
// llama-family load needs to not get fully rejected on missing arch.
GgufBytes build_valid_gguf() {
    GgufBytes g;
    Writer w(g.buf);

    // --- header ---
    g.off_magic = w.pos();
    w.u32(GGUF_MAGIC);
    g.off_version = w.pos();
    w.u32(3);
    g.off_tensor_count = w.pos();
    w.u64(1);  // one tensor
    g.off_kv_count = w.pos();
    w.u64(2);  // two KV pairs

    // --- metadata KV pairs ---
    // general.architecture = "llama"
    w.str("general.architecture");
    w.u32(T_STRING);
    w.str("llama");
    // llama.block_count = 0  (no layers — keeps the model trivially small)
    w.str("llama.block_count");
    w.u32(T_UINT32);
    g.off_block_count = w.pos();
    w.u32(0);

    // --- tensor info ---
    g.off_tensor_name_len = w.pos();
    w.str("token_embd.weight");
    w.u32(2);  // n_dims
    g.off_tensor_dim0 = w.pos();
    w.u64(4);  // ne[0]
    w.u64(4);  // ne[1]
    g.off_tensor_type = w.pos();
    w.u32(GGML_F32);
    g.off_tensor_data_offset = w.pos();
    w.u64(0);  // data offset (relative to aligned tensor-data start)

    // --- align + tensor data (16 floats) ---
    w.pad_to(GGUF_DEFAULT_ALIGNMENT);
    for (int i = 0; i < 16; i++)
        w.f32(0.01f * static_cast<float>(i));

    return g;
}

std::string write_temp(const std::vector<uint8_t>& data) {
    char path[] = "/tmp/imp_fault_XXXXXX.gguf";
    int fd = mkstemps(path, 5);
    if (fd < 0)
        return "";
    ssize_t n = write(fd, data.data(), data.size());
    (void)n;
    close(fd);
    return std::string(path);
}

// Run load_gguf on a buffer and return whether it crashed-free. Wraps the
// throw→nullptr ambiguity: a clean error is EITHER nullptr OR a thrown
// exception we catch here. The only failure is a crash/hang (which GTest would
// surface as a SIGSEGV/timeout, not a returned value).
bool load_is_clean(const std::vector<uint8_t>& data) {
    std::string path = write_temp(data);
    if (path.empty())
        return false;
    bool clean = true;
    try {
        auto model = load_gguf(path);
        // nullptr or a model are both "clean" — we only assert no crash. A
        // model returned from a corrupt-but-survivable file is acceptable as
        // long as it didn't fault building it.
        (void)model;
    } catch (...) {
        // Throwing is the documented internal-error channel (translated to
        // ImpError at the API boundary). Catching here proves we unwound
        // rather than faulted.
        clean = true;
    }
    unlink(path.c_str());
    return clean;
}

// Load a buffer and return the model (may be nullptr). Used by the bounds
// tests to assert the OBSERVABLE consequence of the fix: a tensor whose data
// window escapes the file is SKIPPED, so token_embedding().data stays null —
// the unfixed loader instead assigns a wild pointer (non-null). This is the
// non-tautological discriminator between fixed and unfixed.
std::unique_ptr<Model> load_buf(const std::vector<uint8_t>& data) {
    std::string path = write_temp(data);
    if (path.empty())
        return nullptr;
    std::unique_ptr<Model> m;
    try {
        m = load_gguf(path);
    } catch (...) {
        m = nullptr;
    }
    unlink(path.c_str());
    return m;
}

void patch_u64(std::vector<uint8_t>& buf, size_t off, uint64_t v) {
    std::memcpy(buf.data() + off, &v, 8);
}
void patch_u32(std::vector<uint8_t>& buf, size_t off, uint32_t v) {
    std::memcpy(buf.data() + off, &v, 4);
}

// ---- Sanity: the baseline buffer actually loads ----

TEST(GgufFaultInjection, ValidBaselineLoads) {
    // If this fails, every corruption test below is vacuous (the input was
    // never valid to begin with). This is the anti-tautology anchor.
    GgufBytes g = build_valid_gguf();
    std::string path = write_temp(g.buf);
    ASSERT_FALSE(path.empty());
    auto model = load_gguf(path);
    EXPECT_NE(model, nullptr) << "baseline GGUF must load, else corruption tests are meaningless";
    unlink(path.c_str());
}

// ---- Config vs tensors ----
//
// CHARACTERISATION, not an invariant. A GGUF whose metadata declares N
// transformer blocks but ships no layer tensor at all loads *successfully*:
// load_gguf returns a Model reporting n_layers == N with every layer weight
// null, and emits no diagnostic about it. That is #1312.
//
// This test pins the behaviour that exists so the change is visible when it is
// fixed — it deliberately does NOT assert the invariant the loader ought to
// enforce ("declared layers must have weights, or the load fails"), because
// this file runs in the CI lane and a red required check blocks every merge.
// The strict version is one edit away and is written out in #1312.
//
// Every config/tensor consistency check in the loader is a WARN
// (gguf_loader.cpp:855, :862, :866) and none covers this case: n_attn is
// counted at :730 and printed at :769, never compared against cfg.n_layers.
TEST(GgufFaultInjection, DeclaredLayersWithoutTensorsLoadWithNullWeights) {
    GgufBytes g = build_valid_gguf();
    patch_u32(g.buf, g.off_block_count, 2);
    auto model = load_buf(g.buf);

    ASSERT_NE(model, nullptr) << "documenting today's behaviour: the load succeeds";
    EXPECT_EQ(model->config().n_layers, 2);
    ASSERT_EQ(model->layers_.size(), 2u);
    for (size_t i = 0; i < model->layers_.size(); ++i) {
        EXPECT_EQ(model->layers_[i].wq.data, nullptr)
            << "layer " << i << ": if this is now non-null the loader changed — see #1312";
        EXPECT_EQ(model->layers_[i].wk.data, nullptr) << "layer " << i;
        EXPECT_EQ(model->layers_[i].w_down.data, nullptr) << "layer " << i;
    }
    // The one thing the file did supply is present, so the null layers above
    // are a missing-tensor consequence and not a wholesale load failure.
    EXPECT_NE(model->token_embedding().data, nullptr);
}

// ---- Magic / version ----

TEST(GgufFaultInjection, BadMagic) {
    // Wrong magic → load_gguf checks magic first and bails. Expect nullptr:
    // there is no recovery from a non-GGUF file.
    GgufBytes g = build_valid_gguf();
    patch_u32(g.buf, g.off_magic, 0xDEADBEEF);
    std::string path = write_temp(g.buf);
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(load_gguf(path), nullptr);
    unlink(path.c_str());
}

TEST(GgufFaultInjection, BadVersion) {
    // Only v2/v3 supported. v1 and v999 must be rejected (nullptr), never
    // parsed with v3 field layout assumptions.
    for (uint32_t ver : {0u, 1u, 4u, 999u, 0xFFFFFFFFu}) {
        GgufBytes g = build_valid_gguf();
        patch_u32(g.buf, g.off_version, ver);
        std::string path = write_temp(g.buf);
        ASSERT_FALSE(path.empty());
        EXPECT_EQ(load_gguf(path), nullptr) << "version " << ver << " must be rejected";
        unlink(path.c_str());
    }
}

// ---- Truncation ----

TEST(GgufFaultInjection, TruncatedHeader) {
    // File cut to 8 bytes (magic+version present, counts missing). The u64
    // reads for tensor_count/kv_count must fail the EOF check → nullptr.
    GgufBytes g = build_valid_gguf();
    g.buf.resize(8);
    std::string path = write_temp(g.buf);
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(load_gguf(path), nullptr);
    unlink(path.c_str());
}

TEST(GgufFaultInjection, TruncatedMidMetadata) {
    // Cut the file in the middle of the metadata section (after the header but
    // before the second KV pair completes). read_string / read_u32 must hit
    // EOF and the metadata loop must terminate with reader.failed() → nullptr.
    GgufBytes g = build_valid_gguf();
    // off_tensor_name_len marks the end of metadata; cut a few bytes before it.
    ASSERT_GT(g.off_tensor_name_len, 4u);
    g.buf.resize(g.off_tensor_name_len - 4);
    EXPECT_TRUE(load_is_clean(g.buf));
    std::string path = write_temp(g.buf);
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(load_gguf(path), nullptr);
    unlink(path.c_str());
}

TEST(GgufFaultInjection, TruncatedMidTensorData) {
    // Header + metadata + tensor-info intact, but the tensor's data bytes are
    // cut short. The tensor claims 64 bytes; we leave only 8. The bounds check
    // must reject the tensor (offset+size escapes the mapped region) instead
    // of handing weight_upload a pointer that reads off the end of the file.
    GgufBytes g = build_valid_gguf();
    // Keep everything up to ~16 bytes into the data section, drop the rest.
    size_t keep = g.off_tensor_data_offset + 8 + 16;  // through offset field + a little data
    if (keep < g.buf.size())
        g.buf.resize(keep);
    EXPECT_TRUE(load_is_clean(g.buf));
}

// ---- Absurd counts (must not allocate-to-OOM) ----

TEST(GgufFaultInjection, HugeKvCount) {
    // kv_count = 2^60 with a tiny file. Pre-fix, metadata.reserve(2^60) would
    // attempt to allocate petabytes → bad_alloc / OOM-kill before any read.
    // Post-fix the reserve is clamped to remaining()/12 and the first KV read
    // fails on EOF → clean nullptr.
    GgufBytes g = build_valid_gguf();
    patch_u64(g.buf, g.off_kv_count, (uint64_t{1} << 60));
    EXPECT_TRUE(load_is_clean(g.buf));
    std::string path = write_temp(g.buf);
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(load_gguf(path), nullptr);
    unlink(path.c_str());
}

TEST(GgufFaultInjection, HugeTensorCount) {
    // tensor_count = 2^60. Same class as HugeKvCount but for the tensor-info
    // reserve. Must not OOM; must fail cleanly.
    GgufBytes g = build_valid_gguf();
    patch_u64(g.buf, g.off_tensor_count, (uint64_t{1} << 60));
    EXPECT_TRUE(load_is_clean(g.buf));
    std::string path = write_temp(g.buf);
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(load_gguf(path), nullptr);
    unlink(path.c_str());
}

TEST(GgufFaultInjection, MaxU64TensorCount) {
    // tensor_count = UINT64_MAX. Stresses the reserve clamp and the parse loop
    // bound simultaneously; the loop must stop on the first failed read.
    GgufBytes g = build_valid_gguf();
    patch_u64(g.buf, g.off_tensor_count, UINT64_MAX);
    EXPECT_TRUE(load_is_clean(g.buf));
}

// ---- String length past EOF / overflow ----

TEST(GgufFaultInjection, StringLengthPastEof) {
    // The tensor-name length prefix is set to a value larger than the file.
    // read_string's bounds check must reject it (set failed_) rather than
    // construct a std::string spanning past the mmap.
    GgufBytes g = build_valid_gguf();
    patch_u64(g.buf, g.off_tensor_name_len, 1'000'000'000ULL);
    EXPECT_TRUE(load_is_clean(g.buf));
    std::string path = write_temp(g.buf);
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(load_gguf(path), nullptr);
    unlink(path.c_str());
}

TEST(GgufFaultInjection, StringLengthOverflow) {
    // Length = UINT64_MAX. Pre-fix, `pos_ + len <= size_` wrapped around and
    // admitted the read → a ~2^64-byte std::string construction → crash. The
    // overflow-safe check (`len <= remaining()`) must reject it.
    GgufBytes g = build_valid_gguf();
    patch_u64(g.buf, g.off_tensor_name_len, UINT64_MAX);
    EXPECT_TRUE(load_is_clean(g.buf));
}

// ---- Unknown array element type (infinite-loop guard) ----

TEST(GgufFaultInjection, UnknownArrayElementTypeHugeCount) {
    // Build a file whose single KV value is an ARRAY whose element type is an
    // unknown/unsupported enum (99) with count = 2^60. read_gguf_value's switch
    // has no default → it consumes 0 bytes per element. Pre-fix the loop would
    // spin 2^60 times (~forever). Post-fix the unknown element type fails the
    // reader immediately. We assert it returns within a generous time bound.
    constexpr uint32_t T_ARRAY = 9;
    GgufBytes g;
    Writer w(g.buf);
    w.u32(GGUF_MAGIC);
    w.u32(3);
    w.u64(0);  // tensor_count
    w.u64(1);  // kv_count
    // one KV: key="bad", value = ARRAY<unknown=99>[2^60]
    w.str("bad");
    w.u32(T_ARRAY);
    w.u32(99);                    // element type — not a real GGUFValueType
    w.u64(uint64_t{1} << 60);     // count

    auto start = std::chrono::steady_clock::now();
    std::string path = write_temp(g.buf);
    ASSERT_FALSE(path.empty());
    auto model = load_gguf(path);
    auto elapsed = std::chrono::steady_clock::now() - start;
    unlink(path.c_str());
    (void)model;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 5)
        << "unknown array element type with huge count must not spin";
}

// ---- Tensor offset / dim corruption ----

TEST(GgufFaultInjection, TensorOffsetPastEof) {
    // Set token_embd.weight's data offset far beyond the file. data_base+offset
    // would point past the mmap. The bounds check must SKIP the tensor — the
    // unfixed loader instead hands weight_upload a wild pointer (an OOB read /
    // GPU fault on a real model). Observable here: the embedding stays unset.
    GgufBytes g = build_valid_gguf();
    patch_u64(g.buf, g.off_tensor_data_offset, 1'000'000'000ULL);
    auto m = load_buf(g.buf);
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->token_embedding().data, nullptr) << "out-of-bounds tensor must be skipped, not assigned";
}

TEST(GgufFaultInjection, TensorOffsetMaxU64) {
    // Offset = UINT64_MAX: data_base + offset overflows the pointer. The bounds
    // check (offset > data_limit) rejects it before the addition is ever used.
    GgufBytes g = build_valid_gguf();
    patch_u64(g.buf, g.off_tensor_data_offset, UINT64_MAX);
    auto m = load_buf(g.buf);
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->token_embedding().data, nullptr);
}

TEST(GgufFaultInjection, TensorDimOverflow) {
    // ne[0] = 2^40 and ne[1] = 2^40 → element count overflows int64 and the
    // byte size overflows size_t. The saturating size computation must return
    // SIZE_MAX and the tensor must be rejected (not wrap to a small "valid"
    // size that then passes the bounds check and yields a wild pointer).
    GgufBytes g = build_valid_gguf();
    patch_u64(g.buf, g.off_tensor_dim0, uint64_t{1} << 40);
    patch_u64(g.buf, g.off_tensor_dim0 + 8, uint64_t{1} << 40);
    auto m = load_buf(g.buf);
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->token_embedding().data, nullptr) << "overflowing-size tensor must be skipped";
}

TEST(GgufFaultInjection, TensorDimNegative) {
    // A dim with the high bit set reads back as a negative int64. The size
    // computation must treat it as invalid (reject), not compute a bogus span.
    GgufBytes g = build_valid_gguf();
    patch_u64(g.buf, g.off_tensor_dim0, uint64_t{0x8000000000000000ULL});
    auto m = load_buf(g.buf);
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->token_embedding().data, nullptr);
}

TEST(GgufFaultInjection, NonexistentTensorType) {
    // ggml type id = 9999 (no such quant). gguf_blck_size/gguf_type_size return
    // 0 for unknown types → byte-size computation returns SIZE_MAX → tensor
    // rejected (also guards against a divide-by-zero on block size 0).
    GgufBytes g = build_valid_gguf();
    patch_u32(g.buf, g.off_tensor_type, 9999);
    auto m = load_buf(g.buf);
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->token_embedding().data, nullptr) << "unknown-type tensor must be skipped";
}

// ---- Alignment abuse ----

TEST(GgufFaultInjection, ZeroAlignmentMetadata) {
    // general.alignment = 0 would make `pos_ % 0` a divide-by-zero in the
    // reader's align(). The loader guards alignment==0 → falls back to the
    // default. We inject it by appending a KV; simplest is to rebuild with an
    // alignment KV. Here we assert the guard via a from-scratch buffer.
    constexpr uint32_t T_U32 = 4;
    GgufBytes g;
    Writer w(g.buf);
    w.u32(GGUF_MAGIC);
    w.u32(3);
    w.u64(0);  // tensor_count
    w.u64(2);  // kv_count
    w.str("general.architecture");
    w.u32(T_STRING);
    w.str("llama");
    w.str("general.alignment");
    w.u32(T_U32);
    w.u32(0);  // zero alignment — must not crash the align() modulo
    EXPECT_TRUE(load_is_clean(g.buf));
}

}  // namespace
}  // namespace imp
