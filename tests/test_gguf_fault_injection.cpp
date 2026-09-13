// GGUF loader fault-injection (TEST_AUDIT(retired) risk #10): the contract is NOT that a
// malformed file loads correctly, only that the loader never crashes, hangs, or allocates
// unbounded memory on adversarial input (throw or nullptr are both clean; UB/OOM/infinite
// loops are the hunted failures).
// Method: build a minimal VALID GGUF v3 buffer load_gguf() accepts, inject ONE targeted
// corruption per test, assert graceful rejection against the real loader, not a mock.
// Several cases are regression tests for real weaknesses found while writing them
// (length/count overflow, unknown-array-type spin, unbounded reserve, unchecked tensor
// offsets).

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

// Minimal valid single-tensor GGUF v3 builder; field offsets are recorded so a test can
// patch exactly one value without rebuilding the rest.

struct GgufBytes {
    std::vector<uint8_t> buf;

    // Recorded byte offsets into buf (set during build()).
    size_t off_magic = 0;
    size_t off_version = 0;
    size_t off_tensor_count = 0;
    size_t off_kv_count = 0;
    size_t off_tensor_name_len = 0;   // u64 length prefix of the tensor's name
    size_t off_tensor_n_dims = 0;     // u32 n_dims of the tensor
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
    g.off_tensor_n_dims = w.pos();
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

// load_is_clean wraps the throw-vs-nullptr ambiguity: a clean error is EITHER nullptr or a
// caught exception. The only failure is a crash/hang, which GTest surfaces as SIGSEGV/timeout.
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

// load_buf returns the model (may be nullptr); bounds tests assert the OBSERVABLE
// consequence of the fix: a tensor whose data window escapes the file is SKIPPED so
// token_embedding().data stays null - the unfixed loader instead assigns a wild non-null
// pointer. This is the non-tautological fixed-vs-unfixed discriminator.
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

// A GGUF declaring N transformer blocks but shipping no layer tensor must not load: before
// #1312 it did, reporting n_layers==N with every layer weight null and only an unrelated
// tokenizer warning, so a truncated download was indistinguishable from a complete file.
// The check is deliberately weak per layer (attention/GDN/SSM tensors mix freely across
// architectures, none universally required); it rejects only a block carrying none of them
// and no FFN either, which no architecture could execute.
TEST(GgufFaultInjection, DeclaredLayersWithoutTensorsAreRejected) {
    GgufBytes g = build_valid_gguf();
    patch_u32(g.buf, g.off_block_count, 2);
    EXPECT_EQ(load_buf(g.buf), nullptr)
        << "a GGUF declaring layers it has no tensors for must not load";
}

// block_count=0 has no block to be empty, so the declared-layers check above must not fire
// on it - tightening to "every declared layer needs weights" must not silently reject
// embedding-only models.
TEST(GgufFaultInjection, ZeroDeclaredLayersStillLoads) {
    GgufBytes g = build_valid_gguf();  // block_count = 0
    auto model = load_buf(g.buf);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->config().n_layers, 0);
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
    // Tensor data bytes cut short (claims 64 bytes, file has 8): the bounds check must reject
    // it (offset+size escapes the mapped region) instead of handing weight_upload a pointer that
    // reads off the end of the file.
    GgufBytes g = build_valid_gguf();
    // Keep everything up to ~16 bytes into the data section, drop the rest.
    size_t keep = g.off_tensor_data_offset + 8 + 16;  // through offset field + a little data
    if (keep < g.buf.size())
        g.buf.resize(keep);
    EXPECT_TRUE(load_is_clean(g.buf));
}

// ---- Absurd counts (must not allocate-to-OOM) ----

TEST(GgufFaultInjection, HugeKvCount) {
    // kv_count=2^60 with a tiny file: pre-fix, metadata.reserve(2^60) would attempt to allocate
    // petabytes (bad_alloc/OOM-kill) before any read. Post-fix the reserve clamps to
    // remaining()/12 and the first KV read fails on EOF -> clean nullptr.
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
    // A KV array value with an unknown/unsupported element type (99) and count=2^60:
    // read_gguf_value's switch has no default, so it consumed 0 bytes/element and pre-fix the
    // loop spun 2^60 times (~forever). Post-fix the unknown type fails the reader immediately;
    // asserted to return within a generous time bound.
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
    // token_embd.weight's data offset set far beyond the file (would point past the mmap): the
    // bounds check must SKIP the tensor - the unfixed loader instead hands weight_upload a wild
    // pointer. Observable here: the embedding stays unset.
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
    // ne[0]=2^40, ne[1]=2^40: element count overflows int64 and byte size overflows size_t. The
    // saturating size computation must return SIZE_MAX and reject the tensor, not wrap to a
    // small "valid" size that passes bounds checking and yields a wild pointer.
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
    // general.alignment=0 would make pos_ % 0 a divide-by-zero in the reader's align(); the
    // loader must guard alignment==0 and fall back to the default.
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

// ---- AUDIT_arch_2026 F1: the header fields that had no ceiling ----

constexpr uint32_t T_ARRAY_TYPE = 9;

// n_dims = 5 on a 4-slot dims array. The parser used to skip the extra word
// and carry on; the loader then wrote 5 entries into `int64_t shape[4]` (a
// stack write past the array, F1-1). Refused at parse now: the whole file.
TEST(GgufFaultInjection, NDimsAboveFourIsRefused) {
    GgufBytes g = build_valid_gguf();
    patch_u32(g.buf, g.off_tensor_n_dims, 5);
    EXPECT_EQ(load_buf(g.buf), nullptr) << "a tensor with n_dims > 4 must refuse the file";
}

// Wire type 9 = Q8_1 (llama.cpp's activation format): no dequant, kernel, or registry entry
// exists for it, and the 1:1 map to QType::Q8_1 let such a tensor load and reach dispatch
// with no path (AUDIT_arch_2026 G-8). An unknown type is SKIPPED (unknown size); Q8_1 has a
// known size, so it must be REFUSED instead.
TEST(GgufFaultInjection, Q8_1WeightTypeIsRefusedAtParse) {
    GgufBytes g = build_valid_gguf();
    patch_u32(g.buf, g.off_tensor_type, 9);
    EXPECT_EQ(load_buf(g.buf), nullptr) << "a Q8_1 weight tensor must refuse the file";
}

// n_dims=5 with five dim words: the unfixed parser read the first four, skipped the fifth,
// passed the bounds check (16 floats), and the loader's shape loop then wrote shape[4] and
// read dims[4] - one stack slot and one struct field past the end - before the Tensor
// constructor's IMP_CHECK aborted the process. Fixed: refused at parse.
TEST(GgufFaultInjection, FiveDimWordsAreRefusedAtParse) {
    GgufBytes g;
    Writer w(g.buf);
    w.u32(GGUF_MAGIC);
    w.u32(3);
    w.u64(1);
    w.u64(2);
    w.str("general.architecture");
    w.u32(T_STRING);
    w.str("llama");
    w.str("llama.block_count");
    w.u32(T_UINT32);
    w.u32(0);
    w.str("token_embd.weight");
    w.u32(5);
    for (uint64_t d : {4, 4, 1, 1, 1})
        w.u64(d);
    w.u32(GGML_F32);
    w.u64(0);
    w.pad_to(GGUF_DEFAULT_ALIGNMENT);
    for (int i = 0; i < 16; i++)
        w.f32(0.01f * static_cast<float>(i));
    EXPECT_EQ(load_buf(g.buf), nullptr) << "five dim words must refuse the file at parse";
}

// A gemma4 file supplying its own sliding_window_pattern: swa_layers gets the ARRAY's
// length, and the per-layer head_dim loop indexed it with block_count (F1-5). The same
// builder with block_count past the cap exercises the check that used to run 300 lines after
// the per-layer resizes.
GgufBytes build_gemma4_gguf(uint32_t block_count, const std::vector<uint32_t>& pattern) {
    GgufBytes g;
    Writer w(g.buf);
    w.u32(GGUF_MAGIC);
    w.u32(3);
    w.u64(1);  // one tensor
    w.u64(5);  // five KV pairs
    w.str("general.architecture");
    w.u32(T_STRING);
    w.str("gemma4");
    w.str("gemma4.block_count");
    w.u32(T_UINT32);
    g.off_block_count = w.pos();
    w.u32(block_count);
    w.str("gemma4.attention.sliding_window_pattern");
    w.u32(T_ARRAY_TYPE);
    w.u32(T_UINT32);
    w.u64(pattern.size());
    for (auto v : pattern)
        w.u32(v);
    w.str("gemma4.attention.key_length");
    w.u32(T_UINT32);
    w.u32(256);
    w.str("gemma4.attention.key_length_swa");
    w.u32(T_UINT32);
    w.u32(512);
    w.str("token_embd.weight");
    w.u32(2);
    w.u64(4);
    w.u64(4);
    w.u32(GGML_F32);
    w.u64(0);
    w.pad_to(GGUF_DEFAULT_ALIGNMENT);
    for (int i = 0; i < 16; i++)
        w.f32(0.01f * static_cast<float>(i));
    return g;
}

TEST(GgufFaultInjection, Gemma4ShortSlidingWindowPatternIsNotOverread) {
    // 8 layers, a 1-entry pattern: layers 1..7 read swa_layers[1..7] past a
    // 1-element vector before the guard. Clean means no fault; the sanitizer
    // lane is what sees the over-read on the unfixed tree.
    GgufBytes g = build_gemma4_gguf(8, {1});
    EXPECT_TRUE(load_is_clean(g.buf));
}

TEST(GgufFaultInjection, Gemma4BlockCountAboveLimitIsRefusedBeforeSizing) {
    // 2^31-1 layers: the unfixed loader resized head_dim_per_layer to it
    // (8.6 GiB) before validate_declared_dimensions ever ran.
    GgufBytes g = build_gemma4_gguf(2147483647u, {1});
    auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(load_buf(g.buf), nullptr);
    auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 5)
        << "the layer cap must run before anything is sized from block_count";
}

// bos/eos ids land in the embedding gather as a row index with no clamp in
// between (F1-7): outside the vocab is refused at load, inside still loads.
GgufBytes build_tokenizer_gguf(uint32_t bos_id, uint32_t eos_id) {
    GgufBytes g;
    Writer w(g.buf);
    w.u32(GGUF_MAGIC);
    w.u32(3);
    w.u64(1);  // one tensor
    w.u64(5);  // five KV pairs
    w.str("general.architecture");
    w.u32(T_STRING);
    w.str("llama");
    w.str("llama.block_count");
    w.u32(T_UINT32);
    w.u32(0);
    w.str("tokenizer.ggml.tokens");
    w.u32(T_ARRAY_TYPE);
    w.u32(T_STRING);
    w.u64(3);
    w.str("a");
    w.str("b");
    w.str("c");
    w.str("tokenizer.ggml.bos_token_id");
    w.u32(T_UINT32);
    w.u32(bos_id);
    w.str("tokenizer.ggml.eos_token_id");
    w.u32(T_UINT32);
    w.u32(eos_id);
    w.str("token_embd.weight");
    w.u32(2);
    w.u64(4);
    w.u64(4);
    w.u32(GGML_F32);
    w.u64(0);
    w.pad_to(GGUF_DEFAULT_ALIGNMENT);
    for (int i = 0; i < 16; i++)
        w.f32(0.01f * static_cast<float>(i));
    return g;
}

TEST(GgufFaultInjection, SpecialTokenIdOutsideVocabIsRefused) {
    EXPECT_NE(load_buf(build_tokenizer_gguf(1, 2).buf), nullptr) << "in-range ids must still load";
    EXPECT_EQ(load_buf(build_tokenizer_gguf(0x40000000u, 2).buf), nullptr) << "bos past the vocab";
    EXPECT_EQ(load_buf(build_tokenizer_gguf(1, 0xFFFFFFFFu).buf), nullptr) << "eos = -1 as a row index";
}

}  // namespace
}  // namespace imp
