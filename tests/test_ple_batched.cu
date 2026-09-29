// Qwen4Exp PLE conv with per-sequence state (compute/ple.h ple_conv_add): a batched decode
// step (one row per sequence, conv rows addressed through a slot table) must equal the same
// rows run one sequence at a time, bit for bit: same taps, same FP order per (row, channel).
// A slot mix-up reads another sequence's past rows and differs in every channel.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/ple.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace imp {
namespace {

constexpr int kChannels = 4 * 64;  // hc * d, reduced
constexpr int kKernel = 4;
constexpr int kDilation = 3;  // ngram_size
constexpr int kStateLen = (kKernel - 1) * kDilation;
constexpr int kSlots = 6;
// Slot stride in halves: state rows plus padding, as the SSM slab tail sits past the GDN layers.
constexpr int64_t kStride = static_cast<int64_t>(kStateLen) * kChannels + 4096;

class PLEBatched : public ::testing::Test {
protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            GTEST_SKIP() << "no CUDA device";
    }
};

std::vector<uint16_t> rand_half(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<uint16_t> v(n);
    for (auto& x : v) {
        const half h = __float2half(d(rng));
        std::memcpy(&x, &h, sizeof(x));
    }
    return v;
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t bytes) { EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess); }
    ~Dev() { cudaFree(p); }
};

Tensor rows_tensor(void* p, int rows, int cols) {
    const int64_t shape[2] = {rows, cols};
    return Tensor(p, QType::F16, 2, shape, true);
}

template <typename T>
void h2d(void* dst, const std::vector<T>& src) {
    ASSERT_EQ(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
}

template <typename T>
std::vector<T> d2h(const void* src, size_t n) {
    std::vector<T> v(n);
    EXPECT_EQ(cudaMemcpy(v.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return v;
}

// Runs n_seq rows either batched (one launch, slot table) or one launch per sequence.
struct PleRun {
    std::vector<uint16_t> hidden, slab;
};

PleRun run_conv(const std::vector<int>& slots, bool batched, uint32_t seed_state) {
    const int n = static_cast<int>(slots.size());
    const size_t row_elems = static_cast<size_t>(n) * kChannels;
    const auto gv = rand_half(row_elems, 11);
    const auto gvn = rand_half(row_elems, 12);
    const auto w = rand_half(static_cast<size_t>(kChannels) * kKernel, 13);
    const auto hidden0 = rand_half(row_elems, 14);
    const auto slab0 = rand_half(static_cast<size_t>(kSlots) * kStride, seed_state);

    Dev d_gv(row_elems * 2), d_gvn(row_elems * 2), d_w(w.size() * 2), d_hidden(row_elems * 2);
    Dev d_slab(slab0.size() * 2), d_slots(slots.size() * sizeof(int));
    h2d(d_gv.p, gv);
    h2d(d_gvn.p, gvn);
    h2d(d_w.p, w);
    h2d(d_hidden.p, hidden0);
    h2d(d_slab.p, slab0);
    h2d(d_slots.p, slots);
    const Tensor tw = rows_tensor(d_w.p, kChannels, kKernel);
    if (batched) {
        const Tensor tgv = rows_tensor(d_gv.p, n, kChannels);
        const Tensor tgvn = rows_tensor(d_gvn.p, n, kChannels);
        Tensor th = rows_tensor(d_hidden.p, n, kChannels);
        ple_conv_add(tgv, tgvn, tw, d_slab.p, kStride, static_cast<const int*>(d_slots.p), n, th, kChannels,
                     kKernel, kDilation, nullptr);
    } else {
        for (int s = 0; s < n; s++) {
            const size_t off = static_cast<size_t>(s) * kChannels * 2;
            const Tensor tgv = rows_tensor(static_cast<char*>(d_gv.p) + off, 1, kChannels);
            const Tensor tgvn = rows_tensor(static_cast<char*>(d_gvn.p) + off, 1, kChannels);
            Tensor th = rows_tensor(static_cast<char*>(d_hidden.p) + off, 1, kChannels);
            void* st = static_cast<uint16_t*>(d_slab.p) +
                       static_cast<size_t>(slots[static_cast<size_t>(s)]) * kStride;
            ple_conv_add(tgv, tgvn, tw, st, 0, nullptr, 1, th, kChannels, kKernel, kDilation, nullptr);
        }
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return {d2h<uint16_t>(d_hidden.p, row_elems), d2h<uint16_t>(d_slab.p, slab0.size())};
}

TEST_F(PLEBatched, BatchedDecodeMatchesPerSequenceBitwise) {
    const std::vector<int> slots = {3, 0, 5, 2};  // not in slot order, slots 1 and 4 idle
    const PleRun a = run_conv(slots, /*batched=*/true, 21);
    const PleRun b = run_conv(slots, /*batched=*/false, 21);
    ASSERT_EQ(a.hidden.size(), b.hidden.size());
    EXPECT_EQ(a.hidden, b.hidden) << "batched hidden rows differ from per-sequence rows";
    EXPECT_EQ(a.slab, b.slab) << "batched conv rows differ from per-sequence conv rows";
    // Idle slots are untouched.
    const auto slab0 = rand_half(static_cast<size_t>(kSlots) * kStride, 21);
    for (int idle : {1, 4}) {
        const size_t off = static_cast<size_t>(idle) * kStride;
        for (size_t i = 0; i < static_cast<size_t>(kStride); i++)
            ASSERT_EQ(a.slab[off + i], slab0[off + i]) << "idle slot " << idle << " written at " << i;
    }
}

TEST_F(PLEBatched, NeighbourStateDoesNotReachARow) {
    // Row 0 (slot 3) must not change when every other slot's past rows change.
    const std::vector<int> slots = {3, 0, 5, 2};
    PleRun a = run_conv(slots, true, 31);
    const auto slab_a = rand_half(static_cast<size_t>(kSlots) * kStride, 31);
    // Same slot-3 rows, different neighbours: rerun with a slab that only shares slot 3.
    const int n = static_cast<int>(slots.size());
    const size_t row_elems = static_cast<size_t>(n) * kChannels;
    auto slab_b = rand_half(static_cast<size_t>(kSlots) * kStride, 32);
    std::memcpy(slab_b.data() + 3 * kStride, slab_a.data() + 3 * kStride, sizeof(uint16_t) * kStride);
    const auto gv = rand_half(row_elems, 11);
    const auto gvn = rand_half(row_elems, 12);
    const auto w = rand_half(static_cast<size_t>(kChannels) * kKernel, 13);
    const auto hidden0 = rand_half(row_elems, 14);
    Dev d_gv(row_elems * 2), d_gvn(row_elems * 2), d_w(w.size() * 2), d_hidden(row_elems * 2);
    Dev d_slab(slab_b.size() * 2), d_slots(slots.size() * sizeof(int));
    h2d(d_gv.p, gv);
    h2d(d_gvn.p, gvn);
    h2d(d_w.p, w);
    h2d(d_hidden.p, hidden0);
    h2d(d_slab.p, slab_b);
    h2d(d_slots.p, slots);
    Tensor th = rows_tensor(d_hidden.p, n, kChannels);
    ple_conv_add(rows_tensor(d_gv.p, n, kChannels), rows_tensor(d_gvn.p, n, kChannels),
                 rows_tensor(d_w.p, kChannels, kKernel), d_slab.p, kStride,
                 static_cast<const int*>(d_slots.p), n, th, kChannels, kKernel, kDilation, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto hidden_b = d2h<uint16_t>(d_hidden.p, row_elems);
    const auto slab_out = d2h<uint16_t>(d_slab.p, slab_b.size());
    for (int c = 0; c < kChannels; c++)
        ASSERT_EQ(hidden_b[static_cast<size_t>(c)], a.hidden[static_cast<size_t>(c)])
            << "row 0 channel " << c;
    for (size_t i = 0; i < static_cast<size_t>(kStride); i++)
        ASSERT_EQ(slab_out[3 * kStride + i], a.slab[3 * kStride + i]) << "slot 3 state at " << i;
    // Control: the neighbours' rows did change, so the comparison above had something to catch.
    bool differs = false;
    for (size_t i = kChannels; i < row_elems && !differs; i++)
        differs = hidden_b[i] != a.hidden[i];
    EXPECT_TRUE(differs);
}

TEST_F(PLEBatched, PrefillChunkMatchesTokenSteps) {
    // One sequence: a 13-row chunk from a zero state equals 13 one-row steps.
    constexpr int kRows = 13;
    const size_t row_elems = static_cast<size_t>(kRows) * kChannels;
    const auto gv = rand_half(row_elems, 41);
    const auto gvn = rand_half(row_elems, 42);
    const auto w = rand_half(static_cast<size_t>(kChannels) * kKernel, 43);
    const auto hidden0 = rand_half(row_elems, 44);
    const size_t st_elems = static_cast<size_t>(kStateLen) * kChannels;
    std::vector<uint16_t> out[2], st[2];
    for (int mode = 0; mode < 2; mode++) {
        Dev d_gv(row_elems * 2), d_gvn(row_elems * 2), d_w(w.size() * 2), d_hidden(row_elems * 2),
            d_st(st_elems * 2);
        h2d(d_gv.p, gv);
        h2d(d_gvn.p, gvn);
        h2d(d_w.p, w);
        h2d(d_hidden.p, hidden0);
        ASSERT_EQ(cudaMemset(d_st.p, 0, st_elems * 2), cudaSuccess);
        const Tensor tw = rows_tensor(d_w.p, kChannels, kKernel);
        const int chunk = mode == 0 ? kRows : 1;
        for (int r0 = 0; r0 < kRows; r0 += chunk) {
            const size_t off = static_cast<size_t>(r0) * kChannels * 2;
            Tensor th = rows_tensor(static_cast<char*>(d_hidden.p) + off, chunk, kChannels);
            ple_conv_add(rows_tensor(static_cast<char*>(d_gv.p) + off, chunk, kChannels),
                         rows_tensor(static_cast<char*>(d_gvn.p) + off, chunk, kChannels), tw, d_st.p, 0,
                         nullptr, 1, th, kChannels, kKernel, kDilation, nullptr);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        out[mode] = d2h<uint16_t>(d_hidden.p, row_elems);
        st[mode] = d2h<uint16_t>(d_st.p, st_elems);
    }
    EXPECT_EQ(out[0], out[1]);
    EXPECT_EQ(st[0], st[1]);
}

}  // namespace
}  // namespace imp
