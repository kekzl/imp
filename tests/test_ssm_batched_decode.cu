// Batched Mamba2 decode (runtime.gdn_batched_decode on Nemotron-H): N sequences, one token
// each, slot table in device memory. Asserted bitwise per row against the single-sequence
// kernels (ssm_conv1d_decode, ssm_scan_decode) on the same slot, with a permuted slot table and
// an untouched spare slot, so a row reading a neighbour's slot or row fails.

#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/ssm.h"
#include "core/tensor.h"
#include "test_cuda_skip.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace imp {
namespace {

template <typename T>
T* to_device(const std::vector<T>& h) {
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}

template <typename T>
std::vector<T> to_host(const void* d, size_t n) {
    std::vector<T> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}

std::vector<half> halfs(size_t n, float a, float b) {
    std::vector<half> h(n);
    for (size_t i = 0; i < n; ++i)
        h[i] = __float2half(std::sin((a * static_cast<float>(i)) + b));
    return h;
}

Tensor view(void* p, QType t, std::initializer_list<int64_t> shape) {
    int64_t s[4] = {};
    int i = 0;
    for (auto v : shape)
        s[i++] = v;
    return Tensor(p, t, static_cast<int>(shape.size()), s, true);
}

constexpr int kSeq = 3;
constexpr int kSlots = 5;                 // slot 0 and slot 2 stay unused: their bytes must not move
constexpr int kSlotOf[kSeq] = {4, 1, 3};  // permuted, non-contiguous

TEST(SsmDecodeBatchedKernelTest, ConvRowsMatchSingleSequenceOnTheirSlot) {
    SKIP_IF_NO_CUDA();
    constexpr int ch = 200, k = 4;
    constexpr int64_t stride = static_cast<int64_t>(ch) * k + 64;  // floats per slot, padded
    std::vector<float> pool(static_cast<size_t>(stride) * kSlots);
    for (size_t i = 0; i < pool.size(); ++i)
        pool[i] = std::cos(0.013f * static_cast<float>(i));
    const auto x = halfs(static_cast<size_t>(kSeq) * ch, 0.07f, 0.3f);
    const auto w = halfs(static_cast<size_t>(ch) * k, 0.11f, 1.0f);
    const auto b = halfs(ch, 0.05f, 2.0f);
    float* d_pool = to_device(pool);
    half* d_x = to_device(x);
    half* d_w = to_device(w);
    half* d_b = to_device(b);
    half* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(half) * kSeq * ch), cudaSuccess);
    int* d_slots = to_device(std::vector<int>(kSlotOf, kSlotOf + kSeq));
    Tensor tw = view(d_w, QType::F16, {ch, k}), tb = view(d_b, QType::F16, {ch});
    Tensor tx = view(d_x, QType::F16, {kSeq, ch}), to = view(d_out, QType::F16, {kSeq, ch});
    ssm_conv1d_decode_batched(d_pool, d_slots, stride, tx, tw, tb, to, kSeq, k, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto got_out = to_host<half>(d_out, static_cast<size_t>(kSeq) * ch);
    const auto got_pool = to_host<float>(d_pool, pool.size());

    // Reference: one single-sequence call per row on a fresh copy of that row's slot.
    for (int s = 0; s < kSeq; ++s) {
        std::vector<float> slot(pool.begin() + (kSlotOf[s] * stride),
                                pool.begin() + ((kSlotOf[s] + 1) * stride));
        float* d_slot = to_device(slot);
        half* d_o1 = nullptr;
        ASSERT_EQ(cudaMalloc(&d_o1, sizeof(half) * ch), cudaSuccess);
        Tensor x1 = view(d_x + (static_cast<size_t>(s) * ch), QType::F16, {1, ch});
        Tensor o1 = view(d_o1, QType::F16, {1, ch});
        ssm_conv1d_decode(d_slot, x1, tw, tb, o1, k, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto want_out = to_host<half>(d_o1, ch);
        const auto want_slot = to_host<float>(d_slot, slot.size());
        EXPECT_EQ(std::memcmp(want_out.data(), got_out.data() + (static_cast<size_t>(s) * ch),
                              sizeof(half) * ch),
                  0)
            << "row " << s << " output differs from the single-sequence conv on slot " << kSlotOf[s];
        EXPECT_EQ(std::memcmp(want_slot.data(), got_pool.data() + (kSlotOf[s] * stride),
                              sizeof(float) * stride),
                  0)
            << "slot " << kSlotOf[s] << " state differs from the single-sequence conv";
        cudaFree(d_slot);
        cudaFree(d_o1);
    }
    for (int unused : {0, 2})
        EXPECT_EQ(std::memcmp(pool.data() + (unused * stride), got_pool.data() + (unused * stride),
                              sizeof(float) * stride),
                  0)
            << "unused slot " << unused << " was written";
    cudaFree(d_pool);
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_slots);
}

void scan_rows_match(QType h_dtype) {
    // Nemotron-3-Nano head geometry (head_dim 64, state 128, 8 groups), fewer heads.
    constexpr int n_heads = 16, hd = 64, ss = 128, n_groups = 8;
    constexpr int inner = n_heads * hd, bc = n_groups * ss;
    const size_t es = h_dtype == QType::F16 ? sizeof(half) : sizeof(float);
    const size_t h_bytes = static_cast<size_t>(n_heads) * ss * hd * es;
    const size_t stride = h_bytes + 256;  // bytes per slot, padded like a slab with conv state
    std::vector<unsigned char> pool(stride * kSlots);
    for (size_t i = 0; i < pool.size() / es; ++i) {
        const float v = 0.5f * std::sin(0.017f * static_cast<float>(i));
        if (h_dtype == QType::F16) {
            const half hv = __float2half(v);
            std::memcpy(pool.data() + (i * es), &hv, es);
        } else {
            std::memcpy(pool.data() + (i * es), &v, es);
        }
    }
    const auto x = halfs(static_cast<size_t>(kSeq) * inner, 0.031f, 0.2f);
    const auto B = halfs(static_cast<size_t>(kSeq) * bc, 0.023f, 0.9f);
    const auto C = halfs(static_cast<size_t>(kSeq) * bc, 0.019f, 1.7f);
    const auto dt = halfs(static_cast<size_t>(kSeq) * n_heads, 0.3f, 0.1f);
    const auto z = halfs(static_cast<size_t>(kSeq) * inner, 0.043f, 2.5f);
    std::vector<float> A(n_heads), D(n_heads), dtb(n_heads);
    for (int h = 0; h < n_heads; ++h) {
        A[h] = -0.2f - (0.05f * static_cast<float>(h));
        D[h] = 0.1f * static_cast<float>(h % 5);
        dtb[h] = 0.03f * static_cast<float>(h % 7);
    }
    unsigned char* d_pool = to_device(pool);
    half* d_x = to_device(x);
    half* d_B = to_device(B);
    half* d_C = to_device(C);
    half* d_dt = to_device(dt);
    half* d_z = to_device(z);
    float* d_A = to_device(A);
    float* d_D = to_device(D);
    float* d_dtb = to_device(dtb);
    half* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_y, sizeof(half) * kSeq * inner), cudaSuccess);
    int* d_slots = to_device(std::vector<int>(kSlotOf, kSlotOf + kSeq));
    Tensor tA = view(d_A, QType::F32, {n_heads}), tD = view(d_D, QType::F32, {n_heads});
    Tensor tdtb = view(d_dtb, QType::F32, {n_heads});
    Tensor tx = view(d_x, QType::F16, {kSeq, inner}), ty = view(d_y, QType::F16, {kSeq, inner});
    Tensor tB = view(d_B, QType::F16, {kSeq, bc}), tC = view(d_C, QType::F16, {kSeq, bc});
    Tensor tdt = view(d_dt, QType::F16, {kSeq, n_heads});
    ssm_scan_decode_batched(tx, tB, tC, tdt, tA, tD, tdtb, d_pool, d_slots, stride, ty, d_z, kSeq, n_heads,
                            hd, ss, n_groups, h_dtype, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto got_y = to_host<half>(d_y, static_cast<size_t>(kSeq) * inner);
    const auto got_pool = to_host<unsigned char>(d_pool, pool.size());

    for (int s = 0; s < kSeq; ++s) {
        std::vector<unsigned char> slot(pool.begin() + static_cast<std::ptrdiff_t>(kSlotOf[s] * stride),
                                        pool.begin() +
                                            static_cast<std::ptrdiff_t>((kSlotOf[s] + 1) * stride));
        unsigned char* d_slot = to_device(slot);
        half* d_y1 = nullptr;
        ASSERT_EQ(cudaMalloc(&d_y1, sizeof(half) * inner), cudaSuccess);
        Tensor x1 = view(d_x + (static_cast<size_t>(s) * inner), QType::F16, {inner});
        Tensor B1 = view(d_B + (static_cast<size_t>(s) * bc), QType::F16, {bc});
        Tensor C1 = view(d_C + (static_cast<size_t>(s) * bc), QType::F16, {bc});
        Tensor dt1 = view(d_dt + (static_cast<size_t>(s) * n_heads), QType::F16, {n_heads});
        Tensor y1 = view(d_y1, QType::F16, {inner});
        ssm_scan_decode(x1, B1, C1, dt1, tA, tD, tdtb, d_slot, y1, d_z + (static_cast<size_t>(s) * inner),
                        n_heads, hd, ss, n_groups, h_dtype, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto want_y = to_host<half>(d_y1, inner);
        const auto want_slot = to_host<unsigned char>(d_slot, slot.size());
        EXPECT_EQ(std::memcmp(want_y.data(), got_y.data() + (static_cast<size_t>(s) * inner),
                              sizeof(half) * inner),
                  0)
            << "row " << s << " y differs from the single-sequence scan on slot " << kSlotOf[s];
        EXPECT_EQ(std::memcmp(want_slot.data(), got_pool.data() + (kSlotOf[s] * stride), stride), 0)
            << "slot " << kSlotOf[s] << " h_state differs from the single-sequence scan";
        cudaFree(d_slot);
        cudaFree(d_y1);
    }
    for (int unused : {0, 2})
        EXPECT_EQ(std::memcmp(pool.data() + (unused * stride), got_pool.data() + (unused * stride), stride),
                  0)
            << "unused slot " << unused << " was written";
    for (void* p : {static_cast<void*>(d_pool), static_cast<void*>(d_x), static_cast<void*>(d_B),
                    static_cast<void*>(d_C), static_cast<void*>(d_dt), static_cast<void*>(d_z),
                    static_cast<void*>(d_A), static_cast<void*>(d_D), static_cast<void*>(d_dtb),
                    static_cast<void*>(d_y), static_cast<void*>(d_slots)})
        cudaFree(p);
}

TEST(SsmDecodeBatchedKernelTest, ScanRowsMatchSingleSequenceOnTheirSlotF16State) {
    SKIP_IF_NO_CUDA();
    scan_rows_match(QType::F16);
}

TEST(SsmDecodeBatchedKernelTest, ScanRowsMatchSingleSequenceOnTheirSlotF32State) {
    SKIP_IF_NO_CUDA();
    scan_rows_match(QType::F32);
}

}  // namespace
}  // namespace imp
