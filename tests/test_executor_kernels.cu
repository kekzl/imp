#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "exec/executor_kernels.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>
#include <cstring>
#include <numeric>

namespace imp {
namespace {

// ── GPU tensor helpers (same pattern as test_activation.cu) ────────────────

Tensor make_gpu_fp16(const float* host_data, std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.qtype = QType::F16;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list)
        t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    std::vector<half> h(t.numel());
    for (int64_t j = 0; j < t.numel(); j++)
        h[j] = __float2half(host_data[j]);
    cudaMalloc(&t.data, t.nbytes());
    cudaMemcpy(t.data, h.data(), t.nbytes(), cudaMemcpyHostToDevice);
    return t;
}

Tensor make_gpu_fp32(const float* host_data, std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.qtype = QType::F32;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list)
        t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemcpy(t.data, host_data, t.nbytes(), cudaMemcpyHostToDevice);
    return t;
}

Tensor alloc_gpu(QType dtype, std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.qtype = dtype;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list)
        t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemset(t.data, 0, t.nbytes());
    return t;
}

std::vector<float> read_fp16(const Tensor& t) {
    std::vector<half> h(t.numel());
    cudaMemcpy(h.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
    std::vector<float> result(t.numel());
    for (int64_t j = 0; j < t.numel(); j++)
        result[j] = __half2float(h[j]);
    return result;
}

std::vector<float> read_fp32(const Tensor& t) {
    std::vector<float> result(t.numel());
    cudaMemcpy(result.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
    return result;
}

void free_tensor(Tensor& t) {
    if (t.data) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

// =========================================================================
// elementwise_add (FP16): a[i] += b[i]
// =========================================================================

TEST(ExecutorKernelsTest, ElementwiseAddFP16) {
    const int N = 1024;
    std::vector<float> ha(N), hb(N);
    for (int i = 0; i < N; i++) {
        ha[i] = static_cast<float>(i) * 0.01f;
        hb[i] = static_cast<float>(N - i) * 0.01f;
    }

    Tensor a = make_gpu_fp16(ha.data(), {N});
    Tensor b = make_gpu_fp16(hb.data(), {N});

    elementwise_add(a, b, nullptr);
    cudaDeviceSynchronize();

    auto result = read_fp16(a);
    for (int i = 0; i < N; i++) {
        float expected = ha[i] + hb[i];
        EXPECT_NEAR(result[i], expected, 0.05f) << "Mismatch at index " << i;
    }

    free_tensor(a);
    free_tensor(b);
}

// =========================================================================
// elementwise_add (FP32): a[i] += b[i]
// =========================================================================

TEST(ExecutorKernelsTest, ElementwiseAddFP32) {
    const int N = 1024;
    std::vector<float> ha(N), hb(N);
    for (int i = 0; i < N; i++) {
        ha[i] = static_cast<float>(i) * 0.001f;
        hb[i] = static_cast<float>(N - i) * 0.001f;
    }

    Tensor a = make_gpu_fp32(ha.data(), {N});
    Tensor b = make_gpu_fp32(hb.data(), {N});

    elementwise_add(a, b, nullptr);
    cudaDeviceSynchronize();

    auto result = read_fp32(a);
    for (int i = 0; i < N; i++) {
        float expected = ha[i] + hb[i];
        EXPECT_NEAR(result[i], expected, 1e-6f) << "Mismatch at index " << i;
    }

    free_tensor(a);
    free_tensor(b);
}

// =========================================================================
// elementwise_add_store (FP16): out[i] = a[i] + b[i]
// =========================================================================

TEST(ExecutorKernelsTest, ElementwiseAddStoreFP16) {
    const int N = 512;
    std::vector<float> ha(N), hb(N);
    for (int i = 0; i < N; i++) {
        ha[i] = static_cast<float>(i) * 0.02f - 5.0f;
        hb[i] = static_cast<float>(i) * -0.01f + 3.0f;
    }

    Tensor a = make_gpu_fp16(ha.data(), {N});
    Tensor b = make_gpu_fp16(hb.data(), {N});
    Tensor out = alloc_gpu(QType::F16, {N});

    elementwise_add_store(a, b, out, nullptr);
    cudaDeviceSynchronize();

    auto result = read_fp16(out);
    for (int i = 0; i < N; i++) {
        float expected = ha[i] + hb[i];
        EXPECT_NEAR(result[i], expected, 0.05f) << "Mismatch at index " << i;
    }

    free_tensor(a);
    free_tensor(b);
    free_tensor(out);
}

// =========================================================================
// add_bias (FP16): out[row, col] += bias[col]
// =========================================================================

TEST(ExecutorKernelsTest, AddBiasFP16) {
    const int rows = 4, cols = 128;
    std::vector<float> h_out(rows * cols, 1.0f);
    std::vector<float> h_bias(cols);
    for (int j = 0; j < cols; j++)
        h_bias[j] = static_cast<float>(j) * 0.1f;

    Tensor out = make_gpu_fp16(h_out.data(), {rows, cols});
    Tensor bias = make_gpu_fp16(h_bias.data(), {cols});

    add_bias(out, bias, nullptr);
    cudaDeviceSynchronize();

    auto result = read_fp16(out);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float expected = 1.0f + h_bias[c];
            EXPECT_NEAR(result[r * cols + c], expected, 0.05f) << "Mismatch at (" << r << ", " << c << ")";
        }
    }

    free_tensor(out);
    free_tensor(bias);
}

// =========================================================================
// scale_fp16_kernel: data[i] *= scale
// =========================================================================

TEST(ExecutorKernelsTest, ScaleFP16) {
    const int N = 256;
    std::vector<float> h_data(N);
    for (int i = 0; i < N; i++)
        h_data[i] = static_cast<float>(i + 1);

    Tensor data = make_gpu_fp16(h_data.data(), {N});
    half scale = __float2half(0.5f);

    int threads = 256;
    int blocks = (N / 2 + threads - 1) / threads;
    scale_fp16_kernel<<<blocks, threads, 0, nullptr>>>(static_cast<half*>(data.data), scale, N);
    cudaDeviceSynchronize();

    auto result = read_fp16(data);
    for (int i = 0; i < N; i++) {
        float expected = h_data[i] * 0.5f;
        EXPECT_NEAR(result[i], expected, 0.1f) << "Mismatch at index " << i;
    }

    free_tensor(data);
}

// =========================================================================
// fp16_to_fp32_kernel + fp32_to_fp16_kernel: roundtrip
// =========================================================================

TEST(ExecutorKernelsTest, FP16ToFP32Roundtrip) {
    const int N = 512;
    std::vector<float> h_data(N);
    for (int i = 0; i < N; i++)
        h_data[i] = static_cast<float>(i) * 0.1f - 25.0f;

    Tensor fp16_in = make_gpu_fp16(h_data.data(), {N});
    Tensor fp32_mid = alloc_gpu(QType::F32, {N});
    Tensor fp16_out = alloc_gpu(QType::F16, {N});

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // FP16 → FP32
    fp16_to_fp32_kernel<<<blocks, threads, 0, nullptr>>>(static_cast<const half*>(fp16_in.data),
                                                         static_cast<float*>(fp32_mid.data), N);

    // FP32 → FP16
    fp32_to_fp16_kernel<<<blocks, threads, 0, nullptr>>>(static_cast<const float*>(fp32_mid.data),
                                                         static_cast<half*>(fp16_out.data), N);
    cudaDeviceSynchronize();

    auto result = read_fp16(fp16_out);
    auto original = read_fp16(fp16_in);
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(result[i], original[i]) << "Roundtrip mismatch at index " << i;
    }

    free_tensor(fp16_in);
    free_tensor(fp32_mid);
    free_tensor(fp16_out);
}

// =========================================================================
// fp32_accum_add_fp16_kernel: accum[i] += half2float(branch[i])
// =========================================================================

TEST(ExecutorKernelsTest, FP32AccumAddFP16) {
    const int N = 256;
    std::vector<float> h_accum(N, 10.0f);
    std::vector<float> h_branch(N);
    for (int i = 0; i < N; i++)
        h_branch[i] = static_cast<float>(i) * 0.1f;

    Tensor accum = make_gpu_fp32(h_accum.data(), {N});
    Tensor branch = make_gpu_fp16(h_branch.data(), {N});

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fp32_accum_add_fp16_kernel<<<blocks, threads, 0, nullptr>>>(static_cast<float*>(accum.data),
                                                                static_cast<const half*>(branch.data), N);
    cudaDeviceSynchronize();

    auto result = read_fp32(accum);
    for (int i = 0; i < N; i++) {
        // branch values lose precision in FP16, so use FP16-rounded value
        float branch_fp16 = __half2float(__float2half(h_branch[i]));
        float expected = h_accum[i] + branch_fp16;
        EXPECT_NEAR(result[i], expected, 1e-5f) << "Mismatch at index " << i;
    }

    free_tensor(accum);
    free_tensor(branch);
}

// =========================================================================
// Edge case: odd-length elementwise_add (tests the scalar tail path)
// =========================================================================

TEST(ExecutorKernelsTest, ElementwiseAddOddLength) {
    const int N = 127;  // odd — tests the scalar tail in the half2 path
    std::vector<float> ha(N, 1.0f), hb(N, 2.0f);

    Tensor a = make_gpu_fp16(ha.data(), {N});
    Tensor b = make_gpu_fp16(hb.data(), {N});

    elementwise_add(a, b, nullptr);
    cudaDeviceSynchronize();

    auto result = read_fp16(a);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(result[i], 3.0f, 0.01f) << "Mismatch at index " << i;
    }

    free_tensor(a);
    free_tensor(b);
}

// =========================================================================
// slice_rows: view of first n rows
// =========================================================================

TEST(ExecutorKernelsTest, SliceRows) {
    Tensor buf = alloc_gpu(QType::F16, {8, 128});
    Tensor sliced = slice_rows(buf, 3);

    EXPECT_EQ(sliced.shape[0], 3);
    EXPECT_EQ(sliced.shape[1], 128);
    EXPECT_EQ(sliced.data, buf.data);  // same base pointer (view)

    // Full slice returns the same tensor
    Tensor full = slice_rows(buf, 8);
    EXPECT_EQ(full.shape[0], 8);

    free_tensor(buf);
}

// =========================================================================
// add_bias_3way: 3 biases in 1 kernel launch
// =========================================================================

TEST(ExecutorKernelsTest, AddBias3Way) {
    const int rows = 2, cols = 64;
    std::vector<float> h_a(rows * cols, 1.0f), h_b(rows * cols, 2.0f), h_c(rows * cols, 3.0f);
    std::vector<float> h_ba(cols), h_bb(cols), h_bc(cols);
    for (int j = 0; j < cols; j++) {
        h_ba[j] = 0.1f;
        h_bb[j] = 0.2f;
        h_bc[j] = 0.3f;
    }

    Tensor a = make_gpu_fp16(h_a.data(), {rows, cols});
    Tensor b = make_gpu_fp16(h_b.data(), {rows, cols});
    Tensor c = make_gpu_fp16(h_c.data(), {rows, cols});
    Tensor ba = make_gpu_fp16(h_ba.data(), {cols});
    Tensor bb = make_gpu_fp16(h_bb.data(), {cols});
    Tensor bc = make_gpu_fp16(h_bc.data(), {cols});

    add_bias_3way(a, ba, b, bb, c, bc, nullptr);
    cudaDeviceSynchronize();

    auto ra = read_fp16(a);
    auto rb = read_fp16(b);
    auto rc = read_fp16(c);
    for (int i = 0; i < rows * cols; i++) {
        EXPECT_NEAR(ra[i], 1.1f, 0.02f) << "a mismatch at " << i;
        EXPECT_NEAR(rb[i], 2.2f, 0.02f) << "b mismatch at " << i;
        EXPECT_NEAR(rc[i], 3.3f, 0.02f) << "c mismatch at " << i;
    }

    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    free_tensor(ba);
    free_tensor(bb);
    free_tensor(bc);
}

// =========================================================================
// residual_add_rmsnorm: fused residual + norm
// =========================================================================

TEST(ExecutorKernelsTest, ResidualAddRMSNorm) {
    const int d = 128;
    std::vector<float> h_hidden(d, 1.0f), h_residual(d, 1.0f), h_weight(d, 1.0f);

    Tensor hidden = make_gpu_fp16(h_hidden.data(), {1, d});
    Tensor residual = make_gpu_fp16(h_residual.data(), {1, d});
    Tensor weight = make_gpu_fp16(h_weight.data(), {d});
    Tensor output = alloc_gpu(QType::F16, {1, d});

    residual_add_rmsnorm(hidden, residual, weight, output, 1e-5f, nullptr);
    cudaDeviceSynchronize();

    // After: hidden = 1.0 + 1.0 = 2.0 for all elements
    // RMSNorm(2.0, 2.0, ...) with weight=1.0:
    //   rms = sqrt(mean(4.0)) = 2.0
    //   output = 2.0 / 2.0 * 1.0 = 1.0
    auto result = read_fp16(output);
    for (int i = 0; i < d; i++) {
        EXPECT_NEAR(result[i], 1.0f, 0.01f) << "norm mismatch at " << i;
    }

    // Verify hidden was modified in-place (should be 2.0)
    auto h_check = read_fp16(hidden);
    for (int i = 0; i < d; i++) {
        EXPECT_NEAR(h_check[i], 2.0f, 0.01f) << "hidden not updated at " << i;
    }

    free_tensor(hidden);
    free_tensor(residual);
    free_tensor(weight);
    free_tensor(output);
}

// =========================================================================
// add_rmsnorm_inplace: h = rmsnorm(a + b, weight)
// =========================================================================

TEST(ExecutorKernelsTest, AddRMSNormInplace) {
    const int d = 128;
    // a = 1.0, b = 1.0, weight = 1.0
    // sum = 2.0, rms = sqrt(4.0) = 2.0, output = 2.0/2.0 * 1.0 = 1.0
    std::vector<float> h_a(d, 1.0f), h_b(d, 1.0f), h_w(d, 1.0f);

    Tensor a = make_gpu_fp16(h_a.data(), {1, d});
    Tensor b = make_gpu_fp16(h_b.data(), {1, d});
    Tensor h = alloc_gpu(QType::F16, {1, d});
    Tensor w = make_gpu_fp16(h_w.data(), {d});

    add_rmsnorm_inplace(a, b, h, w, 1e-5f, nullptr);
    cudaDeviceSynchronize();

    auto result = read_fp16(h);
    for (int i = 0; i < d; i++) {
        EXPECT_NEAR(result[i], 1.0f, 0.01f) << "mismatch at " << i;
    }

    free_tensor(a);
    free_tensor(b);
    free_tensor(h);
    free_tensor(w);
}

// =========================================================================
// rmsnorm_add_residual: output = rmsnorm(input) + residual
// =========================================================================

TEST(ExecutorKernelsTest, RMSNormAddResidual) {
    const int d = 128;
    // input = 2.0, weight = 1.0, residual = 5.0
    // rms = sqrt(4.0) = 2.0, norm = 2.0/2.0 = 1.0, output = 1.0 + 5.0 = 6.0
    std::vector<float> h_in(d, 2.0f), h_w(d, 1.0f), h_r(d, 5.0f);

    Tensor input = make_gpu_fp16(h_in.data(), {1, d});
    Tensor w = make_gpu_fp16(h_w.data(), {d});
    Tensor r = make_gpu_fp16(h_r.data(), {1, d});
    Tensor output = alloc_gpu(QType::F16, {1, d});

    rmsnorm_add_residual(input, w, r, output, 1e-5f, nullptr);
    cudaDeviceSynchronize();

    auto result = read_fp16(output);
    for (int i = 0; i < d; i++) {
        EXPECT_NEAR(result[i], 6.0f, 0.05f) << "mismatch at " << i;
    }

    free_tensor(input);
    free_tensor(w);
    free_tensor(r);
    free_tensor(output);
}

// =========================================================================
// decode_pipeline_advance: chained-step state advance for the pipelined
// batched decode — slot tokens → token_ids, positions/context_lens += 1,
// block-table scatter from mapped pinned patch arrays.
// =========================================================================

TEST(ExecutorKernelsTest, DecodePipelineAdvance) {
    constexpr int n = 3;
    constexpr size_t kSlotStride = 256;  // any multiple of 4 works like SAMPLE_SCRATCH_BYTES
    constexpr int kStride = 8;           // block-table row stride

    // Slot array: token at the first int32 of each slot, garbage after.
    std::vector<char> h_slots(n * kSlotStride, char(0xAB));
    const int32_t slot_tokens[n] = {11, 22, 33};
    for (int i = 0; i < n; i++)
        std::memcpy(h_slots.data() + i * kSlotStride, &slot_tokens[i], sizeof(int32_t));
    char* d_slots = nullptr;
    ASSERT_EQ(cudaMalloc(&d_slots, h_slots.size()), cudaSuccess);
    cudaMemcpy(d_slots, h_slots.data(), h_slots.size(), cudaMemcpyHostToDevice);

    std::vector<int32_t> h_tok(n, -1);
    std::vector<int> h_pos = {4, 15, 31};
    std::vector<int> h_ctx = {5, 16, 32};
    std::vector<int> h_bt(n * kStride, 7);
    int32_t* d_tok = nullptr;
    int *d_pos = nullptr, *d_ctx = nullptr, *d_bt = nullptr;
    ASSERT_EQ(cudaMalloc(&d_tok, n * sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pos, n * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ctx, n * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bt, h_bt.size() * sizeof(int)), cudaSuccess);
    cudaMemcpy(d_tok, h_tok.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h_pos.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx, h_ctx.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, h_bt.data(), h_bt.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Two block-table patches via mapped pinned staging (the engine's setup).
    int *h_off = nullptr, *h_val = nullptr, *d_off = nullptr, *d_val = nullptr;
    ASSERT_EQ(cudaHostAlloc(&h_off, 2 * sizeof(int), cudaHostAllocMapped), cudaSuccess);
    ASSERT_EQ(cudaHostAlloc(&h_val, 2 * sizeof(int), cudaHostAllocMapped), cudaSuccess);
    ASSERT_EQ(cudaHostGetDevicePointer(&d_off, h_off, 0), cudaSuccess);
    ASSERT_EQ(cudaHostGetDevicePointer(&d_val, h_val, 0), cudaSuccess);
    h_off[0] = 1 * kStride + 1;  // row 1, table index 1
    h_val[0] = 42;
    h_off[1] = 2 * kStride + 2;  // row 2, table index 2
    h_val[1] = 99;

    // Per-row history append (penalty rows): positions via mapped pinned.
    constexpr int kHistStride = 32;
    std::vector<int32_t> h_hist(n * kHistStride, -1);
    int32_t* d_hist = nullptr;
    ASSERT_EQ(cudaMalloc(&d_hist, h_hist.size() * sizeof(int32_t)), cudaSuccess);
    cudaMemcpy(d_hist, h_hist.data(), h_hist.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    int *h_hp = nullptr, *d_hp = nullptr;
    ASSERT_EQ(cudaHostAlloc(&h_hp, n * sizeof(int), cudaHostAllocMapped), cudaSuccess);
    ASSERT_EQ(cudaHostGetDevicePointer(&d_hp, h_hp, 0), cudaSuccess);
    h_hp[0] = 0;
    h_hp[1] = 5;
    h_hp[2] = 31;

    decode_pipeline_advance(n, reinterpret_cast<const int32_t*>(d_slots), kSlotStride, d_tok, d_pos,
                            d_ctx, d_bt, /*n_patches=*/2, d_off, d_val, d_hist, kHistStride, d_hp,
                            /*stream=*/nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cudaMemcpy(h_tok.data(), d_tok, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos.data(), d_pos, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ctx.data(), d_ctx, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bt.data(), d_bt, h_bt.size() * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        EXPECT_EQ(h_tok[i], slot_tokens[i]) << "row " << i;
    }
    EXPECT_EQ(h_pos[0], 5);
    EXPECT_EQ(h_pos[1], 16);
    EXPECT_EQ(h_pos[2], 32);
    EXPECT_EQ(h_ctx[0], 6);
    EXPECT_EQ(h_ctx[1], 17);
    EXPECT_EQ(h_ctx[2], 33);
    EXPECT_EQ(h_bt[1 * kStride + 1], 42);
    EXPECT_EQ(h_bt[2 * kStride + 2], 99);
    // Everything else untouched.
    int untouched = 0;
    for (size_t i = 0; i < h_bt.size(); i++)
        if (h_bt[i] == 7) untouched++;
    EXPECT_EQ(untouched, static_cast<int>(h_bt.size()) - 2);

    // History append landed at each row's position, everything else intact.
    cudaMemcpy(h_hist.data(), d_hist, h_hist.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(h_hist[0 * kHistStride + 0], 11);
    EXPECT_EQ(h_hist[1 * kHistStride + 5], 22);
    EXPECT_EQ(h_hist[2 * kHistStride + 31], 33);
    int hist_untouched = 0;
    for (size_t i = 0; i < h_hist.size(); i++)
        if (h_hist[i] == -1) hist_untouched++;
    EXPECT_EQ(hist_untouched, static_cast<int>(h_hist.size()) - 3);

    cudaFree(d_slots);
    cudaFree(d_tok);
    cudaFree(d_pos);
    cudaFree(d_ctx);
    cudaFree(d_bt);
    cudaFree(d_hist);
    cudaFreeHost(h_off);
    cudaFreeHost(h_val);
    cudaFreeHost(h_hp);
}

}  // anonymous namespace
}  // namespace imp
