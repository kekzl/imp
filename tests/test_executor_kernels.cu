#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "graph/executor_kernels.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>
#include <numeric>

namespace imp {
namespace {

// ── GPU tensor helpers (same pattern as test_activation.cu) ────────────────

Tensor make_gpu_fp16(const float* host_data, std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.dtype = DType::FP16;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list) t.shape[i++] = s;
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
    t.dtype = DType::FP32;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list) t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemcpy(t.data, host_data, t.nbytes(), cudaMemcpyHostToDevice);
    return t;
}

Tensor alloc_gpu(DType dtype, std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.dtype = dtype;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list) t.shape[i++] = s;
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
    if (t.data) { cudaFree(t.data); t.data = nullptr; }
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
        EXPECT_NEAR(result[i], expected, 0.05f)
            << "Mismatch at index " << i;
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
        EXPECT_NEAR(result[i], expected, 1e-6f)
            << "Mismatch at index " << i;
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
    Tensor out = alloc_gpu(DType::FP16, {N});

    elementwise_add_store(a, b, out, nullptr);
    cudaDeviceSynchronize();

    auto result = read_fp16(out);
    for (int i = 0; i < N; i++) {
        float expected = ha[i] + hb[i];
        EXPECT_NEAR(result[i], expected, 0.05f)
            << "Mismatch at index " << i;
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
    for (int j = 0; j < cols; j++) h_bias[j] = static_cast<float>(j) * 0.1f;

    Tensor out = make_gpu_fp16(h_out.data(), {rows, cols});
    Tensor bias = make_gpu_fp16(h_bias.data(), {cols});

    add_bias(out, bias, nullptr);
    cudaDeviceSynchronize();

    auto result = read_fp16(out);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float expected = 1.0f + h_bias[c];
            EXPECT_NEAR(result[r * cols + c], expected, 0.05f)
                << "Mismatch at (" << r << ", " << c << ")";
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
    for (int i = 0; i < N; i++) h_data[i] = static_cast<float>(i + 1);

    Tensor data = make_gpu_fp16(h_data.data(), {N});
    half scale = __float2half(0.5f);

    int threads = 256;
    int blocks = (N / 2 + threads - 1) / threads;
    scale_fp16_kernel<<<blocks, threads, 0, nullptr>>>(
        static_cast<half*>(data.data), scale, N);
    cudaDeviceSynchronize();

    auto result = read_fp16(data);
    for (int i = 0; i < N; i++) {
        float expected = h_data[i] * 0.5f;
        EXPECT_NEAR(result[i], expected, 0.1f)
            << "Mismatch at index " << i;
    }

    free_tensor(data);
}

// =========================================================================
// fp16_to_fp32_kernel + fp32_to_fp16_kernel: roundtrip
// =========================================================================

TEST(ExecutorKernelsTest, FP16ToFP32Roundtrip) {
    const int N = 512;
    std::vector<float> h_data(N);
    for (int i = 0; i < N; i++) h_data[i] = static_cast<float>(i) * 0.1f - 25.0f;

    Tensor fp16_in = make_gpu_fp16(h_data.data(), {N});
    Tensor fp32_mid = alloc_gpu(DType::FP32, {N});
    Tensor fp16_out = alloc_gpu(DType::FP16, {N});

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // FP16 → FP32
    fp16_to_fp32_kernel<<<blocks, threads, 0, nullptr>>>(
        static_cast<const half*>(fp16_in.data),
        static_cast<float*>(fp32_mid.data), N);

    // FP32 → FP16
    fp32_to_fp16_kernel<<<blocks, threads, 0, nullptr>>>(
        static_cast<const float*>(fp32_mid.data),
        static_cast<half*>(fp16_out.data), N);
    cudaDeviceSynchronize();

    auto result = read_fp16(fp16_out);
    auto original = read_fp16(fp16_in);
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(result[i], original[i])
            << "Roundtrip mismatch at index " << i;
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
    for (int i = 0; i < N; i++) h_branch[i] = static_cast<float>(i) * 0.1f;

    Tensor accum = make_gpu_fp32(h_accum.data(), {N});
    Tensor branch = make_gpu_fp16(h_branch.data(), {N});

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fp32_accum_add_fp16_kernel<<<blocks, threads, 0, nullptr>>>(
        static_cast<float*>(accum.data),
        static_cast<const half*>(branch.data), N);
    cudaDeviceSynchronize();

    auto result = read_fp32(accum);
    for (int i = 0; i < N; i++) {
        // branch values lose precision in FP16, so use FP16-rounded value
        float branch_fp16 = __half2float(__float2half(h_branch[i]));
        float expected = h_accum[i] + branch_fp16;
        EXPECT_NEAR(result[i], expected, 1e-5f)
            << "Mismatch at index " << i;
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
        EXPECT_NEAR(result[i], 3.0f, 0.01f)
            << "Mismatch at index " << i;
    }

    free_tensor(a);
    free_tensor(b);
}

// =========================================================================
// slice_rows: view of first n rows
// =========================================================================

TEST(ExecutorKernelsTest, SliceRows) {
    Tensor buf = alloc_gpu(DType::FP16, {8, 128});
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

    free_tensor(a); free_tensor(b); free_tensor(c);
    free_tensor(ba); free_tensor(bb); free_tensor(bc);
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
    Tensor output = alloc_gpu(DType::FP16, {1, d});

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

    free_tensor(hidden); free_tensor(residual);
    free_tensor(weight); free_tensor(output);
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
    Tensor h = alloc_gpu(DType::FP16, {1, d});
    Tensor w = make_gpu_fp16(h_w.data(), {d});

    add_rmsnorm_inplace(a, b, h, w, 1e-5f, nullptr);
    cudaDeviceSynchronize();

    auto result = read_fp16(h);
    for (int i = 0; i < d; i++) {
        EXPECT_NEAR(result[i], 1.0f, 0.01f) << "mismatch at " << i;
    }

    free_tensor(a); free_tensor(b); free_tensor(h); free_tensor(w);
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
    Tensor output = alloc_gpu(DType::FP16, {1, d});

    rmsnorm_add_residual(input, w, r, output, 1e-5f, nullptr);
    cudaDeviceSynchronize();

    auto result = read_fp16(output);
    for (int i = 0; i < d; i++) {
        EXPECT_NEAR(result[i], 6.0f, 0.05f) << "mismatch at " << i;
    }

    free_tensor(input); free_tensor(w); free_tensor(r); free_tensor(output);
}

} // anonymous namespace
} // namespace imp
