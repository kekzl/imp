#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/layernorm.h"
#include "quant/nvfp4_quant.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace imp {
namespace {

// ---------------------------------------------------------------------------
// Helper: create a GPU tensor from host float data, with optional FP16 conversion
// ---------------------------------------------------------------------------
Tensor make_gpu_tensor(const float* host_data, QType dtype, std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.qtype = dtype;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list)
        t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());

    if (dtype == QType::F32) {
        cudaMemcpy(t.data, host_data, t.nbytes(), cudaMemcpyHostToDevice);
    } else if (dtype == QType::F16) {
        std::vector<half> h(t.numel());
        for (int64_t j = 0; j < t.numel(); j++)
            h[j] = __float2half(host_data[j]);
        cudaMemcpy(t.data, h.data(), t.nbytes(), cudaMemcpyHostToDevice);
    }
    return t;
}

// ---------------------------------------------------------------------------
// Helper: allocate a zeroed GPU tensor (output buffer)
// ---------------------------------------------------------------------------
Tensor alloc_gpu_tensor(QType dtype, std::initializer_list<int64_t> shape_list) {
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

// ---------------------------------------------------------------------------
// Helper: read a GPU tensor back to host as floats
// ---------------------------------------------------------------------------
std::vector<float> read_gpu_tensor(const Tensor& t) {
    std::vector<float> result(t.numel());
    if (t.qtype == QType::F32) {
        cudaMemcpy(result.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
    } else if (t.qtype == QType::F16) {
        std::vector<half> h(t.numel());
        cudaMemcpy(h.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
        for (int64_t j = 0; j < t.numel(); j++)
            result[j] = __half2float(h[j]);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Helper: free GPU tensor data
// ---------------------------------------------------------------------------
void free_gpu_tensor(Tensor& t) {
    if (t.data) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

// ---------------------------------------------------------------------------
// CPU reference: RMSNorm
//   out[r][c] = (x[r][c] / rms) * weight[c]
//   rms = sqrt( mean(x[r][:]^2) + eps )
// ---------------------------------------------------------------------------
void cpu_rmsnorm(const float* x, const float* weight, float* out, int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        float ss = 0.0f;
        for (int c = 0; c < cols; c++) {
            float v = x[r * cols + c];
            ss += v * v;
        }
        float rms = std::sqrt(ss / static_cast<float>(cols) + eps);
        for (int c = 0; c < cols; c++) {
            out[r * cols + c] = (x[r * cols + c] / rms) * weight[c];
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference: RMSNorm with residual
//   tmp[r][c] = x[r][c] + residual[r][c]
//   out[r][c] = (tmp[r][c] / rms) * weight[c]
//   rms = sqrt( mean(tmp[r][:]^2) + eps )
// ---------------------------------------------------------------------------
void cpu_rmsnorm_residual(const float* x, const float* residual, const float* weight, float* out, int rows,
                          int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        float ss = 0.0f;
        for (int c = 0; c < cols; c++) {
            float v = x[r * cols + c] + residual[r * cols + c];
            ss += v * v;
        }
        float rms = std::sqrt(ss / static_cast<float>(cols) + eps);
        for (int c = 0; c < cols; c++) {
            float v = x[r * cols + c] + residual[r * cols + c];
            out[r * cols + c] = (v / rms) * weight[c];
        }
    }
}

// ===========================================================================
// Test 1: RMSNormBasic -- small FP32 tensor (2 rows x 4 cols)
// ===========================================================================
TEST(LayerNormTest, RMSNormBasic) {
    constexpr int rows = 2;
    constexpr int cols = 4;
    constexpr float eps = 1e-5f;

    // Input data
    std::vector<float> h_x = {
        1.0f, 2.0f,  3.0f, 4.0f,  // row 0
        0.5f, -1.0f, 1.5f, -2.0f  // row 1
    };
    std::vector<float> h_w = {1.0f, 0.5f, 2.0f, 0.1f};

    // CPU reference
    std::vector<float> h_ref(rows * cols);
    cpu_rmsnorm(h_x.data(), h_w.data(), h_ref.data(), rows, cols, eps);

    // GPU computation
    Tensor d_x = make_gpu_tensor(h_x.data(), QType::F32, {rows, cols});
    Tensor d_w = make_gpu_tensor(h_w.data(), QType::F32, {cols});
    Tensor d_out = alloc_gpu_tensor(QType::F32, {rows, cols});

    rmsnorm(d_x, d_w, d_out, eps, nullptr);
    cudaDeviceSynchronize();

    // Read back and compare
    auto h_out = read_gpu_tensor(d_out);
    for (int i = 0; i < rows * cols; i++) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-4f)
            << "Mismatch at index " << i << ": got " << h_out[i] << ", expected " << h_ref[i];
    }

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_w);
    free_gpu_tensor(d_out);
}

// ===========================================================================
// Test 2: RMSNormResidual -- fused residual add + RMSNorm
// ===========================================================================
TEST(LayerNormTest, RMSNormResidual) {
    constexpr int rows = 2;
    constexpr int cols = 4;
    constexpr float eps = 1e-5f;

    std::vector<float> h_x = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, -1.0f, 1.5f, -2.0f};
    std::vector<float> h_residual = {0.1f, -0.2f, 0.3f, -0.4f, 1.0f, 2.0f, 0.0f, 0.5f};
    std::vector<float> h_w = {1.0f, 0.5f, 2.0f, 0.1f};

    // CPU reference
    std::vector<float> h_ref(rows * cols);
    cpu_rmsnorm_residual(h_x.data(), h_residual.data(), h_w.data(), h_ref.data(), rows, cols, eps);

    // GPU computation
    // Note: the kernel modifies x in-place (x <- x + residual), so we use
    // a separate copy for x.
    Tensor d_x = make_gpu_tensor(h_x.data(), QType::F32, {rows, cols});
    Tensor d_res = make_gpu_tensor(h_residual.data(), QType::F32, {rows, cols});
    Tensor d_w = make_gpu_tensor(h_w.data(), QType::F32, {cols});
    Tensor d_out = alloc_gpu_tensor(QType::F32, {rows, cols});

    rmsnorm_residual(d_x, d_res, d_w, d_out, eps, nullptr);
    cudaDeviceSynchronize();

    // Check output
    auto h_out = read_gpu_tensor(d_out);
    for (int i = 0; i < rows * cols; i++) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-4f)
            << "Mismatch at index " << i << ": got " << h_out[i] << ", expected " << h_ref[i];
    }

    // Also verify that x was updated to x + residual
    auto h_x_after = read_gpu_tensor(d_x);
    for (int i = 0; i < rows * cols; i++) {
        float expected_sum = h_x[i] + h_residual[i];
        EXPECT_NEAR(h_x_after[i], expected_sum, 1e-5f) << "x not updated in-place at index " << i;
    }

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_res);
    free_gpu_tensor(d_w);
    free_gpu_tensor(d_out);
}

// ===========================================================================
// Test 3: RMSNormFP16 -- half-precision with relaxed tolerance
// ===========================================================================
TEST(LayerNormTest, RMSNormFP16) {
    constexpr int rows = 3;
    constexpr int cols = 8;
    constexpr float eps = 1e-5f;

    // Build input data (values in a range that FP16 handles well)
    std::vector<float> h_x(rows * cols);
    std::vector<float> h_w(cols);
    for (int i = 0; i < rows * cols; i++)
        h_x[i] = 0.1f * static_cast<float>(i % 7) - 0.3f;  // range [-0.3, 0.3]
    for (int c = 0; c < cols; c++)
        h_w[c] = 0.5f + 0.1f * static_cast<float>(c);

    // CPU reference (computed in float for ground truth)
    std::vector<float> h_ref(rows * cols);
    // Because FP16 has limited precision, the actual half-precision inputs
    // differ slightly from the float originals. We simulate the FP16 roundtrip
    // for the reference to get a fair comparison.
    std::vector<float> h_x_fp16(rows * cols);
    std::vector<float> h_w_fp16(cols);
    for (int i = 0; i < rows * cols; i++)
        h_x_fp16[i] = __half2float(__float2half(h_x[i]));
    for (int c = 0; c < cols; c++)
        h_w_fp16[c] = __half2float(__float2half(h_w[c]));
    cpu_rmsnorm(h_x_fp16.data(), h_w_fp16.data(), h_ref.data(), rows, cols, eps);

    // GPU computation in FP16
    Tensor d_x = make_gpu_tensor(h_x.data(), QType::F16, {rows, cols});
    Tensor d_w = make_gpu_tensor(h_w.data(), QType::F16, {cols});
    Tensor d_out = alloc_gpu_tensor(QType::F16, {rows, cols});

    rmsnorm(d_x, d_w, d_out, eps, nullptr);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int i = 0; i < rows * cols; i++) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-2f)
            << "FP16 mismatch at index " << i << ": got " << h_out[i] << ", expected " << h_ref[i];
    }

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_w);
    free_gpu_tensor(d_out);
}

// ===========================================================================
// Test 4: RMSNormLargeRow -- wider rows (4 x 256) to exercise warp reductions
// ===========================================================================
TEST(LayerNormTest, RMSNormLargeRow) {
    constexpr int rows = 4;
    constexpr int cols = 256;
    constexpr float eps = 1e-5f;

    // Generate deterministic pseudo-random data
    std::vector<float> h_x(rows * cols);
    std::vector<float> h_w(cols);
    for (int i = 0; i < rows * cols; i++) {
        // Simple LCG-like deterministic sequence
        h_x[i] = std::sin(static_cast<float>(i) * 0.1f) * 2.0f;
    }
    for (int c = 0; c < cols; c++) {
        h_w[c] = 0.5f + std::cos(static_cast<float>(c) * 0.05f) * 0.5f;
    }

    // CPU reference
    std::vector<float> h_ref(rows * cols);
    cpu_rmsnorm(h_x.data(), h_w.data(), h_ref.data(), rows, cols, eps);

    // GPU
    Tensor d_x = make_gpu_tensor(h_x.data(), QType::F32, {rows, cols});
    Tensor d_w = make_gpu_tensor(h_w.data(), QType::F32, {cols});
    Tensor d_out = alloc_gpu_tensor(QType::F32, {rows, cols});

    rmsnorm(d_x, d_w, d_out, eps, nullptr);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int i = 0; i < rows * cols; i++) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-4f)
            << "LargeRow mismatch at index " << i << " (row " << (i / cols) << ", col " << (i % cols) << ")"
            << ": got " << h_out[i] << ", expected " << h_ref[i];
    }

    free_gpu_tensor(d_x);
    free_gpu_tensor(d_w);
    free_gpu_tensor(d_out);
}

// ===========================================================================
// Test 5: EpsilonEffect -- different eps values produce different results
//         for near-zero inputs
// ===========================================================================
TEST(LayerNormTest, EpsilonEffect) {
    constexpr int rows = 1;
    constexpr int cols = 4;

    // Near-zero input: sum-of-squares is very small, so eps dominates the rms
    std::vector<float> h_x = {1e-7f, -1e-7f, 1e-7f, -1e-7f};
    std::vector<float> h_w = {1.0f, 1.0f, 1.0f, 1.0f};

    float eps_small = 1e-8f;
    float eps_large = 1e-1f;

    // CPU reference for both epsilon values
    std::vector<float> ref_small(rows * cols), ref_large(rows * cols);
    cpu_rmsnorm(h_x.data(), h_w.data(), ref_small.data(), rows, cols, eps_small);
    cpu_rmsnorm(h_x.data(), h_w.data(), ref_large.data(), rows, cols, eps_large);

    // Sanity check: the two references should differ significantly
    bool refs_differ = false;
    for (int i = 0; i < rows * cols; i++) {
        if (std::fabs(ref_small[i] - ref_large[i]) > 1e-4f) {
            refs_differ = true;
            break;
        }
    }
    ASSERT_TRUE(refs_differ) << "CPU references for different eps values should differ on near-zero input";

    // GPU with small eps
    Tensor d_x_s = make_gpu_tensor(h_x.data(), QType::F32, {rows, cols});
    Tensor d_w_s = make_gpu_tensor(h_w.data(), QType::F32, {cols});
    Tensor d_out_s = alloc_gpu_tensor(QType::F32, {rows, cols});
    rmsnorm(d_x_s, d_w_s, d_out_s, eps_small, nullptr);
    cudaDeviceSynchronize();
    auto h_out_small = read_gpu_tensor(d_out_s);

    // GPU with large eps
    Tensor d_x_l = make_gpu_tensor(h_x.data(), QType::F32, {rows, cols});
    Tensor d_w_l = make_gpu_tensor(h_w.data(), QType::F32, {cols});
    Tensor d_out_l = alloc_gpu_tensor(QType::F32, {rows, cols});
    rmsnorm(d_x_l, d_w_l, d_out_l, eps_large, nullptr);
    cudaDeviceSynchronize();
    auto h_out_large = read_gpu_tensor(d_out_l);

    // Verify each GPU run matches its CPU reference
    for (int i = 0; i < rows * cols; i++) {
        EXPECT_NEAR(h_out_small[i], ref_small[i], 1e-4f) << "Small-eps mismatch at index " << i;
        EXPECT_NEAR(h_out_large[i], ref_large[i], 1e-4f) << "Large-eps mismatch at index " << i;
    }

    // Verify the two GPU outputs differ from each other
    bool gpu_outputs_differ = false;
    for (int i = 0; i < rows * cols; i++) {
        if (std::fabs(h_out_small[i] - h_out_large[i]) > 1e-4f) {
            gpu_outputs_differ = true;
            break;
        }
    }
    EXPECT_TRUE(gpu_outputs_differ) << "GPU outputs with different eps should differ for near-zero input";

    // With large eps and near-zero input, the output magnitude should be small
    // because rms ~ sqrt(eps) is large relative to the input values.
    for (int i = 0; i < rows * cols; i++) {
        EXPECT_LT(std::fabs(h_out_large[i]), 1e-3f)
            << "Large-eps output should be small for near-zero input at index " << i;
    }

    free_gpu_tensor(d_x_s);
    free_gpu_tensor(d_w_s);
    free_gpu_tensor(d_out_s);
    free_gpu_tensor(d_x_l);
    free_gpu_tensor(d_w_l);
    free_gpu_tensor(d_out_l);
}

// ===========================================================================
// RMSNormRowBlockDecodeShapes -- the batched-decode row-block kernel
// (2 <= rows <= 64): real decode shapes incl. the kVecs=2 instantiation
// (d_vec > 512) and both edges of the row gate.
// ===========================================================================
TEST(LayerNormTest, RMSNormRowBlockDecodeShapes) {
    constexpr float eps = 1e-5f;
    const int shapes[][2] = {{2, 2048}, {32, 5120}, {64, 8192}};
    for (auto& sh : shapes) {
        const int rows = sh[0], cols = sh[1];
        std::vector<float> h_x((size_t)rows * cols), h_w(cols);
        for (size_t i = 0; i < h_x.size(); i++)
            h_x[i] = 0.02f * static_cast<float>((i * 37) % 101) - 1.0f;
        for (int c = 0; c < cols; c++)
            h_w[c] = 0.5f + 0.001f * static_cast<float>(c % 500);
        std::vector<float> h_x_fp16(h_x.size()), h_w_fp16(cols), h_ref(h_x.size());
        for (size_t i = 0; i < h_x.size(); i++)
            h_x_fp16[i] = __half2float(__float2half(h_x[i]));
        for (int c = 0; c < cols; c++)
            h_w_fp16[c] = __half2float(__float2half(h_w[c]));
        cpu_rmsnorm(h_x_fp16.data(), h_w_fp16.data(), h_ref.data(), rows, cols, eps);

        Tensor d_x = make_gpu_tensor(h_x.data(), QType::F16, {rows, cols});
        Tensor d_w = make_gpu_tensor(h_w.data(), QType::F16, {cols});
        Tensor d_out = alloc_gpu_tensor(QType::F16, {rows, cols});
        rmsnorm(d_x, d_w, d_out, eps, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        auto h_out = read_gpu_tensor(d_out);
        double max_err = 0.0;
        for (size_t i = 0; i < h_x.size(); i++)
            max_err = std::max(max_err, (double)std::abs(h_out[i] - h_ref[i]));
        EXPECT_LT(max_err, 1e-2) << "rows=" << rows << " cols=" << cols;
        free_gpu_tensor(d_x);
        free_gpu_tensor(d_w);
        free_gpu_tensor(d_out);
    }
}

// ===========================================================================
// RMSNormNvfp4ProducerBitIdentity -- the fused rmsnorm + NVFP4 quantize
// producer kernel must emit (a) the same FP16 bytes as rmsnorm() and (b) the
// same packed nibbles + FP8 micro-scales as quantize_fp16_to_nvfp4_into()
// run on that FP16 output. Inputs include near-zero rows (low scale clamp)
// and +/-3000-range rows (micro_scale > 448 clamp).
// ===========================================================================
TEST(LayerNormTest, RMSNormNvfp4ProducerBitIdentity) {
    constexpr float eps = 1e-5f;
    const int shapes[][2] = {{2, 2048}, {32, 5120}, {64, 8192}};
    for (auto& sh : shapes) {
        const int rows = sh[0], cols = sh[1];
        std::vector<float> h_x((size_t)rows * cols), h_w(cols);
        for (size_t i = 0; i < h_x.size(); i++) {
            const int64_t r = static_cast<int64_t>(i) / cols;
            if (r % 5 == 3)
                h_x[i] = 0.0f;  // near-zero row: exercises the low scale clamp
            else if (r % 5 == 4)
                h_x[i] = 3000.0f * (((i * 13) % 7) - 3.0f) / 3.0f;  // clamp-high row
            else
                h_x[i] = 0.02f * static_cast<float>((i * 37) % 101) - 1.0f;
        }
        for (int c = 0; c < cols; c++)
            h_w[c] = 0.5f + 0.001f * static_cast<float>(c % 500);

        Tensor d_x = make_gpu_tensor(h_x.data(), QType::F16, {rows, cols});
        Tensor d_w = make_gpu_tensor(h_w.data(), QType::F16, {cols});
        Tensor d_out_ref = alloc_gpu_tensor(QType::F16, {rows, cols});
        Tensor d_out_fused = alloc_gpu_tensor(QType::F16, {rows, cols});
        const size_t packed_bytes = (size_t)rows * cols / 2;
        const size_t scale_bytes = (size_t)rows * cols / 16;
        uint8_t *ref_packed, *ref_scales, *f_packed, *f_scales;
        ASSERT_EQ(cudaMalloc(&ref_packed, packed_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&ref_scales, scale_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&f_packed, packed_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&f_scales, scale_bytes), cudaSuccess);
        cudaMemset(f_packed, 0xAA, packed_bytes);
        cudaMemset(f_scales, 0xAA, scale_bytes);

        rmsnorm(d_x, d_w, d_out_ref, eps, nullptr);
        quantize_fp16_to_nvfp4_into(d_out_ref.data, rows, cols, ref_packed, ref_scales,
                                    /*tensor_scale=*/1.0f, nullptr);
        ASSERT_TRUE(rmsnorm_nvfp4(d_x, d_w, d_out_fused, f_packed, f_scales, eps, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<uint8_t> h_ref_out(d_out_ref.nbytes()), h_f_out(d_out_fused.nbytes());
        cudaMemcpy(h_ref_out.data(), d_out_ref.data, d_out_ref.nbytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_f_out.data(), d_out_fused.data, d_out_fused.nbytes(), cudaMemcpyDeviceToHost);
        EXPECT_EQ(0, memcmp(h_ref_out.data(), h_f_out.data(), h_ref_out.size()))
            << "FP16 out differs, rows=" << rows << " cols=" << cols;

        std::vector<uint8_t> h_rp(packed_bytes), h_fp(packed_bytes), h_rs(scale_bytes),
            h_fs(scale_bytes);
        cudaMemcpy(h_rp.data(), ref_packed, packed_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fp.data(), f_packed, packed_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rs.data(), ref_scales, scale_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fs.data(), f_scales, scale_bytes, cudaMemcpyDeviceToHost);
        EXPECT_EQ(0, memcmp(h_rs.data(), h_fs.data(), scale_bytes))
            << "micro-scales differ, rows=" << rows << " cols=" << cols;
        EXPECT_EQ(0, memcmp(h_rp.data(), h_fp.data(), packed_bytes))
            << "packed nibbles differ, rows=" << rows << " cols=" << cols;

        // Envelope refusals: rows out of range / d not a multiple of 256.
        if (rows == 2) {
            Tensor d_x1 = make_gpu_tensor(h_x.data(), QType::F16, {1, cols});
            Tensor d_o1 = alloc_gpu_tensor(QType::F16, {1, cols});
            EXPECT_FALSE(rmsnorm_nvfp4(d_x1, d_w, d_o1, f_packed, f_scales, eps, nullptr));
            free_gpu_tensor(d_x1);
            free_gpu_tensor(d_o1);
        }
        cudaFree(ref_packed);
        cudaFree(ref_scales);
        cudaFree(f_packed);
        cudaFree(f_scales);
        free_gpu_tensor(d_x);
        free_gpu_tensor(d_w);
        free_gpu_tensor(d_out_ref);
        free_gpu_tensor(d_out_fused);
    }
}

}  // namespace
}  // namespace imp
