#include <gtest/gtest.h>
#include "runtime/green_ctx.h"
#include "runtime/cuda_graph.h"
#include "runtime/pdl.h"

#include <cuda_runtime.h>
#include <vector>
#include <atomic>

namespace imp {
namespace {

// ============================================================================
// Helper: skip if no CUDA device
// ============================================================================
static bool HasCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

#define SKIP_IF_NO_CUDA()                                                     \
    do {                                                                       \
        if (!HasCudaDevice()) {                                                \
            GTEST_SKIP() << "No CUDA device available";                        \
        }                                                                      \
    } while (0)

// Simple kernel for testing graph capture
__global__ void add_one_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

// ============================================================================
// GreenContextManager Tests
// ============================================================================

// 1. Default state
TEST(GreenCtxTest, DefaultState) {
    GreenContextManager mgr;
    EXPECT_FALSE(mgr.is_available());
    EXPECT_FALSE(mgr.has_green_contexts());
    EXPECT_EQ(mgr.prefill_stream(), nullptr);
    EXPECT_EQ(mgr.decode_stream(), nullptr);
    EXPECT_EQ(mgr.total_sms(), 0);
    EXPECT_EQ(mgr.prefill_sms(), 0);
    EXPECT_EQ(mgr.decode_sms(), 0);
}

// 2. Init creates streams
TEST(GreenCtxTest, InitCreatesStreams) {
    SKIP_IF_NO_CUDA();

    GreenContextManager mgr;
    ASSERT_TRUE(mgr.init(0, 0.8f));

    EXPECT_TRUE(mgr.is_available());
    EXPECT_NE(mgr.prefill_stream(), nullptr);
    EXPECT_NE(mgr.decode_stream(), nullptr);
    EXPECT_NE(mgr.prefill_stream(), mgr.decode_stream());

    // SM counts should be positive
    EXPECT_GT(mgr.total_sms(), 0);
    EXPECT_GT(mgr.prefill_sms(), 0);
    EXPECT_GT(mgr.decode_sms(), 0);
    EXPECT_EQ(mgr.prefill_sms() + mgr.decode_sms(), mgr.total_sms());

    mgr.destroy();
    EXPECT_FALSE(mgr.is_available());
}

// 3. SM ratio correctness
TEST(GreenCtxTest, SMRatioCorrectness) {
    SKIP_IF_NO_CUDA();

    GreenContextManager mgr;
    ASSERT_TRUE(mgr.init(0, 0.75f));

    int total = mgr.total_sms();
    int expected_prefill = static_cast<int>(total * 0.75f);
    expected_prefill = std::max(expected_prefill, 1);

    EXPECT_EQ(mgr.prefill_sms(), expected_prefill);
    EXPECT_EQ(mgr.decode_sms(), total - expected_prefill);
    EXPECT_FLOAT_EQ(mgr.prefill_ratio(), 0.75f);

    mgr.destroy();
}

// 4. Reconfigure SM split
TEST(GreenCtxTest, Reconfigure) {
    SKIP_IF_NO_CUDA();

    GreenContextManager mgr;
    ASSERT_TRUE(mgr.init(0, 0.8f));

    int prefill_80 = mgr.prefill_sms();
    int decode_80 = mgr.decode_sms();

    ASSERT_TRUE(mgr.reconfigure(0.5f));
    EXPECT_TRUE(mgr.is_available());

    // With 50/50 split, prefill should have fewer SMs than before
    int prefill_50 = mgr.prefill_sms();
    if (mgr.total_sms() > 2) {
        EXPECT_LT(prefill_50, prefill_80);
    }

    EXPECT_FLOAT_EQ(mgr.prefill_ratio(), 0.5f);

    mgr.destroy();
}

// 5. Destroy is idempotent
TEST(GreenCtxTest, DoubleDestroy) {
    SKIP_IF_NO_CUDA();

    GreenContextManager mgr;
    ASSERT_TRUE(mgr.init(0));

    mgr.destroy();
    EXPECT_FALSE(mgr.is_available());

    // Second destroy should be safe
    mgr.destroy();
    EXPECT_FALSE(mgr.is_available());
}

// 6. Streams are usable (launch a kernel on each)
TEST(GreenCtxTest, StreamsAreUsable) {
    SKIP_IF_NO_CUDA();

    GreenContextManager mgr;
    ASSERT_TRUE(mgr.init(0, 0.8f));

    const int N = 256;
    float* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    // Launch on prefill stream
    add_one_kernel<<<1, N, 0, mgr.prefill_stream()>>>(d_data, N);
    cudaStreamSynchronize(mgr.prefill_stream());

    // Launch on decode stream
    add_one_kernel<<<1, N, 0, mgr.decode_stream()>>>(d_data, N);
    cudaStreamSynchronize(mgr.decode_stream());

    // Read back: should be 2.0 everywhere
    std::vector<float> result(N);
    cudaMemcpy(result.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(result[i], 2.0f) << "index " << i;
    }

    cudaFree(d_data);
    mgr.destroy();
}

// ============================================================================
// CudaGraphCapture Tests
// ============================================================================

// 7. Capture and replay a simple kernel
TEST(CudaGraphTest, CaptureAndReplay) {
    SKIP_IF_NO_CUDA();

    const int N = 1024;
    float* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    CudaGraphCapture graph;

    // Capture
    ASSERT_TRUE(graph.begin_capture(stream));
    add_one_kernel<<<(N + 255) / 256, 256, 0, stream>>>(d_data, N);
    ASSERT_TRUE(graph.end_capture());
    EXPECT_TRUE(graph.is_captured());

    // Replay 3 times
    for (int i = 0; i < 3; i++) {
        ASSERT_TRUE(graph.replay(stream));
    }
    cudaStreamSynchronize(stream);

    // Should be 3.0 (3 replays, capture doesn't execute)
    std::vector<float> result(N);
    cudaMemcpy(result.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(result[i], 3.0f) << "index " << i;
    }

    graph.reset();
    EXPECT_FALSE(graph.is_captured());

    cudaFree(d_data);
    cudaStreamDestroy(stream);
}

// 8. Replay fails if not captured
TEST(CudaGraphTest, ReplayWithoutCapture) {
    CudaGraphCapture graph;
    EXPECT_FALSE(graph.replay(nullptr));
    EXPECT_FALSE(graph.is_captured());
}

// 9. Begin capture with null stream fails
TEST(CudaGraphTest, NullStreamCapture) {
    CudaGraphCapture graph;
    EXPECT_FALSE(graph.begin_capture(nullptr));
}

// 10. Reset after capture
TEST(CudaGraphTest, ResetAfterCapture) {
    SKIP_IF_NO_CUDA();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    CudaGraphCapture graph;
    ASSERT_TRUE(graph.begin_capture(stream));
    // Empty graph (no kernels captured)
    ASSERT_TRUE(graph.end_capture());
    EXPECT_TRUE(graph.is_captured());

    graph.reset();
    EXPECT_FALSE(graph.is_captured());

    // Can re-capture after reset
    float* d_data = nullptr;
    cudaMalloc(&d_data, sizeof(float));
    cudaMemset(d_data, 0, sizeof(float));

    ASSERT_TRUE(graph.begin_capture(stream));
    add_one_kernel<<<1, 1, 0, stream>>>(d_data, 1);
    ASSERT_TRUE(graph.end_capture());
    EXPECT_TRUE(graph.is_captured());

    ASSERT_TRUE(graph.replay(stream));
    cudaStreamSynchronize(stream);

    float val;
    cudaMemcpy(&val, d_data, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(val, 1.0f);

    cudaFree(d_data);
    cudaStreamDestroy(stream);
}

// ============================================================================
// CudaGraphRunner Tests
// ============================================================================

// 11. Runner warmup -> capture -> replay lifecycle
TEST(CudaGraphRunnerTest, FullLifecycle) {
    SKIP_IF_NO_CUDA();

    const int N = 256;
    float* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    CudaGraphRunner runner;
    runner.set_warmup_steps(1);

    runner.set_decode_fn([d_data, N](cudaStream_t s) {
        add_one_kernel<<<1, N, 0, s>>>(d_data, N);
    });

    // Step 1: warmup (direct execution)
    ASSERT_TRUE(runner.execute(stream));
    EXPECT_FALSE(runner.is_ready());
    EXPECT_EQ(runner.replay_count(), 0);
    EXPECT_EQ(runner.capture_count(), 0);

    cudaStreamSynchronize(stream);

    // Step 2: capture + first replay
    ASSERT_TRUE(runner.execute(stream));
    EXPECT_TRUE(runner.is_ready());
    EXPECT_EQ(runner.capture_count(), 1);
    EXPECT_EQ(runner.replay_count(), 1);

    // Steps 3-5: replay
    for (int i = 0; i < 3; i++) {
        ASSERT_TRUE(runner.execute(stream));
    }
    EXPECT_EQ(runner.replay_count(), 4);

    cudaStreamSynchronize(stream);

    // Total: 1 warmup + 1 capture + 4 replays = value should be 5.0
    // Wait: warmup executes the function, capture records but doesn't execute,
    // then capture also replays once, plus 3 more replays = 1 + 0 + 4 = 5 runs
    // Actually let me re-check the CudaGraphRunner implementation:
    // - warmup: decode_fn_(stream) -> data += 1 (value = 1)
    // - capture: begin_capture, decode_fn_(stream) [recorded, not executed], end_capture, replay -> data += 1 (value = 2)
    // - replay x3 -> data += 3 (value = 5)
    std::vector<float> result(N);
    cudaMemcpy(result.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(result[0], 5.0f);

    cudaFree(d_data);
    cudaStreamDestroy(stream);
}

// 12. Runner invalidate resets lifecycle
TEST(CudaGraphRunnerTest, Invalidate) {
    SKIP_IF_NO_CUDA();

    float* d_data = nullptr;
    cudaMalloc(&d_data, sizeof(float));
    cudaMemset(d_data, 0, sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    CudaGraphRunner runner;
    runner.set_warmup_steps(1);
    runner.set_decode_fn([d_data](cudaStream_t s) {
        add_one_kernel<<<1, 1, 0, s>>>(d_data, 1);
    });

    // Warmup + capture
    runner.execute(stream);
    runner.execute(stream);
    EXPECT_TRUE(runner.is_ready());

    // Invalidate
    runner.invalidate();
    EXPECT_FALSE(runner.is_ready());

    // Should restart warmup
    runner.execute(stream);  // warmup again
    EXPECT_FALSE(runner.is_ready());
    runner.execute(stream);  // capture again
    EXPECT_TRUE(runner.is_ready());

    EXPECT_EQ(runner.capture_count(), 2);

    cudaStreamSynchronize(stream);
    cudaFree(d_data);
    cudaStreamDestroy(stream);
}

// 13. Runner without decode fn fails gracefully
TEST(CudaGraphRunnerTest, NoDecodeFn) {
    CudaGraphRunner runner;
    EXPECT_FALSE(runner.execute(nullptr));
}

// ============================================================================
// PDL Tests
// ============================================================================

// 14. PDL is_available check (doesn't crash)
TEST(PDLTest, IsAvailableCheck) {
    // Just check it doesn't crash; result depends on hardware
    bool avail = pdl::is_available();
    (void)avail;
    EXPECT_TRUE(true);
}

// 15. PDL enable/disable on kernel (doesn't crash)
TEST(PDLTest, EnableDisableKernel) {
    SKIP_IF_NO_CUDA();

    // Enable PDL on our test kernel
    pdl::enable(reinterpret_cast<const void*>(&add_one_kernel));

    // Kernel should still work correctly
    const int N = 64;
    float* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    add_one_kernel<<<1, N>>>(d_data, N);
    cudaDeviceSynchronize();

    std::vector<float> result(N);
    cudaMemcpy(result.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(result[i], 1.0f);
    }

    // Disable
    pdl::disable(reinterpret_cast<const void*>(&add_one_kernel));

    cudaFree(d_data);
}

// 16. PDL enable with null is safe
TEST(PDLTest, EnableNullSafe) {
    pdl::enable(nullptr);
    pdl::disable(nullptr);
    EXPECT_TRUE(true);
}

// 17. ScopedPDL RAII
TEST(PDLTest, ScopedPDL) {
    SKIP_IF_NO_CUDA();

    {
        pdl::ScopedPDL guard(reinterpret_cast<const void*>(&add_one_kernel),
                             /*auto_disable=*/true);
        // Kernel should work with PDL enabled
        float* d_data = nullptr;
        cudaMalloc(&d_data, sizeof(float));
        cudaMemset(d_data, 0, sizeof(float));

        add_one_kernel<<<1, 1>>>(d_data, 1);
        cudaDeviceSynchronize();

        float val;
        cudaMemcpy(&val, d_data, sizeof(float), cudaMemcpyDeviceToHost);
        EXPECT_FLOAT_EQ(val, 1.0f);

        cudaFree(d_data);
    }
    // ScopedPDL destructor should have disabled PDL (no crash)
    EXPECT_TRUE(true);
}

} // namespace
} // namespace imp
