// #874: an exception thrown by the forward fn while CudaGraphRunner is
// capturing must not leave the stream in capture state. Before the fix,
// decode_fn_ throwing mid-capture skipped cudaStreamEndCapture entirely;
// every subsequent async op on the stream then failed with "operation
// failed due to a previous error during capture" until process restart,
// while /health kept reporting ok (observed with Ornith-1.0-35B Q4_K_M:
// legacy MoE host-args prefill guard throws under the prefill-chunk
// capture).

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "runtime/cuda_graph.h"

#include "test_cuda_skip.h"

using namespace imp;

namespace {
__global__ void touch_kernel(int* p) { *p = 42; }
}  // namespace

TEST(CaptureAbort, ThrowDuringCaptureLeavesStreamUsable) {
    SKIP_IF_NO_CUDA();
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    int* d = nullptr;
    ASSERT_EQ(cudaMalloc(&d, sizeof(int)), cudaSuccess);

    CudaGraphRunner runner;
    runner.set_warmup_steps(0);  // go straight to capture on the first execute
    int calls = 0;
    runner.set_decode_fn([&](cudaStream_t s) {
        calls++;
        touch_kernel<<<1, 1, 0, s>>>(d);
        // Mirror the MoE host-args guard: refuse to run under capture.
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(s, &st);
        if (st != cudaStreamCaptureStatusNone)
            throw std::runtime_error("host-args path not graph-capturable (test)");
    });

    // The throw mid-capture must be handled inside execute(): abort the
    // capture, fall back to eager, run the fn for real.
    EXPECT_NO_THROW(runner.execute(stream));

    // Stream must not be left capturing.
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    ASSERT_EQ(cudaStreamIsCapturing(stream, &st), cudaSuccess);
    EXPECT_EQ(st, cudaStreamCaptureStatusNone);

    // The eager fallback must have actually executed the work.
    EXPECT_GE(calls, 2) << "expected capture attempt + eager re-run";

    // Subsequent async ops on the same stream must succeed — this is the
    // #874 wedge: they all returned cudaErrorStreamCaptureInvalidated.
    int h = 0;
    EXPECT_EQ(cudaMemcpyAsync(&h, d, sizeof(int), cudaMemcpyDeviceToHost, stream), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(h, 42);

    // Runner keeps serving eagerly (capture permanently disabled for it).
    EXPECT_NO_THROW(runner.execute(stream));
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    cudaFree(d);
    cudaStreamDestroy(stream);
}
