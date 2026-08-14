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

// A capture records launches without executing them, so the replay that follows
// it IS that step's execution. When that first replay fails, the step has run
// nothing — and the path used to `return false` without re-running decode_fn_.
// Four of the five execute() call sites ignore the return value; the batched
// decode one (engine_scheduler.cpp) then sees logits_out.data == nullptr and
// falls back to get_logits_view(), which still holds the PREVIOUS step's logits,
// so greedy decoding repeats the token instead of failing.
TEST(CaptureAbort, FirstReplayFailureStillExecutesTheStep) {
    SKIP_IF_NO_CUDA();
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    int* d = nullptr;
    ASSERT_EQ(cudaMalloc(&d, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d, 0, sizeof(int)), cudaSuccess);

    CudaGraphRunner runner;
    runner.set_warmup_steps(0);  // capture on the first execute
    int calls = 0;
    runner.set_decode_fn([&](cudaStream_t s) {
        calls++;
        touch_kernel<<<1, 1, 0, s>>>(d);
    });

    runner.set_fail_next_replay_for_test();
    EXPECT_TRUE(runner.execute(stream)) << "a step whose work ran eagerly is not a failure";
    EXPECT_EQ(calls, 2) << "capture pass + eager re-run after the failed first replay";

    // The work must have actually reached the device, not just been recorded.
    int h = 0;
    EXPECT_EQ(cudaMemcpyAsync(&h, d, sizeof(int), cudaMemcpyDeviceToHost, stream), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(h, 42) << "the failed replay left the step unexecuted";

    // A first-replay failure is structural: stay eager instead of re-capturing
    // (and skipping a forward) on every later step.
    EXPECT_TRUE(runner.execute(stream));
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(runner.capture_count(), 1) << "must not retry capture after a failed first replay";

    cudaFree(d);
    cudaStreamDestroy(stream);
}
