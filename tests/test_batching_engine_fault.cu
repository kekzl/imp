// AUDIT_arch_2026 D-1: a device fault inside a step must reach the server as
// a host signal. The in-flight request finishes "internal_error" (the handler
// maps it to 500), faulted() flips so /health reports engine_faulted, and a
// later submit is refused at once instead of parking on a queue nobody drains.
//
// Death test in threadsafe style: the child is a fresh process with its own
// CUDA context, and the poisoned context dies with it. Requires a real model
// on disk: IMP_TEST_MODEL or the default /models/Qwen3-8B-Q8_0.gguf, matching
// test_engine_relaunch.cpp.
#include <gtest/gtest.h>
#include "imp/imp.h"
#include "api/imp_internal.h"
#include "batching_engine.h"
#include "core/cuda_errors.h"
#include "model/model.h"
#include "model/tokenizer.h"
#include "runtime/engine.h"
#include "test_models.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace {

__global__ void write_through_bad_pointer(int* p) { p[threadIdx.x] = 1; }

const char* model_path() {
    return imp_test::env_cstr_or(imp_test::kEnvModel, "/models/Qwen3-8B-Q8_0.gguf");
}

bool model_exists() {
    FILE* f = std::fopen(model_path(), "r");
    if (!f)
        return false;
    std::fclose(f);
    return true;
}

constexpr int kChildOk = 42;

[[noreturn]] void die(int code, const char* what) {
    std::fprintf(stderr, "child: %s\n", what);
    std::_Exit(code);
}

std::shared_ptr<ServerRequest> make_request(ImpContext ctx, const char* prompt, int max_tokens) {
    auto sr = std::make_shared<ServerRequest>();
    sr->request = std::make_shared<imp::Request>();
    sr->request->input_tokens = ctx->engine->model()->tokenizer()->encode(prompt);
    sr->request->max_tokens = max_tokens;
    return sr;
}

// Pops until the request's final event; false when `ms` elapse first.
bool pop_until_last(ServerRequest& sr, TokenEvent& ev, int ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    for (;;) {
        if (sr.pop_token(ev, 500) && ev.is_last)
            return true;
        if (std::chrono::steady_clock::now() > deadline)
            return false;
    }
}

[[noreturn]] void run_faulted_engine_child() {
    ImpModel model = nullptr;
    if (imp_model_load(model_path(), IMP_FORMAT_GGUF, &model) != IMP_SUCCESS)
        die(3, "model load failed");
    ImpConfig config = imp_config_default();
    config.max_seq_len = 1024;
    config.max_batch_size = 2;
    ImpContext ctx = nullptr;
    if (imp_context_create(model, &config, &ctx) != IMP_SUCCESS)
        die(4, "context create failed");

    BatchingEngine be;
    be.start(ctx);
    auto first = make_request(ctx, "Write a long story about a dragon who learns to read.", 256);
    be.submit(first);

    // Wait for the first token so the fault lands inside a running generation.
    TokenEvent ev{};
    const auto first_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (!first->pop_token(ev, 500))
        if (std::chrono::steady_clock::now() > first_deadline)
            die(5, "no first token within 120 s");
    if (ev.is_last)
        die(6, "the request finished before the fault");

    // Poison the context from this thread; the worker meets the sticky error
    // at its next sync or launch.
    write_through_bad_pointer<<<1, 32>>>(reinterpret_cast<int*>(0x10));
    const cudaError_t sync = cudaDeviceSynchronize();
    if (!imp::cuda_error_is_unrecoverable(sync))
        die(7, "the fault kernel did not poison the context");

    if (!pop_until_last(*first, ev, 30000))
        die(8, "the in-flight request never finished after the fault");
    if (std::strcmp(ev.finish_reason, "internal_error") != 0) {
        std::fprintf(stderr, "child: in-flight request finished with %s\n", ev.finish_reason);
        std::_Exit(9);
    }
    const auto fault_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!be.faulted())
        if (std::chrono::steady_clock::now() > fault_deadline)
            die(10, "faulted() stayed false for 10 s");

    // A later request must not park on the queue and must carry no token:
    // internal_error at once.
    auto second = make_request(ctx, "Say hi.", 8);
    be.submit(second);
    if (!second->pop_token(ev, 5000))
        die(11, "the post-fault request got no answer in 5 s");
    if (!ev.is_last || ev.token_id != -1 || std::strcmp(ev.finish_reason, "internal_error") != 0)
        die(12, "the post-fault request was not refused with internal_error");

    std::fprintf(stderr, "faulted-engine contract held\n");
    std::_Exit(kChildOk);  // no teardown: the context is poisoned
}

TEST(DeviceFaultSignalTest, FaultInsideAStepCancelsFaultsAndRefusesLaterRequests) {
    if (!model_exists())
        GTEST_SKIP() << "Model not found: " << model_path();
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(run_faulted_engine_child(), ::testing::ExitedWithCode(kChildOk),
                "faulted-engine contract held");
}

}  // namespace
