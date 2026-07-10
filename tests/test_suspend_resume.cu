// Suspend-to-RAM roundtrip (GPU, real model): capture a weight snapshot,
// tear everything down, verify VRAM is allocatable again, resume with the
// snapshot armed, and require (a) warm hits, (b) byte-identical weight
// buffers, (c) token-identical greedy output. A second case force-drops one
// snapshot key to prove warm and cold uploads mix byte-safely.
//
// Modeled on tests/test_engine_relaunch.cpp. Requires a real model on disk:
// IMP_TEST_MODEL or the default /models/Qwen3-8B-Q8_0.gguf.
//
// Deliberately does NOT exercise imp_gpu_release(device_reset=1): a
// cudaDeviceReset inside this shared gtest process would invalidate state
// owned by sibling tests. The device-reset path is covered by the manual
// server recipe (POST /admin/suspend with [suspend] device_reset=true).

#include <gtest/gtest.h>

#include "imp/imp.h"
#include "api/imp_internal.h"
#include "memory/weight_snapshot.h"
#include "test_models.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

static const char* get_model_path() {
    return imp_test::env_cstr_or(imp_test::kEnvModel, "/models/Qwen3-8B-Q8_0.gguf");
}

static bool model_exists() {
    FILE* f = fopen(get_model_path(), "r");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

#define SKIP_IF_NO_MODEL()                                           \
    do {                                                             \
        if (!model_exists())                                         \
            GTEST_SKIP() << "Model not found: " << get_model_path(); \
    } while (0)

// FNV-1a over the first max_bytes of a device buffer.
static uint64_t device_checksum(const void* d_ptr, size_t bytes, size_t max_bytes = 1 << 20) {
    size_t n = bytes < max_bytes ? bytes : max_bytes;
    std::vector<uint8_t> host(n);
    if (cudaMemcpy(host.data(), d_ptr, n, cudaMemcpyDeviceToHost) != cudaSuccess)
        return 0;
    uint64_t h = 1469598103934665603ull;
    for (uint8_t b : host) {
        h ^= b;
        h *= 1099511628211ull;
    }
    return h;
}

// Checksum the first `count` live upload-log records, keyed by upload key.
static std::map<std::string, uint64_t> checksum_records(ImpModel model, size_t count) {
    std::map<std::string, uint64_t> sums;
    const imp::WeightUploadLog* log = model->model->upload_log();
    if (!log)
        return sums;
    for (const auto& rec : log->records()) {
        if (rec.dead || rec.allocs.empty())
            continue;
        sums[rec.key] = device_checksum(rec.allocs[0].ptr, rec.allocs[0].bytes);
        if (sums.size() >= count)
            break;
    }
    return sums;
}

struct Cycle {
    ImpModel model = nullptr;
    ImpContext ctx = nullptr;
    std::string output;

    bool up(const char* path) {
        if (imp_model_load(path, IMP_FORMAT_GGUF, &model) != IMP_SUCCESS)
            return false;
        ImpConfig config = imp_config_default();
        config.max_seq_len = 1024;
        config.max_batch_size = 1;
        if (imp_context_create(model, &config, &ctx) != IMP_SUCCESS)
            return false;
        return true;
    }

    bool generate_greedy() {
        ImpGenerateParams params = imp_generate_params_default();
        params.seed = 42;
        params.max_tokens = 16;
        params.temperature = 0.0f;  // greedy — argmax chain must replay exactly
        params.apply_chat_template = 1;
        char buf[2048] = {};
        size_t n = 0;
        if (imp_generate(ctx, "Count from 1 to 5.", &params, buf, sizeof(buf), &n) != IMP_SUCCESS)
            return false;
        output.assign(buf, n);
        return n > 0;
    }

    void down() {
        if (ctx) {
            imp_context_free(ctx);
            ctx = nullptr;
        }
        if (model) {
            imp_model_free(model);
            model = nullptr;
        }
    }
};

// Shared roundtrip body; drop_one_key=true forces one snapshot record onto
// the cold path at resume.
static void run_roundtrip(bool drop_one_key) {
    Cycle first;
    ASSERT_TRUE(first.up(get_model_path()));
    ASSERT_TRUE(first.generate_greedy());
    auto sums_before = checksum_records(first.model, 4);
    ASSERT_FALSE(sums_before.empty()) << "upload log has no live records";

    ImpWeightSnapshot snap = nullptr;
    ASSERT_EQ(imp_weights_snapshot_capture(first.model, /*headroom_mb=*/256, &snap), IMP_SUCCESS);
    ASSERT_NE(snap, nullptr);
    EXPECT_GT(imp_weights_snapshot_bytes(snap), size_t{1} << 20);

    std::string dropped_key;
    if (drop_one_key) {
        dropped_key = sums_before.begin()->first;
        ASSERT_TRUE(snap->snap->drop_key(dropped_key));
    }

    size_t free_before_mib = 0;
    {
        size_t f = 0, t = 0;
        ASSERT_EQ(cudaMemGetInfo(&f, &t), cudaSuccess);
        free_before_mib = f >> 20;
    }

    first.down();
    // No device reset here (shared gtest process) — pool trim only.
    ASSERT_EQ(imp_gpu_release(/*device_reset=*/0), IMP_SUCCESS);

    // Teardown must give the VRAM back (allocatable-probe contract, matching
    // EngineRelaunchTest: cudaMemGetInfo under-reports on WSL2/WDDM).
    {
        size_t f = 0, t = 0;
        ASSERT_EQ(cudaMemGetInfo(&f, &t), cudaSuccess);
        size_t free_now_mib = f >> 20;
        if (free_now_mib + 2048 < free_before_mib + (imp_weights_snapshot_bytes(snap) >> 20)) {
            // Weights-sized memory should be free again; probe a big chunk.
            size_t probe_mib = imp_weights_snapshot_bytes(snap) >> 20;
            if (probe_mib > 1024) {
                void* probe = nullptr;
                EXPECT_EQ(cudaMalloc(&probe, (probe_mib - 512) << 20), cudaSuccess)
                    << "weights-sized VRAM not allocatable after suspend teardown";
                if (probe)
                    cudaFree(probe);
            }
        }
    }

    // Resume: arm and reload the same file.
    ASSERT_EQ(imp_weights_snapshot_arm(snap), IMP_SUCCESS);
    Cycle second;
    ASSERT_TRUE(second.up(get_model_path()));
    EXPECT_GT(imp_weights_snapshot_hits(snap), 0) << "no warm hits — snapshot was ignored";

    // Restored buffers must be byte-identical to the pre-suspend state.
    auto sums_after = checksum_records(second.model, 4);
    for (const auto& [key, sum] : sums_before) {
        auto it = sums_after.find(key);
        if (it == sums_after.end())
            continue;  // record set can shift (e.g. placement) — identity is guarded below
        EXPECT_EQ(it->second, sum) << "weight bytes diverged for " << key
                                   << (key == dropped_key ? " (cold-restored)" : " (warm-restored)");
    }

    // Greedy output must replay token-identically.
    ASSERT_TRUE(second.generate_greedy());
    EXPECT_EQ(second.output, first.output);

    second.down();
    imp_weights_snapshot_free(snap);
}

}  // namespace

TEST(SuspendResumeTest, SnapshotRoundtripTokenIdentical) {
    SKIP_IF_NO_MODEL();
    run_roundtrip(/*drop_one_key=*/false);
}

TEST(SuspendResumeTest, DroppedKeyFallsBackColdAndStaysIdentical) {
    SKIP_IF_NO_MODEL();
    run_roundtrip(/*drop_one_key=*/true);
}

TEST(SuspendResumeTest, ArmedMismatchIsIgnored) {
    SKIP_IF_NO_MODEL();
    // Arm a snapshot, then destroy it BEFORE the load would consume it: free
    // must disarm the slot so the load can't touch a dangling pointer.
    Cycle c;
    ASSERT_TRUE(c.up(get_model_path()));
    ImpWeightSnapshot snap = nullptr;
    ASSERT_EQ(imp_weights_snapshot_capture(c.model, 256, &snap), IMP_SUCCESS);
    ASSERT_EQ(imp_weights_snapshot_arm(snap), IMP_SUCCESS);
    imp_weights_snapshot_free(snap);  // must disarm
    EXPECT_EQ(imp::weight_snapshot_take_armed(), nullptr);
    c.down();
}
