// On-disk warm weight cache (GPU, real model): a fully-cold load persists the
// transformed uploads next to the model; the next load restores them (warm
// hits > 0) and must produce token-identical greedy output. A corrupted cache
// is ignored (clean cold load). See memory/weight_cache_file.h.
//
// Requires a real model on disk: IMP_TEST_MODEL or /models/Qwen3-8B-Q8_0.gguf.
// The cache is redirected to a scratch [warm_cache] dir (the model mount is
// read-only for the container user), which also exercises the dir override.

#include <gtest/gtest.h>

#include "imp/imp.h"
#include "api/imp_internal.h"
#include "memory/weight_cache_file.h"
#include "runtime/config.h"
#include "test_models.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

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

constexpr const char* kCacheDir = "/tmp/imp-warm-cache-test";

struct Cycle {
    ImpModel model = nullptr;
    ImpContext ctx = nullptr;
    std::string output;

    bool up(const char* path) {
        if (imp_model_load(path, IMP_FORMAT_GGUF, &model) != IMP_SUCCESS)
            return false;
        // Redirect the warm cache into a writable scratch dir. The pending
        // runtime config is consumed per context create, so re-arm each time.
        imp::RuntimeConfig rc;
        rc.warm_cache.enabled = true;
        rc.warm_cache.dir = kCacheDir;
        imp::set_pending_runtime_config(rc);
        ImpConfig config = imp_config_default();
        config.max_seq_len = 1024;
        config.max_batch_size = 1;
        return imp_context_create(model, &config, &ctx) == IMP_SUCCESS;
    }

    bool generate_greedy() {
        ImpGenerateParams params = imp_generate_params_default();
        params.seed = 42;
        params.max_tokens = 16;
        params.temperature = 0.0f;
        params.apply_chat_template = 1;
        char buf[2048] = {};
        size_t n = 0;
        if (imp_generate(ctx, "Name three colors.", &params, buf, sizeof(buf), &n) != IMP_SUCCESS)
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

struct CacheCleanup {
    std::string path;
    explicit CacheCleanup(std::string p) : path(std::move(p)) { remove(); }
    ~CacheCleanup() { remove(); }
    void remove() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

}  // namespace

TEST(WarmCacheTest, ColdLoadWritesCacheWarmBootTokenIdentical) {
    SKIP_IF_NO_MODEL();
    const std::string cache = imp::weight_cache_path_for(get_model_path(), kCacheDir);
    CacheCleanup cleanup(cache);  // start from a guaranteed-cold state

    Cycle cold;
    ASSERT_TRUE(cold.up(get_model_path()));
    EXPECT_EQ(cold.model->model->last_warm_hits(), 0) << "first load must be fully cold";
    ASSERT_TRUE(cold.generate_greedy());
    cold.down();

    ASSERT_TRUE(std::filesystem::exists(cache))
        << "cold load did not persist a warm cache at " << cache;

    Cycle warm;
    ASSERT_TRUE(warm.up(get_model_path()));
    EXPECT_GT(warm.model->model->last_warm_hits(), 0) << "second load ignored the warm cache";
    ASSERT_TRUE(warm.generate_greedy());
    EXPECT_EQ(warm.output, cold.output) << "warm boot diverged from cold boot";
    warm.down();
}

TEST(WarmCacheTest, CorruptCacheIsIgnored) {
    SKIP_IF_NO_MODEL();
    const std::string cache = imp::weight_cache_path_for(get_model_path(), kCacheDir);
    CacheCleanup cleanup(cache);

    {
        std::ofstream f(cache, std::ios::binary | std::ios::trunc);
        f << "NOTACACHEFILE garbage garbage garbage";
    }

    Cycle c;
    ASSERT_TRUE(c.up(get_model_path()));
    EXPECT_EQ(c.model->model->last_warm_hits(), 0) << "corrupt cache must fall back to a cold load";
    ASSERT_TRUE(c.generate_greedy());
    c.down();
}
