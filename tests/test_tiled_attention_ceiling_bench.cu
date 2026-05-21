// =============================================================================
// test_tiled_attention_ceiling_bench.cu — gtest harness for Säule 3
// =============================================================================

#include "bench/tiled_attention_ceiling_bench.h"

#include <gtest/gtest.h>

#include <cstdio>

TEST(TiledAttentionCeilingBench, FP16_Br64_Bkv64_HD128) {
    imp::TiledAttnCeilingResult r{};
    bool ok = imp::tiled_attention_ceiling_bench(&r);
    ASSERT_TRUE(ok) << "tiled_attention_ceiling_bench launch failed";

    std::printf(
        "\nTILED_CEILING_RESULT: tile_ns_overlap=%.2f tflops=%.0f kv_bw=%.0f_GBs\n",
        r.tile_ns, r.effective_tflops, r.kv_bandwidth_gb_per_s);
    std::fflush(stdout);
}
