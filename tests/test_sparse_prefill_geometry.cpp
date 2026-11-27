// Host-only cases of tests/test_sparse_prefill.cu, in test-core (CPU lane): no CUDA call.
#include <gtest/gtest.h>
#include "exec/sparse_attn_select.h"

namespace imp {
namespace {

TEST(SparsePrefillGeometry, PastTokensKeepsTheTail) {
    EXPECT_EQ(sparse_prefill_past_tokens(100, 16, 7), 100);
    EXPECT_EQ(sparse_prefill_past_tokens(100, 16, 4), 3 * 16 + 4);
    EXPECT_EQ(sparse_prefill_past_tokens(128, 16, 4), 4 * 16);
}

}  // namespace
}  // namespace imp
