// PagedQ8Epilogue (compute/attention_paged.h, audit D-9): the split-K reduce's Q8_1 epilogue is a
// per-call launcher argument, taken once per call at batch 1 and head_dim % 32 == 0. Before, a
// process global armed per layer stayed armed when no reduce ran and leaked into the next call.

#include <gtest/gtest.h>

#include "compute/attention_paged.h"

namespace {

using imp::paged_q8_epilogue_take;
using imp::PagedQ8Epilogue;

PagedQ8Epilogue armed() {
    static int q8_slot = 0;
    static float d8_slot = 0.0f;
    PagedQ8Epilogue e;
    e.q8 = &q8_slot;
    e.d8 = &d8_slot;
    return e;
}

TEST(PagedQ8Epilogue, TakenOncePerCall) {
    PagedQ8Epilogue e = armed();
    EXPECT_TRUE(paged_q8_epilogue_take(&e, 1, 128));
    EXPECT_TRUE(e.written);
    EXPECT_FALSE(paged_q8_epilogue_take(&e, 1, 128)) << "a second reduce of the same call";
    EXPECT_TRUE(e.written);
}

TEST(PagedQ8Epilogue, NotTakenOutsideItsShape) {
    PagedQ8Epilogue e = armed();
    EXPECT_FALSE(paged_q8_epilogue_take(&e, 2, 128)) << "batch 2";
    EXPECT_FALSE(paged_q8_epilogue_take(&e, 1, 80)) << "head_dim 80";
    EXPECT_FALSE(e.written);
    EXPECT_TRUE(paged_q8_epilogue_take(&e, 1, 64));
}

TEST(PagedQ8Epilogue, NothingArmedIsNeverTaken) {
    PagedQ8Epilogue none;
    EXPECT_FALSE(paged_q8_epilogue_take(&none, 1, 128));
    EXPECT_FALSE(none.written);
    EXPECT_FALSE(paged_q8_epilogue_take(nullptr, 1, 128));
}

}  // namespace
