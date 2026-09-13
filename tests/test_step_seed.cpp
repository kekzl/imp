#include <gtest/gtest.h>
#include "runtime/engine_internal.h"

#include <climits>

namespace imp {
namespace {

// compute_step_seed feeds the host samplers; a negative seed reads as unset and draws with the
// fixed 42u. Unmasked, hash(id) ^ clock went negative whenever clock bit 31 was set (2.1 s of every
// 4.3 s), which a test cannot reach on demand; seed INT_MAX + step takes the same overflow.
TEST(StepSeed, SeededRequestAddsTheOutputCount) {
    Request req;
    req.seed = 7;
    req.output_tokens = {11, 12, 13};
    EXPECT_EQ(engine_internal::compute_step_seed(req), 10);
    req.seed = INT_MAX;
    EXPECT_GE(engine_internal::compute_step_seed(req), 0);
}

}  // namespace
}  // namespace imp
