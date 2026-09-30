// diagnostics.profile gate (exec/forward_profile.h). CPU lane: pure predicate.
#include <gtest/gtest.h>

#include "exec/forward_profile.h"

namespace imp {
namespace {

// The profile tail syncs the stream; during capture that invalidated the ConditionalRunner graph.
TEST(ForwardProfile, NeverProfilesACapturingStream) {
    EXPECT_FALSE(forward_profile_allowed(true, cudaStreamCaptureStatusActive));
    EXPECT_FALSE(forward_profile_allowed(true, cudaStreamCaptureStatusInvalidated));
}

TEST(ForwardProfile, ProfilesAnEagerStreamOnlyWhenFlagged) {
    EXPECT_TRUE(forward_profile_allowed(true, cudaStreamCaptureStatusNone));
    EXPECT_FALSE(forward_profile_allowed(false, cudaStreamCaptureStatusNone));
}

}  // namespace
}  // namespace imp
