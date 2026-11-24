// #2347: "library reserve MISMATCH ... measured 0 MiB" (Qwen3.8-27B) and 6 MiB (Flash-Next): the
// whole-init residual (device use minus the pool ledger) was negative, -253 MiB on the 27B, because
// a pool is counted twice; clamped to 0 it read as a measurement and was recorded for the next start.
#include "runtime/engine_internal.h"
#include <gtest/gtest.h>
#include <cstdint>

namespace imp {
namespace {

constexpr size_t kMiB = size_t{1} << 20;

TEST(LibraryReserveMeasurement, NoForwardWindowIsNoMeasurement) {
    EXPECT_FALSE(engine_internal::library_reserve_measurement(SIZE_MAX, 4000 * int64_t{1 << 20}).has_value());
}

TEST(LibraryReserveMeasurement, TakesTheLargerOfWindowAndResidual) {
    auto m = engine_internal::library_reserve_measurement(500 * kMiB, 3000 * int64_t{1 << 20});
    ASSERT_TRUE(m.has_value());
    EXPECT_EQ(*m, 3000 * kMiB);
    m = engine_internal::library_reserve_measurement(7500 * kMiB, 100 * int64_t{1 << 20});
    ASSERT_TRUE(m.has_value());
    EXPECT_EQ(*m, 7500 * kMiB);
}

// The 27B start: forward window 0, residual -253 MiB. Not a measurement of 0 MiB.
TEST(LibraryReserveMeasurement, NegativeResidualIsNotAMeasurement) {
    EXPECT_FALSE(engine_internal::library_reserve_measurement(0, -253 * int64_t{1 << 20}).has_value());
    // Flash-Next: forward window 6 MiB, residual negative: 6 MiB is not the library charge either.
    EXPECT_FALSE(
        engine_internal::library_reserve_measurement(6 * kMiB, -27269 * int64_t{1 << 20}).has_value());
}

}  // namespace
}  // namespace imp

// Inkling-2-Mini, 2 x 81920: the first forward took 2048 MiB of device memory, 733 MiB of it
// context rows a tracked pool committed in the window. The residual grew by the difference only.
TEST(LibraryForwardWindow, TrackedGrowthInTheWindowIsNotALibraryClaim) {
    const int64_t mib = int64_t{1} << 20;
    const int64_t before = 0, device_growth = 2048 * mib, pool_growth = 733 * mib;
    EXPECT_EQ(imp::engine_internal::library_forward_window(before, before + device_growth - pool_growth, 0),
              static_cast<size_t>(1315 * mib));
}

// Bytes attributed outside the ledger (IMMA planes) come off as well; a window that explains
// more than the residual grew reads 0, never a wrapped size_t.
TEST(LibraryForwardWindow, NamedBytesComeOffAndTheResultNeverWraps) {
    const int64_t mib = int64_t{1} << 20;
    EXPECT_EQ(imp::engine_internal::library_forward_window(100 * mib, 900 * mib,
                                                           static_cast<size_t>(300 * mib)),
              static_cast<size_t>(500 * mib));
    EXPECT_EQ(imp::engine_internal::library_forward_window(100 * mib, 300 * mib,
                                                           static_cast<size_t>(300 * mib)),
              0u);
    EXPECT_EQ(imp::engine_internal::library_forward_window(500 * mib, 100 * mib, 0), 0u);
    // A ledger above device use before the window (negative residual) still measures the growth.
    EXPECT_EQ(imp::engine_internal::library_forward_window(-200 * mib, 300 * mib, 0),
              static_cast<size_t>(500 * mib));
}
