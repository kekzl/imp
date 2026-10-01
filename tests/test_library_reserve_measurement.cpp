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
