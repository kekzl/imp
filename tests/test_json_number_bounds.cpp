// minimum / maximum / exclusive bounds for the schema FSM (compute/json_number_bounds.h): exact
// decimal comparison, finished-value checks, and prefix reachability against brute force.
#include <gtest/gtest.h>

#include "compute/json_number_bounds.h"

#include <random>
#include <string>

namespace imp {
namespace {

Decimal dec(const char* s) {
    Decimal d;
    EXPECT_TRUE(parse_decimal(s, d)) << s;
    return d;
}

NumberBounds bounds(const char* lo, bool lo_excl, const char* hi, bool hi_excl) {
    NumberBounds b;
    if (lo)
        b.tighten_lo(dec(lo), lo_excl);
    if (hi)
        b.tighten_hi(dec(hi), hi_excl);
    return b;
}

TEST(JsonNumberBounds, ParsesAndComparesExactly) {
    EXPECT_EQ(compare_decimal(dec("0.1"), dec("0.10")), 0);
    EXPECT_EQ(compare_decimal(dec("-0"), dec("0")), 0);
    EXPECT_LT(compare_decimal(dec("-2.5"), dec("-2.4")), 0);
    EXPECT_GT(compare_decimal(dec("10"), dec("9.999")), 0);
    EXPECT_EQ(compare_decimal(dec("1.5e3"), dec("1500")), 0);
    EXPECT_EQ(compare_decimal(dec("15e-3"), dec("0.015")), 0);
    Decimal d;
    EXPECT_FALSE(parse_decimal("1.", d));
    EXPECT_FALSE(parse_decimal("-", d));
    EXPECT_FALSE(parse_decimal("1e999", d));
}

TEST(JsonNumberBounds, FinishedValuesHonourExclusivity) {
    const NumberBounds b = bounds("0", true, "1", false);  // (0, 1]
    EXPECT_FALSE(number_in_bounds(b, "0"));
    EXPECT_TRUE(number_in_bounds(b, "0.0001"));
    EXPECT_TRUE(number_in_bounds(b, "1"));
    EXPECT_TRUE(number_in_bounds(b, "1.0"));
    EXPECT_FALSE(number_in_bounds(b, "1.0001"));
    EXPECT_FALSE(number_in_bounds(b, "-0.5"));
}

TEST(JsonNumberBounds, TightenKeepsTheStricterBound) {
    NumberBounds b;
    b.tighten_lo(dec("1"), false);
    b.tighten_lo(dec("1"), true);   // same value, exclusive wins
    b.tighten_lo(dec("0"), false);  // looser, ignored
    EXPECT_TRUE(b.lo_excl);
    EXPECT_FALSE(number_in_bounds(b, "1"));
    EXPECT_TRUE(number_in_bounds(b, "1.5"));
}

TEST(JsonNumberBounds, PrefixReachability) {
    const NumberBounds b = bounds("-10", false, "-2", false);  // integers [-10, -2]
    EXPECT_TRUE(number_prefix_feasible(b, "-", true, 19));
    EXPECT_TRUE(number_prefix_feasible(b, "-1", true, 19));  // -10
    EXPECT_FALSE(number_prefix_feasible(b, "-11", true, 19));
    EXPECT_FALSE(number_prefix_feasible(b, "1", true, 19));
    EXPECT_FALSE(number_prefix_feasible(b, "-0", true, 19));
    const NumberBounds unit = bounds("0", true, "1", false);     // (0, 1]
    EXPECT_TRUE(number_prefix_feasible(unit, "0", false, 40));   // 0.x
    EXPECT_FALSE(number_prefix_feasible(unit, "0", true, 19));   // integer 0 only
    EXPECT_TRUE(number_prefix_feasible(unit, "1.", false, 40));  // 1.0
    EXPECT_FALSE(number_prefix_feasible(unit, "1.5", false, 40));
    EXPECT_FALSE(number_prefix_feasible(unit, "2", false, 40));
    EXPECT_FALSE(number_prefix_feasible(unit, "-", false, 40));
    const NumberBounds big = bounds("100000", false, nullptr, false);
    EXPECT_TRUE(number_prefix_feasible(big, "9", true, 19));  // 900000
    EXPECT_FALSE(number_prefix_feasible(big, "9", true, 5));  // digit cap: at most 99999
}

// Brute force over every integer prefix of up to 4 digits: feasible iff some integer with that
// prefix (same sign, at most 4 digits) lies in the bounds.
TEST(JsonNumberBounds, IntegerPrefixMatchesBruteForce) {
    std::mt19937 rng(0xB0D5);
    std::uniform_int_distribution<int> pick(-120, 120);
    for (int round = 0; round < 25; ++round) {
        int lo = pick(rng), hi = pick(rng);
        if (lo > hi)
            std::swap(lo, hi);
        const bool lo_excl = round % 2 == 1, hi_excl = round % 3 == 1;
        const NumberBounds b = bounds(std::to_string(lo).c_str(), lo_excl, std::to_string(hi).c_str(),
                                      hi_excl);
        auto in = [&](long v) { return (lo_excl ? v > lo : v >= lo) && (hi_excl ? v < hi : v <= hi); };
        for (long v = -9999; v <= 9999; ++v) {
            const std::string s = std::to_string(v);
            if (in(v) != number_in_bounds(b, s))
                FAIL() << "value " << s << " bounds " << lo << ".." << hi;
        }
        for (const std::string sign : {"", "-"}) {
            for (long p = 0; p <= 199; ++p) {
                const std::string prefix = sign + std::to_string(p);
                // Completions with at most 4 digits: p, p0..p9, p00..p99, ... (0 takes no digit after).
                bool reachable = false;
                for (long scale = 1; p * scale <= 9999 && !reachable; scale *= 10) {
                    for (long v = p * scale; v < (p + 1) * scale && v <= 9999 && !reachable; ++v)
                        reachable = in(sign.empty() ? v : -v);
                    if (p == 0)
                        break;
                }
                EXPECT_EQ(number_prefix_feasible(b, prefix, true, 4), reachable)
                    << "prefix " << prefix << " bounds " << (lo_excl ? "(" : "[") << lo << "," << hi
                    << (hi_excl ? ")" : "]");
            }
        }
    }
}

TEST(JsonNumberBounds, FractionAtDigitCapMustBeInBounds) {
    // exclusiveMinimum 0, cap 40 digits: "0." + 38 zeros still reaches 0.0...01; with 39 zeros
    // nothing can follow and 0 is out of bounds, so the 40th zero is infeasible.
    const NumberBounds b = bounds("0", true, nullptr, false);
    const std::string z38 = "0." + std::string(38, '0');
    EXPECT_TRUE(number_prefix_feasible(b, z38, false, 40));
    EXPECT_FALSE(number_prefix_feasible(b, z38 + "0", false, 40));
    EXPECT_TRUE(number_prefix_feasible(b, z38 + "1", false, 40));
    EXPECT_FALSE(number_prefix_feasible(b, "1" + std::string(39, '0') + ".", false, 40));
    EXPECT_TRUE(number_prefix_feasible(bounds("0", false, nullptr, false), z38 + "0", false, 40));
}

}  // namespace
}  // namespace imp
