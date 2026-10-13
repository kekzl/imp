#pragma once
// JSON-Schema minimum / maximum / exclusiveMinimum / exclusiveMaximum for the schema FSM: whether a
// number prefix can still end in range, and whether a finished number is in range. Exact decimal
// arithmetic on digit strings, no floating point: a bound of 0.1 is 0.1.
#include <string>
#include <string_view>

namespace imp {

// A finite decimal: integer digits without leading zeros ("" = 0), fraction digits without
// trailing zeros; zero is never negative.
struct Decimal {
    bool neg = false;
    std::string int_digits;
    std::string frac_digits;
};

// Parses a JSON number literal; an exponent shifts the digits (|exponent| <= 400).
[[nodiscard]] bool parse_decimal(std::string_view text, Decimal& out);
// <0, 0, >0 as a is less than, equal to, greater than b.
[[nodiscard]] int compare_decimal(const Decimal& a, const Decimal& b);

struct NumberBounds {
    bool has_lo = false;
    bool has_hi = false;
    bool lo_excl = false;
    bool hi_excl = false;
    Decimal lo;
    Decimal hi;
    [[nodiscard]] bool active() const { return has_lo || has_hi; }
    // Tightens the lower / upper bound; an equal value keeps the exclusive form.
    void tighten_lo(const Decimal& v, bool excl);
    void tighten_hi(const Decimal& v, bool excl);
};

// `prefix`: a JSON number prefix without exponent ("-", "0", "12", "12.", "12.5"). True when a
// completion of it with at most `max_digits` digits in all is in bounds; integer completions only
// when `integer`.
[[nodiscard]] bool number_prefix_feasible(const NumberBounds& b, std::string_view prefix, bool integer,
                                          int max_digits);
// True when the finished number `text` (no exponent) is in bounds.
[[nodiscard]] bool number_in_bounds(const NumberBounds& b, std::string_view text);

}  // namespace imp
