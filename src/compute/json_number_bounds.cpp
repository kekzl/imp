#include "compute/json_number_bounds.h"

#include <algorithm>
#include <cstdlib>

namespace imp {
namespace {

constexpr int kMaxExponent = 400;

void normalize(Decimal& d) {
    const size_t i = d.int_digits.find_first_not_of('0');
    d.int_digits = i == std::string::npos ? std::string() : d.int_digits.substr(i);
    const size_t j = d.frac_digits.find_last_not_of('0');
    d.frac_digits = j == std::string::npos ? std::string() : d.frac_digits.substr(0, j + 1);
    if (d.int_digits.empty() && d.frac_digits.empty())
        d.neg = false;
}

Decimal make(bool neg, std::string int_digits, std::string frac_digits = {}) {
    Decimal d{neg, std::move(int_digits), std::move(frac_digits)};
    normalize(d);
    return d;
}

int compare_magnitude(const Decimal& a, const Decimal& b) {
    if (a.int_digits.size() != b.int_digits.size())
        return a.int_digits.size() < b.int_digits.size() ? -1 : 1;
    if (const int c = a.int_digits.compare(b.int_digits); c != 0)
        return c < 0 ? -1 : 1;
    const size_t n = std::max(a.frac_digits.size(), b.frac_digits.size());
    for (size_t i = 0; i < n; ++i) {
        const char x = i < a.frac_digits.size() ? a.frac_digits[i] : '0';
        const char y = i < b.frac_digits.size() ? b.frac_digits[i] : '0';
        if (x != y)
            return x < y ? -1 : 1;
    }
    return 0;
}

// "199" -> "200", "" -> "1".
std::string inc_digits(std::string s) {
    int i = static_cast<int>(s.size()) - 1;
    while (i >= 0 && s[i] == '9')
        s[i--] = '0';
    if (i < 0)
        s.insert(s.begin(), '1');
    else
        ++s[i];
    return s;
}

// "200" -> "199"; s > 0.
std::string dec_digits(std::string s) {
    int i = static_cast<int>(s.size()) - 1;
    while (i >= 0 && s[i] == '0')
        s[i--] = '9';
    if (i >= 0)
        --s[i];
    return s;
}

Decimal negate(Decimal d) {
    d.neg = !d.neg;
    normalize(d);
    return d;
}

// Integer +1 / -1 (frac_digits empty).
Decimal add_one(const Decimal& d) {
    if (!d.neg)
        return make(false, inc_digits(d.int_digits));
    return make(true, dec_digits(d.int_digits));
}
Decimal sub_one(const Decimal& d) {
    if (d.neg || d.int_digits.empty())
        return make(true, inc_digits(d.int_digits));
    return make(false, dec_digits(d.int_digits));
}

// One end of an interval; `inf` is -inf on a lower end, +inf on an upper end.
struct End {
    Decimal v;
    bool inf = false;
    bool incl = true;
};

End lower_of(const End& a, const End& b) {  // the larger lower end
    if (a.inf)
        return b;
    if (b.inf)
        return a;
    const int c = compare_decimal(a.v, b.v);
    if (c != 0)
        return c > 0 ? a : b;
    return End{a.v, false, a.incl && b.incl};
}

End upper_of(const End& a, const End& b) {  // the smaller upper end
    if (a.inf)
        return b;
    if (b.inf)
        return a;
    const int c = compare_decimal(a.v, b.v);
    if (c != 0)
        return c < 0 ? a : b;
    return End{a.v, false, a.incl && b.incl};
}

bool nonempty(const End& lo, const End& hi) {
    if (lo.inf || hi.inf)
        return true;
    const int c = compare_decimal(lo.v, hi.v);
    return c < 0 || (c == 0 && lo.incl && hi.incl);
}

End bound_lo(const NumberBounds& b) { return b.has_lo ? End{b.lo, false, !b.lo_excl} : End{{}, true, false}; }
End bound_hi(const NumberBounds& b) { return b.has_hi ? End{b.hi, false, !b.hi_excl} : End{{}, true, false}; }

// Integer window of the bounds: the smallest / largest integer inside them.
End int_lo(const NumberBounds& b) {
    if (!b.has_lo)
        return End{{}, true, true};
    const Decimal& d = b.lo;
    if (d.frac_digits.empty())
        return End{b.lo_excl ? add_one(d) : d, false, true};
    // ceil: 3.5 -> 4, -3.5 -> -3.
    return End{d.neg ? make(true, d.int_digits) : make(false, inc_digits(d.int_digits)), false, true};
}
End int_hi(const NumberBounds& b) {
    if (!b.has_hi)
        return End{{}, true, true};
    const Decimal& d = b.hi;
    if (d.frac_digits.empty())
        return End{b.hi_excl ? sub_one(d) : d, false, true};
    // floor: 3.5 -> 3, -3.5 -> -4.
    return End{d.neg ? make(true, inc_digits(d.int_digits)) : make(false, d.int_digits), false, true};
}

// Magnitudes [ma, mz) (number) or the integers [ma, mz - 1] (integer), signed by `neg`, meet the bounds.
bool magnitude_ok(const NumberBounds& b, bool neg, const Decimal& ma, const Decimal& mz, bool integer) {
    End lo, hi;
    if (integer) {
        const Decimal top = sub_one(mz);
        lo = neg ? End{negate(top), false, true} : End{ma, false, true};
        hi = neg ? End{negate(ma), false, true} : End{top, false, true};
        return nonempty(lower_of(lo, int_lo(b)), upper_of(hi, int_hi(b)));
    }
    lo = neg ? End{negate(mz), false, false} : End{ma, false, true};
    hi = neg ? End{negate(ma), false, true} : End{mz, false, false};
    return nonempty(lower_of(lo, bound_lo(b)), upper_of(hi, bound_hi(b)));
}

// Magnitudes I*10^k .. (I+1)*10^k for k = 0..max_k: the integer part may still grow.
bool growing_ok(const NumberBounds& b, bool neg, const std::string& digits, bool integer, int max_k) {
    const std::string next = inc_digits(digits);
    for (int k = 0; k <= max_k; ++k) {
        const std::string zeros(static_cast<size_t>(k), '0');
        const Decimal ma = make(false, digits + zeros);
        if (magnitude_ok(b, neg, ma, make(false, next + zeros), integer))
            return true;
        // Larger k only moves away from a bound on the far side.
        if (!neg && b.has_hi && compare_decimal(ma, b.hi) > 0)
            return false;
        if (neg && b.has_lo && compare_decimal(negate(ma), b.lo) < 0)
            return false;
    }
    return false;
}

}  // namespace

bool parse_decimal(std::string_view t, Decimal& out) {
    size_t i = 0;
    out = Decimal{};
    if (i < t.size() && t[i] == '-') {
        out.neg = true;
        ++i;
    }
    const size_t int_start = i;
    while (i < t.size() && t[i] >= '0' && t[i] <= '9')
        ++i;
    if (i == int_start)
        return false;
    std::string int_digits(t.substr(int_start, i - int_start));
    std::string frac_digits;
    if (i < t.size() && t[i] == '.') {
        const size_t f = ++i;
        while (i < t.size() && t[i] >= '0' && t[i] <= '9')
            ++i;
        if (i == f)
            return false;
        frac_digits = std::string(t.substr(f, i - f));
    }
    long exp = 0;
    if (i < t.size() && (t[i] == 'e' || t[i] == 'E')) {
        const std::string rest(t.substr(i + 1));
        char* end = nullptr;
        exp = std::strtol(rest.c_str(), &end, 10);
        if (end == rest.c_str() || *end != '\0' || exp > kMaxExponent || exp < -kMaxExponent)
            return false;
        i = t.size();
    }
    if (i != t.size())
        return false;
    if (exp > 0) {
        frac_digits.resize(std::max(frac_digits.size(), static_cast<size_t>(exp)), '0');
        int_digits += frac_digits.substr(0, static_cast<size_t>(exp));
        frac_digits.erase(0, static_cast<size_t>(exp));
    } else if (exp < 0) {
        const size_t n = static_cast<size_t>(-exp);
        if (int_digits.size() < n)
            int_digits.insert(0, n - int_digits.size(), '0');
        frac_digits.insert(0, int_digits.substr(int_digits.size() - n));
        int_digits.erase(int_digits.size() - n);
    }
    out.int_digits = std::move(int_digits);
    out.frac_digits = std::move(frac_digits);
    normalize(out);
    return true;
}

int compare_decimal(const Decimal& a, const Decimal& b) {
    if (a.neg != b.neg)
        return a.neg ? -1 : 1;
    const int m = compare_magnitude(a, b);
    return a.neg ? -m : m;
}

void NumberBounds::tighten_lo(const Decimal& v, bool excl) {
    const int c = has_lo ? compare_decimal(v, lo) : 1;
    if (c > 0 || (c == 0 && excl)) {
        lo = v;
        lo_excl = excl;
    }
    has_lo = true;
}

void NumberBounds::tighten_hi(const Decimal& v, bool excl) {
    const int c = has_hi ? compare_decimal(v, hi) : -1;
    if (c < 0 || (c == 0 && excl)) {
        hi = v;
        hi_excl = excl;
    }
    has_hi = true;
}

bool number_prefix_feasible(const NumberBounds& b, std::string_view prefix, bool integer, int max_digits) {
    if (!b.active() || prefix.empty())
        return true;
    const bool neg = prefix[0] == '-';
    const std::string_view rest = neg ? prefix.substr(1) : prefix;
    const size_t dot = rest.find('.');
    const std::string digits(rest.substr(0, dot));
    if (digits.empty()) {  // "-": any negative value, or zero
        const End zero{Decimal{}, false, true};
        return integer ? nonempty(int_lo(b), upper_of(zero, int_hi(b)))
                       : nonempty(bound_lo(b), upper_of(zero, bound_hi(b)));
    }
    if (dot != std::string_view::npos) {  // [I.F, I.F + 10^-|F|)
        const std::string frac(rest.substr(dot + 1));
        // At the digit cap no digit follows: the prefix is the value ("min_total": 0.000... under
        // exclusiveMinimum 0 ran to 40 zeros, then neither a digit nor the close was legal).
        if (static_cast<int>(digits.size() + frac.size()) >= max_digits)
            return !frac.empty() && number_in_bounds(b, prefix);
        const std::string scaled = inc_digits(digits + frac);
        const size_t split = scaled.size() - frac.size();
        return magnitude_ok(b, neg, make(false, digits, frac),
                            make(false, scaled.substr(0, split), scaled.substr(split)), false);
    }
    if (digits == "0")  // no further integer digits: 0 (integer) or [0, 1) (number)
        return magnitude_ok(b, neg, Decimal{}, make(false, "1"), integer);
    const int max_k = std::max(0, max_digits - static_cast<int>(digits.size()));
    return growing_ok(b, neg, digits, integer, max_k);
}

bool number_in_bounds(const NumberBounds& b, std::string_view text) {
    if (!b.active())
        return true;
    Decimal v;
    if (!parse_decimal(text, v))
        return false;
    if (b.has_lo) {
        const int c = compare_decimal(v, b.lo);
        if (c < 0 || (c == 0 && b.lo_excl))
            return false;
    }
    if (b.has_hi) {
        const int c = compare_decimal(v, b.hi);
        if (c > 0 || (c == 0 && b.hi_excl))
            return false;
    }
    return true;
}

}  // namespace imp
