#pragma once

// Compile-time property tables keyed by a sparse enum (#2212): one row per enumerator,
// looked up by std::to_underlying, completeness proven by static_assert.

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>
#include <utility>

namespace imp::enum_table {

// True when V names an enumerator: GCC and Clang print "V = (E)4" for a value without one.
template <auto V>
consteval bool is_enumerator() {
    std::string_view sig = __PRETTY_FUNCTION__;
    size_t eq = sig.rfind("= ");
    return eq != std::string_view::npos && sig[eq + 2] != '(';
}

namespace probe_check {
enum class Probe : int { A = 0, B = 2 };
}  // namespace probe_check
static_assert(is_enumerator<probe_check::Probe::A>() && is_enumerator<probe_check::Probe::B>() &&
                  !is_enumerator<static_cast<probe_check::Probe>(1)>(),
              "enumerator probe does not recognise this compiler's __PRETTY_FUNCTION__");

// Every enumerator of E in [0, N) has exactly one row (field `id`), no row sits on a gap.
template <typename E, size_t N, typename Row, size_t R>
consteval bool rows_cover_enumerators(const std::array<Row, R>& rows) {
    constexpr auto is_enum = []<size_t... I>(std::index_sequence<I...>) {
        return std::array<bool, N>{is_enumerator<static_cast<E>(I)>()...};
    }(std::make_index_sequence<N>{});
    std::array<int, N> hits{};
    for (const auto& row : rows) {
        auto v = static_cast<size_t>(std::to_underlying(row.id));
        if (v >= N)
            return false;
        ++hits[v];
    }
    for (size_t v = 0; v < N; ++v) {
        if (hits[v] != (is_enum[v] ? 1 : 0))
            return false;
    }
    return true;
}

// Largest row id + 1: the index size.
template <typename Row, size_t R>
consteval size_t index_size(const std::array<Row, R>& rows) {
    size_t size = 0;
    for (const auto& row : rows)
        size = std::max(size, static_cast<size_t>(std::to_underlying(row.id)) + 1);
    return size;
}

// Dense index: slot v points at the row whose id has underlying value v, nullptr for gaps.
template <size_t Size, typename Row, size_t R>
consteval std::array<const Row*, Size> index_rows(const std::array<Row, R>& rows) {
    std::array<const Row*, Size> index{};
    for (const auto& row : rows)
        index[static_cast<size_t>(std::to_underlying(row.id))] = &row;
    return index;
}

// Row for `id`, or `unknown` for gaps and values past the index.
template <typename Row, size_t Size, typename E>
constexpr const Row& lookup(const std::array<const Row*, Size>& index, E id, const Row& unknown) {
    auto v = static_cast<size_t>(std::to_underlying(id));
    return v < Size && index[v] != nullptr ? *index[v] : unknown;
}

}  // namespace imp::enum_table
