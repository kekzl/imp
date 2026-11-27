# Sourced by check_makefile_filters.sh and check_verify_filter.sh: list a gtest binary ONCE and
# count pattern matches on the cached names. One --gtest_list_tests costs 60 ms per module and
# 450 ms for the imp-tests wrapper; one run per pattern made the two guards 11.9 s and 8.5 s.

# gtest_names <exe> <out>: full names "Suite.Test" (TEST_P: "Prefix/Suite.Test/0"), one per line.
gtest_names() {
    "$1" --gtest_list_tests 2>/dev/null | awk '/^[^ ]/ { s = $1; next } /^  / { print s $1 }' > "$2"
}

# gtest_count <names> <exe> <pattern>: the number of tests `<exe> --gtest_list_tests
# --gtest_filter=<pattern>` lists. A positive pattern matches the cached names ('*' any string,
# '?' one character, the rest literal, as gtest's UnitTestOptions::MatchesFilter); a negative
# one ('-' prefix) runs the binary.
gtest_count() {
    case "$3" in
        -*) "$2" --gtest_list_tests --gtest_filter="$3" 2>/dev/null | grep -c '^  ' || true ;;
        *)
            _re=$(printf '%s' "$3" | sed -e 's/[][\.^$+(){}|]/\\&/g' -e 's/\*/.*/g' -e 's/?/./g')
            grep -cE "^${_re}\$" "$1" || true
            ;;
    esac
}
