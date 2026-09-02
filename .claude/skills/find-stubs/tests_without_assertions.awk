# Gtest bodies that assert nothing. Splits on `^TEST...` / `^}` because a regex
# over the whole body stops at the first line-initial `}`, which any helper
# lambda or struct at the top of a test provides - that form flagged 184 of
# 2410 tests, nearly all of them false.
#
# Most tests delegate to a harness (`run_test(...)`, `check_gemma_result(...)`)
# and assert inside it, so a call whose name starts with run/check/expect/
# verify/assert counts as an assertion. A harness named differently
# (`compare_dp4a_vs_mmvq`, `chunkpar_matches_fused`, `hd256_bench_vs_wmma`)
# still shows up: read it, do not grow the prefix list.
#
# Three bugs fixed 2026-09-02, together good for 32 of 53 hits:
#   - the harness class was [a-zA-Z_]*, so a harness with a DIGIT in its name
#     (`run_fa2`, `run_pv256`) never matched and every caller was flagged. That
#     is what the 21 hits in tests/test_fmha_fp8.cu were, not "its own harness
#     under a different name".
#   - a single-line `TEST(...) { run_x(); }` never sees a line-initial `}`, so
#     it stayed open and was judged against the NEXT block's body.
#   - a C++ raw string literal (`R"({ ... })"`, JSON fixtures) has lines that
#     start with `}` and ended the body early (5 tests in two files).
#
# Baseline after the fixes: 16 hits (2026-09-02, `339ce7c7`), all explained:
# 8 deliberate benchmarks (`Bench*`, they measure and print), 6 MMVQ cases on
# `compare_dp4a_vs_mmvq` (18 EXPECTs inside) and 2 GDN cases on
# `chunkpar_matches_fused` (18 ASSERTs inside). Zero real gaps at this commit.
#
# Usage: awk -f tests_without_assertions.awk $(find tests -name '*.cpp' -o -name '*.cu')

function verdict(name, body) {
  if (body !~ /EXPECT|ASSERT|SUCCEED|FAIL|GTEST_SKIP/ &&
      body !~ /(run|check|expect|verify|assert)[a-zA-Z0-9_]*\(/) print FILENAME ": " name
}

# Raw-string state, evaluated before the block rules. `was_raw` is what the
# CURRENT line is, so the closing `})";` line is still treated as literal.
{
  was_raw = in_raw
  if (in_raw) { if ($0 ~ /\)"/) in_raw = 0 }
  else if ($0 ~ /R"\(/ && $0 !~ /\)"/) in_raw = 1
}

!was_raw && /^TEST(_F|_P)?\(/ {
  name = $0
  if ($0 ~ /\}[ \t]*$/) { inb = 0; verdict(name, $0) }   # single-line body
  else { inb = 1; body = "" }
  next
}
inb && !was_raw && /^\}/ { inb = 0; verdict(name, body); next }
inb { body = body "\n" $0 }
