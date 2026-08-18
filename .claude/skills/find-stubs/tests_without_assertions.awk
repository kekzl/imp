# Gtest bodies that assert nothing. Splits on `^TEST...` / `^}` because a regex
# over the body stops at the first line-initial `}`, which any helper lambda or
# struct at the top of a test provides — that form flagged 184 of 2410 tests,
# nearly all of them false.
#
# The second condition is what makes the output readable: most tests here
# delegate to a harness (`run_test(...)`, `check_gemma_result(...)`) and assert
# inside it. Excluding that leaves 44 (2026-08-19), 21 of them in
# tests/test_fmha_fp8.cu, which has its own harness under a different name —
# so read by file, and add its harness here rather than reading 21 hits twice.
#
# Usage: awk -f tests_without_assertions.awk $(find tests -name '*.cpp' -o -name '*.cu')
/^TEST(_F|_P)?\(/ { inb=1; body=""; name=$0; next }
inb && /^\}/ {
  inb=0
  if (body !~ /EXPECT|ASSERT|SUCCEED|FAIL|GTEST_SKIP/ &&
      body !~ /(run|check|expect|verify|assert)[a-zA-Z_]*\(/) print FILENAME ": " name
  next
}
inb { body = body "\n" $0 }
