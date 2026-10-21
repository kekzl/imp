# Single source of truth for third-party dependency pins. Each dep carries a TAG (label) and
# a SHA (authoritative, what's built); check_dep_pins.sh --online catches a TAG/SHA mismatch.
# Bump TAG and SHA together, then make check-deps-online. Used by FetchContent_Declare in CMakeLists.txt.

set(IMP_DEP_GOOGLETEST_TAG    v1.18.0  CACHE STRING "googletest git tag")
set(IMP_DEP_CUTLASS_TAG       v4.8.1   CACHE STRING "NVIDIA/cutlass git tag")
set(IMP_DEP_HTTPLIB_TAG       v0.62.0  CACHE STRING "cpp-httplib git tag")
set(IMP_DEP_NLOHMANN_JSON_TAG v3.12.0  CACHE STRING "nlohmann/json git tag")
set(IMP_DEP_PCRE2_TAG         pcre2-10.49 CACHE STRING "PCRE2Project/pcre2 git tag")

set(IMP_DEP_GOOGLETEST_SHA    063de7e9578f82b369302001269680b4b1553359 CACHE STRING "googletest commit")
set(IMP_DEP_CUTLASS_SHA       2c85f497547eb441f0fe80bfb0fb2e6f9add1719 CACHE STRING "NVIDIA/cutlass commit")
set(IMP_DEP_HTTPLIB_SHA       74ef3c42188acd6391f027c4337fb0e1a68c0024 CACHE STRING "cpp-httplib commit")
set(IMP_DEP_NLOHMANN_JSON_SHA 55f93686c01528224f448c19128836e7df245f72 CACHE STRING "nlohmann/json commit")
set(IMP_DEP_PCRE2_SHA         6f9d7c1373262c541324a16a358785b33ef116cf CACHE STRING "PCRE2Project/pcre2 commit")
