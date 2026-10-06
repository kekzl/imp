# Single source of truth for third-party dependency pins. Each dep carries a TAG (label) and
# a SHA (authoritative, what's built); check_dep_pins.sh --online catches a TAG/SHA mismatch.
# Bump TAG and SHA together, then make check-deps-online. Used by FetchContent_Declare in CMakeLists.txt.

set(IMP_DEP_GOOGLETEST_TAG    v1.18.0  CACHE STRING "googletest git tag")
set(IMP_DEP_CUTLASS_TAG       v4.8.0   CACHE STRING "NVIDIA/cutlass git tag")
set(IMP_DEP_HTTPLIB_TAG       v0.59.0  CACHE STRING "cpp-httplib git tag")
set(IMP_DEP_NLOHMANN_JSON_TAG v3.12.0  CACHE STRING "nlohmann/json git tag")

set(IMP_DEP_GOOGLETEST_SHA    063de7e9578f82b369302001269680b4b1553359 CACHE STRING "googletest commit")
set(IMP_DEP_CUTLASS_SHA       098de2a652cf8f00fd70b2df54051c7eccbb855a CACHE STRING "NVIDIA/cutlass commit")
set(IMP_DEP_HTTPLIB_SHA       cf3693cb5cc0d39b0e6f4122ba89bda9edb5ac6b CACHE STRING "cpp-httplib commit")
set(IMP_DEP_NLOHMANN_JSON_SHA 55f93686c01528224f448c19128836e7df245f72 CACHE STRING "nlohmann/json commit")
