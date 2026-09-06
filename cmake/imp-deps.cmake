# imp-deps.cmake - single source of truth for third-party dependency pins.
#
# Each dep carries a TAG (the human label) and a SHA (what is actually built).
# The SHA is authoritative: CMakeLists.txt hands it to FetchContent as GIT_TAG
# and the Dockerfile fetches that commit directly, so an upstream re-tag cannot
# change what this tree builds without a line moving here. The TAG only labels
# it; `check_dep_pins.sh --online` resolves TAG against SHA, and that mismatch
# IS the re-tag alarm (AUDIT_arch_2026 H-8).
#
# These pins are mirrored by the Dockerfile deps-clone. `make build` extracts
# them from THIS file and injects them as --build-arg, so a version is bumped
# here in ONE place. (The Dockerfile keeps matching ARG defaults so a bare
# `docker build .` still works, but this file is authoritative.)
#
# Bumping a dep: change TAG and SHA together, then `make check-deps-online`.
#
# Used by: CMakeLists.txt FetchContent_Declare GIT_TAG entries.

set(IMP_DEP_GOOGLETEST_TAG    v1.18.0  CACHE STRING "googletest git tag")
set(IMP_DEP_CUTLASS_TAG       v4.7.0   CACHE STRING "NVIDIA/cutlass git tag")
set(IMP_DEP_HTTPLIB_TAG       v0.53.0  CACHE STRING "cpp-httplib git tag")
set(IMP_DEP_NLOHMANN_JSON_TAG v3.12.0  CACHE STRING "nlohmann/json git tag")

set(IMP_DEP_GOOGLETEST_SHA    063de7e9578f82b369302001269680b4b1553359 CACHE STRING "googletest commit")
set(IMP_DEP_CUTLASS_SHA       dcf215af68a2d08d305076c152a06f201728cd53 CACHE STRING "NVIDIA/cutlass commit")
set(IMP_DEP_HTTPLIB_SHA       f00e476f1b2d519343e960f77f57a06c8a24f046 CACHE STRING "cpp-httplib commit")
set(IMP_DEP_NLOHMANN_JSON_SHA 55f93686c01528224f448c19128836e7df245f72 CACHE STRING "nlohmann/json commit")
