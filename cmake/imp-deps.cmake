# imp-deps.cmake — single source of truth for third-party dependency pins.
#
# These tags are mirrored by the Dockerfile deps-clone. `make build` extracts
# them from THIS file and injects them as --build-arg, so a version is bumped
# here in ONE place. (The Dockerfile keeps matching ARG defaults so a bare
# `docker build .` still works, but this file is authoritative.)
#
# Used by: CMakeLists.txt FetchContent_Declare GIT_TAG entries.

set(IMP_DEP_GOOGLETEST_TAG    v1.17.0  CACHE STRING "googletest git tag")
set(IMP_DEP_CUTLASS_TAG       v4.5.2   CACHE STRING "NVIDIA/cutlass git tag")
set(IMP_DEP_HTTPLIB_TAG       v0.48.0  CACHE STRING "cpp-httplib git tag")
set(IMP_DEP_NLOHMANN_JSON_TAG v3.12.0  CACHE STRING "nlohmann/json git tag")
