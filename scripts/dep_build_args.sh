#!/usr/bin/env bash
# Emit --build-arg lines for the dependency pins defined once in
# cmake/imp-deps.cmake. Used by `make build` so the tags are not duplicated
# (inlining the sed in the Makefile breaks make's $(shell ...) paren matching).
set -euo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sed -n 's/^set(IMP_DEP_\([A-Z_]*\)_\(TAG\|SHA\)[ \t]*\([^ \t]*\).*/--build-arg IMP_DEP_\1_\2=\3/p' \
    "$dir/cmake/imp-deps.cmake"
