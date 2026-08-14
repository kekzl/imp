#!/usr/bin/env bash
# Dependency-pin gate.
#
# Two failures this catches, both of which happened on 2026-08-14 and were
# invisible until a Docker layer cache went cold:
#
#   1. DRIFT. The pins live in cmake/imp-deps.cmake and are injected into the
#      Docker build by scripts/dep_build_args.sh, but the Dockerfile also
#      carries ARG defaults for anyone running `docker build` directly. Those
#      defaults had fallen a release behind (CUTLASS v4.6.2 vs the pinned
#      4.7.0), so the same tree built two different dependency sets depending on
#      how it was invoked. AGENTS.md says "bump both dep-pin sites together";
#      nothing enforced it.
#
#   2. A TAG THAT DOES NOT EXIST UPSTREAM. The CUTLASS pin read `4.7.0` while
#      every tag in that repo carries a `v` prefix. `git clone --branch 4.7.0`
#      fails, so every build from a cold cache died -- while cached builds kept
#      working and CI stayed green.
#
# Both checks read the real sources (cmake/imp-deps.cmake and the Dockerfile's
# own clone lines) rather than keeping a third copy of the truth here.
#
# usage: check_dep_pins.sh [--online]
#   default : drift check only, no network
#   --online: additionally resolve every tag against its upstream remote
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMAKE_FILE="$ROOT/cmake/imp-deps.cmake"
DOCKERFILE="$ROOT/Dockerfile"
ONLINE=0
[ "${1:-}" = "--online" ] && ONLINE=1

fail=0
note() { printf '%s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; fail=1; }

# name -> tag, from the single source of truth
declare -A PINNED
while IFS='=' read -r name tag; do
    [ -n "$name" ] && PINNED["$name"]="$tag"
done < <(sed -n 's/^set(IMP_DEP_\([A-Z_]*\)_TAG[ \t]*\([^ \t)]*\).*/\1=\2/p' "$CMAKE_FILE")

if [ ${#PINNED[@]} -eq 0 ]; then
    bad "no IMP_DEP_*_TAG entries parsed from cmake/imp-deps.cmake"
    exit 1
fi

# name -> ARG default, from the Dockerfile
declare -A ARGDEF
while IFS='=' read -r name tag; do
    [ -n "$name" ] && ARGDEF["$name"]="$tag"
done < <(sed -n 's/^ARG IMP_DEP_\([A-Z_]*\)_TAG=\(.*\)$/\1=\2/p' "$DOCKERFILE")

# name -> clone URL, from the Dockerfile's own clone lines
declare -A URL
while read -r name url; do
    [ -n "$name" ] && URL["$name"]="$url"
done < <(grep -oE '\-\-branch \$\{IMP_DEP_[A-Z_]+_TAG\}[[:space:]]+https://[^[:space:]]+' "$DOCKERFILE" \
         | sed -E 's/--branch \$\{IMP_DEP_([A-Z_]+)_TAG\}[[:space:]]+(https:\/\/[^[:space:]]+)/\1 \2/')

note "dependency pins (source: cmake/imp-deps.cmake)"
for name in $(printf '%s\n' "${!PINNED[@]}" | sort); do
    tag="${PINNED[$name]}"
    printf '  %-16s %s\n' "$name" "$tag"

    # 1. drift: the Dockerfile ARG default must match the pin exactly
    if [ -z "${ARGDEF[$name]:-}" ]; then
        bad "$name: pinned in cmake but no 'ARG IMP_DEP_${name}_TAG=' default in the Dockerfile"
    elif [ "${ARGDEF[$name]}" != "$tag" ]; then
        bad "$name: Dockerfile default '${ARGDEF[$name]}' != cmake pin '$tag' (bump both, AGENTS.md)"
    fi

    # 2. existence: the tag must resolve upstream, or a cold build cannot clone it
    if [ "$ONLINE" = "1" ]; then
        u="${URL[$name]:-}"
        if [ -z "$u" ]; then
            bad "$name: no clone URL found in the Dockerfile; cannot verify the tag exists"
        elif ! git ls-remote --exit-code "$u" "refs/tags/$tag" >/dev/null 2>&1 \
             && ! git ls-remote --exit-code "$u" "refs/heads/$tag" >/dev/null 2>&1; then
            bad "$name: '$tag' resolves to no tag or branch at $u (a cold 'git clone --branch' would fail here)"
        fi
    fi
done

# Every Dockerfile ARG must be backed by a pin, or it is a fourth source of truth.
for name in "${!ARGDEF[@]}"; do
    [ -z "${PINNED[$name]:-}" ] && bad "$name: Dockerfile ARG default with no matching pin in cmake/imp-deps.cmake"
done

if [ "$fail" = "0" ]; then
    note "OK ($([ "$ONLINE" = "1" ] && echo 'drift + upstream tags' || echo 'drift only; pass --online to resolve tags'))"
else
    note ""
    note "Pins are the one thing a cached Docker layer will happily hide."
fi
exit "$fail"
