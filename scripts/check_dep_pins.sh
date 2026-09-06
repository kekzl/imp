#!/usr/bin/env bash
# Dependency-pin gate.
#
# Three failures this catches. The first two happened on 2026-08-14 and were
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
#   3. UPSTREAM MUTATION (AUDIT_arch_2026 H-8). Every pin used to be a mutable
#      ref: four tags, four `git clone --branch`, nine `uses:` action majors,
#      zero commit SHAs. A re-tag or a compromised action release changed what
#      the published image contains with nothing in this repo moving. Each dep
#      now carries a TAG *and* a SHA, the build fetches the SHA, and --online
#      asserts the tag still resolves to it -- that assertion failing is the
#      re-tag alarm. Actions are pinned to 40-hex SHAs, checked offline.
#
# All checks read the real sources (cmake/imp-deps.cmake, the Dockerfile's own
# ARG/fetch lines, the workflow files) rather than keeping a copy of the truth.
#
# usage: check_dep_pins.sh [--online]
#   default : drift + form checks, no network
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

# name -> tag / name -> sha, from the single source of truth
declare -A PINNED
declare -A PINNED_SHA
while IFS='=' read -r name tag; do
    [ -n "$name" ] && PINNED["$name"]="$tag"
done < <(sed -n 's/^set(IMP_DEP_\([A-Z_]*\)_TAG[ \t]*\([^ \t)]*\).*/\1=\2/p' "$CMAKE_FILE")
while IFS='=' read -r name sha; do
    [ -n "$name" ] && PINNED_SHA["$name"]="$sha"
done < <(sed -n 's/^set(IMP_DEP_\([A-Z_]*\)_SHA[ \t]*\([^ \t)]*\).*/\1=\2/p' "$CMAKE_FILE")

if [ ${#PINNED[@]} -eq 0 ]; then
    bad "no IMP_DEP_*_TAG entries parsed from cmake/imp-deps.cmake"
    exit 1
fi

# name -> ARG default, from the Dockerfile (tags and SHAs in one table)
declare -A ARGDEF
declare -A ARGDEF_SHA
while IFS='=' read -r name tag; do
    [ -n "$name" ] && ARGDEF["$name"]="$tag"
done < <(sed -n 's/^ARG IMP_DEP_\([A-Z_]*\)_TAG=\(.*\)$/\1=\2/p' "$DOCKERFILE")
while IFS='=' read -r name sha; do
    [ -n "$name" ] && ARGDEF_SHA["$name"]="$sha"
done < <(sed -n 's/^ARG IMP_DEP_\([A-Z_]*\)_SHA=\(.*\)$/\1=\2/p' "$DOCKERFILE")

# name -> clone URL, from the Dockerfile's own fetch lines
declare -A URL
while read -r name url; do
    [ -n "$name" ] && URL["$name"]="$url"
done < <(grep -oE 'pin[[:space:]]+https://[^[:space:]]+[[:space:]]+\$\{IMP_DEP_[A-Z_]+_TAG\}' "$DOCKERFILE" \
         | sed -E 's|pin[[:space:]]+(https://[^[:space:]]+)[[:space:]]+\$\{IMP_DEP_([A-Z_]+)_TAG\}|\2 \1|')

note "dependency pins (source: cmake/imp-deps.cmake)"
for name in $(printf '%s\n' "${!PINNED[@]}" | sort); do
    tag="${PINNED[$name]}"
    sha="${PINNED_SHA[$name]:-}"
    printf '  %-16s %-9s %s\n' "$name" "$tag" "${sha:-<none>}"

    # 1. every tag pin needs a commit pin, and it must look like one
    if [ -z "$sha" ]; then
        bad "$name: tag '$tag' has no IMP_DEP_${name}_SHA in cmake/imp-deps.cmake (H-8: a tag is a mutable ref)"
    elif ! [[ "$sha" =~ ^[0-9a-f]{40}$ ]]; then
        bad "$name: IMP_DEP_${name}_SHA='$sha' is not a 40-hex commit"
    fi

    # 2. drift: the Dockerfile ARG defaults must match the pins exactly
    if [ -z "${ARGDEF[$name]:-}" ]; then
        bad "$name: pinned in cmake but no 'ARG IMP_DEP_${name}_TAG=' default in the Dockerfile"
    elif [ "${ARGDEF[$name]}" != "$tag" ]; then
        bad "$name: Dockerfile tag default '${ARGDEF[$name]}' != cmake pin '$tag' (bump both, AGENTS.md)"
    fi
    if [ -z "${ARGDEF_SHA[$name]:-}" ]; then
        bad "$name: pinned in cmake but no 'ARG IMP_DEP_${name}_SHA=' default in the Dockerfile"
    elif [ -n "$sha" ] && [ "${ARGDEF_SHA[$name]}" != "$sha" ]; then
        bad "$name: Dockerfile SHA default '${ARGDEF_SHA[$name]}' != cmake pin '$sha' (bump both, AGENTS.md)"
    fi

    # 3. the tag must still resolve, and to the pinned commit: a re-tag upstream
    #    is exactly the mutation the SHA pin defends against, and this is the
    #    only place it becomes visible.
    if [ "$ONLINE" = "1" ]; then
        u="${URL[$name]:-}"
        if [ -z "$u" ]; then
            bad "$name: no fetch URL found in the Dockerfile; cannot verify the tag"
            continue
        fi
        remote_sha="$(git ls-remote "$u" "refs/tags/$tag^{}" "refs/tags/$tag" "refs/heads/$tag" 2>/dev/null \
                      | grep -m1 -E "refs/tags/$tag\^\{\}$" | cut -f1)"
        [ -z "$remote_sha" ] && remote_sha="$(git ls-remote "$u" "refs/tags/$tag" "refs/heads/$tag" 2>/dev/null | head -1 | cut -f1)"
        if [ -z "$remote_sha" ]; then
            bad "$name: '$tag' resolves to no tag or branch at $u (a cold fetch would fail here)"
        elif [ -n "$sha" ] && [ "$remote_sha" != "$sha" ]; then
            bad "$name: RE-TAGGED upstream. '$tag' at $u is now $remote_sha, pinned $sha. The build still uses the pinned commit; decide, then bump both lines."
        fi
    fi
done

# Every Dockerfile ARG must be backed by a pin, or it is a fourth source of truth.
for name in "${!ARGDEF[@]}"; do
    [ -z "${PINNED[$name]:-}" ] && bad "$name: Dockerfile ARG tag default with no matching pin in cmake/imp-deps.cmake"
done
for name in "${!ARGDEF_SHA[@]}"; do
    [ -z "${PINNED_SHA[$name]:-}" ] && bad "$name: Dockerfile ARG SHA default with no matching pin in cmake/imp-deps.cmake"
done

# GitHub Actions: a major tag (actions/checkout@v7) is a mutable ref owned by
# someone else and runs with this repo's token. Offline, textual, no network.
note ""
note "action pins (source: .github/workflows/)"
n_actions=0
while IFS=: read -r file line ref; do
    n_actions=$((n_actions + 1))
    case "$ref" in
        ./*|docker://*) continue ;;  # local composite action / image, no ref to pin
    esac
    if ! [[ "$ref" =~ @[0-9a-f]{40}$ ]]; then
        bad "$(basename "$file"):$line uses '$ref' - pin it to a 40-hex commit SHA (Dependabot rewrites SHA pins with a version comment)"
    fi
done < <(grep -rnE '^[[:space:]]*(-[[:space:]]+)?uses:' "$ROOT/.github/workflows/" \
         | sed -E 's/^([^:]+):([0-9]+):[[:space:]]*(-[[:space:]]+)?uses:[[:space:]]*/\1:\2:/' \
         | sed -E 's/[[:space:]]*#.*$//')
if [ "$n_actions" = "0" ]; then
    bad "no 'uses:' lines parsed from .github/workflows/ (parser broken?)"
else
    printf '  %d uses: lines\n' "$n_actions"
fi

if [ "$fail" = "0" ]; then
    note "OK ($([ "$ONLINE" = "1" ] && echo 'drift + form + upstream tag->commit' || echo 'drift + form; pass --online to resolve tags'))"
else
    note ""
    note "Pins are the one thing a cached Docker layer will happily hide."
fi
exit "$fail"
