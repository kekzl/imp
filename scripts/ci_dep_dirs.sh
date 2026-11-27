#!/bin/sh
# Prints -DFETCHCONTENT_SOURCE_DIR_<DEP>=<root>/<dir> for every dependency the CI image pre-cloned at
# the commit cmake/imp-deps.cmake pins (<root>/PINS lines "<dir> <sha>", .github/ci-image/Dockerfile).
# A missing or stale clone prints nothing for that dep (FetchContent fetches it) and one warning.
# Usage: cmake ... $(sh scripts/ci_dep_dirs.sh)   Root: $IMP_CI_DEPS, default /deps.
root="${IMP_CI_DEPS:-/deps}"
pins="$(dirname "$0")/../cmake/imp-deps.cmake"
for spec in GOOGLETEST:googletest HTTPLIB:httplib NLOHMANN_JSON:json PCRE2:pcre2; do
    var="${spec%%:*}"
    dir="${spec#*:}"
    want="$(sed -nE "s/^set\(IMP_DEP_${var}_SHA[[:space:]]+([0-9a-f]{40}).*/\1/p" "$pins")"
    have="$(awk -v d="$dir" '$1 == d { print $2 }' "$root/PINS" 2>/dev/null)"
    if [ -n "$want" ] && [ "$want" = "$have" ] && [ -d "$root/$dir" ]; then
        printf ' -DFETCHCONTENT_SOURCE_DIR_%s=%s/%s' "$var" "$root" "$dir"
    else
        echo "::warning::$dir: image clone ${have:-<none>} != pin ${want:-<none>}, FetchContent fetches it" >&2
    fi
done
echo
