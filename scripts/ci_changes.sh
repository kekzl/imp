#!/bin/sh
# CI change detection, one copy for ci.yml `Changes` (picks Build's container) and `Build` (gates its steps).
# Usage: ci_changes.sh <base sha>. Writes code= and cpp= to $GITHUB_OUTPUT (stdout without it).
# FAIL-OPEN: an unresolvable base or an empty diff gives code=true cpp=true; a failed git diff exits 1.
#
# POSIX sh: `Build` runs it under the CUDA image's `sh -e`, where a bashism fails the one
# required check (ruleset 14716423).
base="${1:-}"
out="${GITHUB_OUTPUT:-/dev/stdout}"
code=true
# `cpp` gates the clang-tidy steps, which lint changed src|tools|tests *.cpp, the host side of
# changed src/*.cu, tests/*.cu (#2210) and up to 3 TUs per changed src|tools|tests|include header
# (scripts/tidy_lane.sh). Same fail-open rule as `code`.
cpp=true
files=""
# Say so when the fail-open fires: a permanently open gate printed a normal-looking code=true and
# an empty file list, the same as a genuine code PR.
if [ -z "$base" ] || ! git cat-file -e "${base}^{commit}" 2>/dev/null; then
    echo "::warning::change detection fell open (base '${base:-<unset>}' unresolvable) - every gated step runs"
elif files="$(git diff --name-only "$base" HEAD)"; then
    # code stays true unless EVERY changed file is markdown, docs/, .claude/ or a workflow other
    # than ci.yml: none is a build input; 4/69 PR runs compiled 91-739 s for them.
    if [ -n "$files" ] && ! printf '%s\n' "$files" | grep -qvE '(^\.github/|^scripts/|\.md$|^docs/|^\.claude/|^\.github/workflows/(codeql|auto-merge|roofline|release-docker|nightly)\.yml$)'; then
        code=false
    fi
    # `grep -c`, not `grep -q` (#1499: -q exits at the first match, the producer takes EPIPE and
    # pipefail reports a failed pipeline for a match), and no here-string (a bashism).
    # `|| true`: grep exits 1 on a count of 0.
    n_cpp="$(printf '%s\n' "$files" | grep -cE '^(src|tools|tests)/.*\.cpp$|^(src|tests)/.*\.cu$|^(src|tools|tests|include)/.*\.(h|hpp|cuh)$' || true)"
    [ "${n_cpp:-0}" -gt 0 ] || cpp=false
else
    echo "::error::git diff against base $base failed"
    exit 1
fi
echo "code=$code" >> "$out"
echo "cpp=$cpp" >> "$out"
printf 'non-docs changes: code=%s (src|tools|tests .cpp, src|tests .cu or header: %s)\nchanged files:\n%s\n' \
    "$code" "$cpp" "${files:-<none>}"
