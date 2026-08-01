#!/usr/bin/env bash
# scripts/check-release.sh — pre-release sanity gate.
#
# Runs the cheap checks that should always pass on a publishable
# tree: doc links, secrets/path leaks, no accidentally tracked
# binaries, then defers to `make verify-fast` for build + test
# + perf + smoke.
#
# Exit code 0 if everything passes; non-zero otherwise.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

RED=$(tput setaf 1 || echo)
GRN=$(tput setaf 2 || echo)
YLW=$(tput setaf 3 || echo)
RST=$(tput sgr0 || echo)

FAIL=0
section() { echo; echo "${YLW}== $* ==${RST}"; }
pass()    { echo "${GRN}PASS${RST} $*"; }
fail()    { echo "${RED}FAIL${RST} $*"; FAIL=$((FAIL+1)); }

# ---------------------------------------------------------------- 1. doc links
# Every (./*.md or docs/*.md) link of the form ](path) — where path looks
# like a relative file — must resolve to a tracked file.
section "doc links"
LINKS=$(grep -rEho '\]\(([^)#]+\.(md|example|json|sh|cmake|h|cpp|cu))\)' \
        --include='*.md' \
        README.md CONTRIBUTING.md CHANGELOG.md docs/ 2>/dev/null \
        | sed -E 's/^\]\(//; s/\)$//' | sort -u || true)
BROKEN=0
while IFS= read -r link; do
    [ -z "$link" ] && continue
    # Skip URLs.
    case "$link" in http*|https*) continue ;; esac
    # Each .md doc is a different cwd; resolve against repo root for absolute
    # links, otherwise against each containing doc. Cheap heuristic: try repo
    # root + any docs/ subdir.
    if [ -e "$link" ] || [ -e "docs/$link" ] || [ -e "docs/audit/$link" ] || [ -e "$(dirname "$link")/$(basename "$link")" ]; then
        :
    else
        echo "  broken: $link"
        BROKEN=$((BROKEN+1))
    fi
done <<< "$LINKS"
[ "$BROKEN" -eq 0 ] && pass "all internal doc links resolve" \
                   || fail "$BROKEN broken internal doc link(s)"

# --------------------------------------------------------- 2. personal paths
# Flag real personal paths but allow the standard /home/user/ placeholder.
section "personal path leaks"
LEAKS=$(git ls-files \
        | grep -vE '^tests/fixtures/' \
        | xargs grep -lE '/home/[a-z]+/|/Users/[a-z]+/' 2>/dev/null \
        | xargs -r grep -lE '/home/(kekz|raph|raphael)/|/Users/(kekz|raph|raphael)/' 2>/dev/null || true)
if [ -z "$LEAKS" ]; then
    pass "no maintainer-username paths in tracked files"
else
    echo "$LEAKS" | sed 's/^/  /'
    fail "personal paths in tracked files"
fi

# ----------------------------------------------------------- 3. secret-shaped
section "credential-shaped strings"
SECRETS=$(git ls-files | xargs grep -lE 'sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}' 2>/dev/null || true)
if [ -z "$SECRETS" ]; then
    pass "no obviously-real API keys in tracked files"
else
    echo "$SECRETS" | sed 's/^/  /'
    fail "potential secrets in tracked files"
fi

# ------------------------------------------------- 4. tracked-but-ignorable
section "tracked binaries / build artefacts"
SUSPECTS=$(git ls-files | grep -E '\.(o|so|a|dylib|dll|fatbin|cubin|gguf|safetensors|bin|pt|onnx|nsys-rep|ncu-rep|sqlite|qdrep)$' || true)
if [ -z "$SUSPECTS" ]; then
    pass "no tracked build artefacts or model weights"
else
    echo "$SUSPECTS" | sed 's/^/  /'
    fail "tracked files match build/weight patterns"
fi

# ------------------------------------------------------- 5. license sanity
section "license"
if [ -f LICENSE ] && grep -q "MIT" LICENSE && grep -q "MIT" README.md; then
    pass "LICENSE present and README mentions MIT"
else
    fail "LICENSE missing or README license claim mismatched"
fi

# --------------------------------------------------- 5b. version consistency
# The version lives in three places that have to agree, and nothing pinned
# them: CMakeLists.txt is the source of truth, CHANGELOG.md must carry a
# released section for it, and docs/BENCHMARKS.md names the release its
# tabulated numbers were taken on. Bumping one and forgetting the others is a
# documented red flag in the shipping playbook — this makes it a failure.
section "version consistency"
CM_VER=$(sed -nE 's/^project\(imp .*VERSION ([0-9]+\.[0-9]+\.[0-9]+)\).*/\1/p' CMakeLists.txt | head -1)
CL_VER=$(grep -oE '^## \[[0-9]+\.[0-9]+\.[0-9]+\]' CHANGELOG.md | head -1 | tr -d '#[] ')
BM_VER=$(sed -nE 's/.*\*\*Toolchain \(current: `v([0-9]+\.[0-9]+\.[0-9]+)`\).*/\1/p' docs/BENCHMARKS.md | head -1)
if [ -z "$CM_VER" ]; then
    fail "no project(imp ... VERSION X.Y.Z) in CMakeLists.txt"
elif [ "$CM_VER" = "$CL_VER" ] && [ "$CM_VER" = "$BM_VER" ]; then
    pass "CMakeLists / CHANGELOG / BENCHMARKS all say $CM_VER"
else
    fail "version drift — CMakeLists '$CM_VER', CHANGELOG '${CL_VER:-none}', BENCHMARKS '${BM_VER:-none}'"
fi

# ----------------------------------------------- 6. defer to make verify-fast
section "make verify-fast"
if [ "${SKIP_VERIFY:-0}" = "1" ]; then
    echo "  (skipped via SKIP_VERIFY=1)"
else
    if make verify-fast >/tmp/imp_check_release_verify.log 2>&1; then
        pass "make verify-fast"
    else
        echo "  log: /tmp/imp_check_release_verify.log"
        tail -30 /tmp/imp_check_release_verify.log | sed 's/^/  /'
        fail "make verify-fast"
    fi
fi

# --------------------------------------------------------------------- end
echo
if [ "$FAIL" -eq 0 ]; then
    echo "${GRN}check-release: all gates passed${RST}"
    exit 0
else
    echo "${RED}check-release: $FAIL gate(s) failed${RST}"
    exit 1
fi
