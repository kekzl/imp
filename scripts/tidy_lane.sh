#!/usr/bin/env bash
# clang-tidy lane (#2210): host .cpp TUs + the host side of src/ and tests/ .cu TUs, one process per
# file, nproc in parallel, then two hard checks:
#   1. every .cu TU parsed (a clang error yields no findings; #2219's first CI run: 89 of 89)
#   2. tools/check_tidy_findings.py: WarningsAsErrors checks of .clang-tidy over their pins
# Other checks print, never fail.
#
# Usage: scripts/tidy_lane.sh --all          # src/tools/tests .cpp + every src, tests .cu TU (make tidy)
#        scripts/tidy_lane.sh FILE...        # these files; a .cu fragment lints its includer, a header
#                                            # (.h .hpp .cuh under src|tests|tools|include) <= 3 TUs including it
#        scripts/tidy_lane.sh --map FILE...  # print the TUs FILE... lint, run nothing (no build needed)
#        scripts/tidy_lane.sh --selftest     # the header -> TU mapping on a fixture tree
# Needs $TIDY_BUILD/compile_commands.json (cmake configure) and clang-tidy on PATH.
# Env: TIDY_BUILD (default build), TIDY_LOGS (default $TIDY_BUILD/tidy-logs), TIDY_JOBS (default nproc),
#      TIDY_HDR_TUS (default 3): TUs per changed header.
set -uo pipefail
SELF="$(readlink -f "$0")"
cd "$(dirname "$SELF")/.." || exit 1

# The src file that textually #include's $1 (a body fragment, not a TU), if any.
includer_of() { grep -rlF --include='*.cu' --include='*.cpp' "#include \"${1#src/}\"" src; }

# Files under src|tests|tools|include whose #include of $1's basename resolves to $1: "compute/x.cuh"
# or <imp/imp.h> (an -I root), "x.cuh" (same directory) or "../tools/y.h" (relative). Sorted.
includers_any() {
    local h base re c s r
    h="$(realpath -m "$1")"
    base="${1##*/}"
    re="#include [\"<]([^\">]*/)?${base//./\\.}[\">]"
    while IFS= read -r c; do
        while IFS= read -r s; do
            for r in "$(dirname "$c")" src tests tools include; do
                [ "$(realpath -m "$r/$s")" = "$h" ] && { echo "$c"; break 2; }
            done
        done < <(grep -oE "$re" "$c" | sed -E 's/^#include .//; s/.$//')
    done < <(grep -rlE --include='*.h' --include='*.hpp' --include='*.cuh' --include='*.cu' --include='*.cpp' \
        "$re" src tests tools include 2>/dev/null | sort)
}

# Lane TUs (.cpp under src|tools|tests, .cu under src|tests) that include header $1: the direct
# includers, else the nearest level of transitive includers that has one (breadth-first, 8 levels).
# At most TIDY_HDR_TUS, src/ then tools/ then tests/: check_tidy_findings.py counts a header finding
# once whichever TU reports it, so one includer suffices; 3 cover #if-guarded parts. Exit 1: no TU.
# DBSET (the .cu files of the tidy database), when set, drops a .cu the lane cannot lint.
header_tus() {
    local -A seen=(["$1"]=1)
    local -a frontier=("$1") next tus
    local depth h c
    for ((depth = 0; depth < 8 && ${#frontier[@]} > 0; depth++)); do
        next=(); tus=()
        for h in "${frontier[@]}"; do
            while IFS= read -r c; do
                [ -n "${seen[$c]:-}" ] && continue
                seen[$c]=1
                case "$c" in
                    src/*.cu | tests/*.cu)
                        [ -z "${DBSET:-}" ] || grep -qxF "$c" <<< "$DBSET" && tus+=("$c") ;;
                    src/*.cpp | tools/*.cpp | tests/*.cpp) tus+=("$c") ;;
                    *.h | *.hpp | *.cuh) next+=("$c") ;;
                esac
            done < <(includers_any "$h")
        done
        if [ "${#tus[@]}" -gt 0 ]; then
            # src/ first: tests/.clang-tidy turns off checks a src/ header must pass.
            mapfile -t tus < <(printf '%s\n' "${tus[@]}" | sort -u |
                awk '{ print (/^src\//) ? 0 : (/^tools\//) ? 1 : 2, $0 }' | sort -k1,1n -k2 | cut -d' ' -f2)
            printf '%s\n' "${tus[@]:0:${TIDY_HDR_TUS:-3}}"
            return 0
        fi
        frontier=("${next[@]}")
    done
    return 1
}

cpp=(); cu=()
# Sorts FILE $1 into cpp / cu: a TU as is, a fragment as its includer, a header as header_tus.
add_file() {
    local f="$1" inc t
    local -a tus
    case "$f" in
        *.cpp)
            # A .cpp #include'd into another body is not a TU: lint its includer instead (#2209).
            if inc="$(includer_of "$f")"; then
                mapfile -t -O "${#cpp[@]}" cpp < <(printf '%s\n' "$inc")
            else
                cpp+=("$f")
            fi ;;
        src/*.cu)
            # A .cu #include'd into another body is not a TU: lint its includer instead.
            if inc="$(grep -rlF --include='*.cu' "#include \"${f#src/}\"" src)"; then
                mapfile -t -O "${#cu[@]}" cu < <(printf '%s\n' "$inc")
            else
                cu+=("$f")
            fi ;;
        tests/*.cu) cu+=("$f") ;;  # tests/ includes no .cu fragment
        src/*.h | src/*.hpp | src/*.cuh | tests/*.h | tests/*.hpp | tests/*.cuh | tools/*.h | tools/*.hpp | \
            tools/*.cuh | include/*.h | include/*.hpp)
            # A header has no TU of its own; its findings show through the TUs that include it.
            if mapfile -t tus < <(header_tus "$f") && [ "${#tus[@]}" -gt 0 ]; then
                for t in "${tus[@]}"; do add_file "$t"; done
            else
                echo "tidy_lane: $f: no lane TU includes it, not linted"
            fi ;;
    esac
}

if [ "${1:-}" = "--selftest" ]; then
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    cd "$tmp" || exit 2
    mkdir -p src/a src/c1 src/c2 src/d src/e src/m src/p src/w tests tools/q tools/r include/imp
    put() { printf '%s\n' "${@:2}" > "$1"; }
    put src/a/x.cuh '#pragma once'
    put src/a/y.cuh '#include "x.cuh"'
    put src/k.cu '#include "a/y.cuh"'
    put src/d/d.cuh '#pragma once'
    for t in src/d/t1.cu src/d/t2.cu src/e/t3.cpp src/e/t4.cpp; do put "$t" '#include "d/d.cuh"'; done
    put src/m/m.h '#pragma once'
    put src/m/hdr.h '#include "m.h"'
    put src/z.cu '#include "m/hdr.h"'
    put tests/t_m.cpp '#include "m/m.h"'
    put tools/q/q.h '#pragma once'
    put tests/t_q.cpp '#include "../tools/q/q.h"'
    put src/c1/same.h '#pragma once'
    put src/c2/same.h '#pragma once'
    put src/c2/u.cu '#include "same.h"'
    put src/w/w.cuh '#pragma once'
    put tools/r/r.cu '#include "w/w.cuh"'
    put include/imp/api.h '#pragma once'
    put src/api.cpp '#include <imp/api.h>'
    put src/p/p.h '#pragma once'
    for t in tests/t_p1.cpp tests/t_p2.cpp tests/t_p3.cpp src/zz.cpp; do put "$t" '#include "p/p.h"'; done
    put src/f.cu '#include "a/frag.cu"'
    put src/a/frag.cu '#include "a/x.cuh"'
    st_pass=0; st_fail=0
    check() {  # $1 label, $2 file, $3 expected TUs (space separated)
        local got
        got="$(cpp=(); cu=(); add_file "$2" > /dev/null; printf '%s\n' "${cpp[@]}" "${cu[@]}" | grep -v '^$' | sort -u | xargs)"
        if [ "$got" = "$3" ]; then st_pass=$((st_pass + 1)); else
            st_fail=$((st_fail + 1)); echo "FAIL $1: $2 -> '$got', want '$3'"; fi
    }
    check "direct includers, capped at 3" src/d/d.cuh "src/d/t1.cu src/d/t2.cu src/e/t3.cpp"
    check "transitive via a same-dir include" src/a/y.cuh "src/k.cu"
    check "direct TU and fragment includer first" src/a/x.cuh "src/f.cu"
    check "a direct TU beats a transitive one" src/m/m.h "tests/t_m.cpp"
    check "src/ TUs before tests/ ones" src/p/p.h "src/zz.cpp tests/t_p1.cpp tests/t_p2.cpp"
    check "relative ../ include" tools/q/q.h "tests/t_q.cpp"
    check "same basename, other dir: no match" src/c1/same.h ""
    check "same basename, same dir" src/c2/same.h "src/c2/u.cu"
    check "tools/ .cu is not a lane TU" src/w/w.cuh ""
    check "include/imp header" include/imp/api.h "src/api.cpp"
    check "a TU passes through" src/k.cu "src/k.cu"
    printf 'selftest: %d/%d\n' "$st_pass" "$((st_pass + st_fail))"
    [ "$st_fail" -eq 0 ]
    exit
fi

if [ "${1:-}" = "--map" ]; then
    shift
    for f in "$@"; do add_file "$f"; done
    printf '%s\n' "${cpp[@]}" "${cu[@]}" | grep -v '^$' | sort -u
    exit 0
fi

B="${TIDY_BUILD:-build}"
LOGS="${TIDY_LOGS:-$B/tidy-logs}"
JOBS="${TIDY_JOBS:-$(nproc)}"
[ -f "$B/compile_commands.json" ] || { echo "tidy_lane: $B/compile_commands.json missing (configure first)"; exit 2; }

python3 tools/tidy_cu_db.py "$B/compile_commands.json" "$B/tidy-cu" || exit 2
cu_db_files() { grep -o '"file": "[^"]*"' "$B/tidy-cu/compile_commands.json" | cut -d'"' -f4 | sed "s|^$PWD/||; s|^/work/||; s|^/src/||"; }

if [ "${1:-}" = "--all" ]; then
    mapfile -t cpp < <(find src tools tests -name '*.cpp' | sort | while read -r f; do includer_of "$f" >/dev/null || echo "$f"; done)
    mapfile -t cu < <(cu_db_files | grep -E '^(src|tests)/' | sort -u)
else
    DBSET="$(cu_db_files)"
    for f in "$@"; do add_file "$f"; done
    [ "${#cpp[@]}" -gt 0 ] && mapfile -t cpp < <(printf '%s\n' "${cpp[@]}" | sort -u)
    [ "${#cu[@]}" -gt 0 ] && mapfile -t cu < <(printf '%s\n' "${cu[@]}" | sort -u)
fi
n=$(( ${#cpp[@]} + ${#cu[@]} ))
[ "$n" -eq 0 ] && { echo "tidy_lane: no .cpp or src/tests .cu to lint"; exit 0; }

# A .cu that is not in the database would lint nothing.
dbset="$(cu_db_files)"
for f in "${cu[@]}"; do
    # Here-string, not `printf | grep -q`: grep -q exits at the match, printf takes EPIPE and
    # pipefail fails a found file (#1499; hit 1 of 141 on a full run).
    grep -qxF "$f" <<< "$dbset" || { echo "tidy_lane: $f is not in $B/tidy-cu/compile_commands.json"; exit 2; }
done

rm -rf "$LOGS"; mkdir -p "$LOGS"
rc_file="$LOGS/.rc"; : > "$rc_file"
echo "tidy_lane: ${#cpp[@]} .cpp + ${#cu[@]} .cu (host side), $JOBS jobs"
{ for f in "${cpp[@]}"; do echo "$B $f"; done; for f in "${cu[@]}"; do echo "$B/tidy-cu $f"; done; } | grep -v ' $' |
    xargs -P "$JOBS" -L 1 sh -c 'clang-tidy -p "$0" --warnings-as-errors="-*" "$1" \
        > "'"$LOGS"'/$(echo "$1" | tr / _).log" 2>&1; echo "$? $1" >> "'"$rc_file"'"'

ran=$(wc -l < "$rc_file")
nonzero=$(grep -vc '^0 ' "$rc_file" || true)
shopt -s nullglob
cu_logs=("$LOGS"/*.cu.log)
broken=$( { grep -l 'clang-diagnostic-error' "${cu_logs[@]}" /dev/null || true; } | wc -l)
# Advisory output: every finding, minus per-TU progress and header-count noise.
cat "$LOGS"/*.log | grep -vE '^\[[0-9]+/[0-9]+\] Processing file|^[0-9]+ warnings? generated\.$|^Suppressed [0-9]+ warnings|^Use -header-filter' || true
echo "tidy_lane: $ran of $n run, $nonzero nonzero exit, ${#cu_logs[@]} .cu log(s), $broken with a clang error"
FAIL=0
[ "$ran" -eq "$n" ] && [ "$nonzero" -eq 0 ] || { echo "tidy_lane: FAIL clang-tidy exit codes"; grep -v '^0 ' "$rc_file"; FAIL=1; }
[ "$broken" -eq 0 ] || { echo "tidy_lane: FAIL .cu TU(s) with a clang error"; grep -l 'clang-diagnostic-error' "${cu_logs[@]}"; FAIL=1; }
python3 tools/check_tidy_findings.py --logs "$LOGS" || FAIL=1
exit "$FAIL"
