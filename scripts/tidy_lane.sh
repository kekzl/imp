#!/usr/bin/env bash
# clang-tidy lane (#2210): host .cpp TUs + the host side of src/ .cu TUs, one process per
# file, nproc in parallel, then two hard checks:
#   1. every .cu TU parsed (a clang error yields no findings; #2219's first CI run: 89 of 89)
#   2. tools/check_tidy_findings.py: WarningsAsErrors checks of .clang-tidy over their pins
# Other checks print, never fail.
#
# Usage: scripts/tidy_lane.sh --all          # src/tools .cpp + every src .cu TU (make tidy)
#        scripts/tidy_lane.sh FILE...        # these files; a .cu fragment lints its includer
# Needs $TIDY_BUILD/compile_commands.json (cmake configure) and clang-tidy on PATH.
# Env: TIDY_BUILD (default build), TIDY_LOGS (default $TIDY_BUILD/tidy-logs), TIDY_JOBS (default nproc).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

B="${TIDY_BUILD:-build}"
LOGS="${TIDY_LOGS:-$B/tidy-logs}"
JOBS="${TIDY_JOBS:-$(nproc)}"
[ -f "$B/compile_commands.json" ] || { echo "tidy_lane: $B/compile_commands.json missing (configure first)"; exit 2; }

python3 tools/tidy_cu_db.py "$B/compile_commands.json" "$B/tidy-cu" || exit 2
cu_db_files() { grep -o '"file": "[^"]*"' "$B/tidy-cu/compile_commands.json" | cut -d'"' -f4 | sed "s|^$PWD/||; s|^/work/||; s|^/src/||"; }

# The src file that textually #include's $1 (a body fragment, not a TU), if any.
includer_of() { grep -rlF --include='*.cu' --include='*.cpp' "#include \"${1#src/}\"" src; }

cpp=(); cu=()
if [ "${1:-}" = "--all" ]; then
    mapfile -t cpp < <(find src tools -name '*.cpp' | sort | while read -r f; do includer_of "$f" >/dev/null || echo "$f"; done)
    mapfile -t cu < <(cu_db_files | grep '^src/' | sort -u)
else
    for f in "$@"; do
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
        esac
    done
    [ "${#cpp[@]}" -gt 0 ] && mapfile -t cpp < <(printf '%s\n' "${cpp[@]}" | sort -u)
    [ "${#cu[@]}" -gt 0 ] && mapfile -t cu < <(printf '%s\n' "${cu[@]}" | sort -u)
fi
n=$(( ${#cpp[@]} + ${#cu[@]} ))
[ "$n" -eq 0 ] && { echo "tidy_lane: no .cpp or src .cu to lint"; exit 0; }

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
