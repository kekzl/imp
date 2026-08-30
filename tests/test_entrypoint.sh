#!/usr/bin/env bash
# docker-entrypoint.sh: env -> CLI argv translation.
#
# Runs the real script against a stub binary on PATH that prints its argv, so
# this needs no Docker, no GPU and no build. Wired into scripts/ci_static_gates.sh
# (group `entrypoint`), where the required `Build` check runs it.
#
# What it guards: the container env surface is the only place a compose
# deployment configures imp, it is hand-written, and until 2026-08-31 nothing
# ran it at all. One name (IMP_KV_FP8) had already inverted its meaning
# unnoticed - docs/plans/2026-08-29-qwen38-long-context-posture.md, trap 1.
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
ENTRYPOINT="$PWD/docker-entrypoint.sh"

STUB_DIR=$(mktemp -d)
trap 'rm -rf "$STUB_DIR"' EXIT

# Each argument is printed as <arg>, so a grep can assert adjacency and exact
# value rather than mere presence.
for name in imp-server imp-cli; do
    cat > "$STUB_DIR/$name" <<'STUB'
#!/usr/bin/env bash
out=""
for a in "$@"; do out="$out<$a>"; done
echo "$out"
STUB
    chmod +x "$STUB_DIR/$name"
done

PASS=0
FAIL=0

# run <env-assignments...> -- <entrypoint args...>; sets OUT and ERR.
run() {
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    local errfile="$STUB_DIR/err"
    OUT=$(env -i PATH="$STUB_DIR:/usr/bin:/bin" "${envs[@]}" \
              bash "$ENTRYPOINT" "$@" 2>"$errfile")
    ERR=$(cat "$errfile")
}

check() {  # check <label> <needle> present|absent <haystack>
    local label="$1" needle="$2" mode="$3" hay="$4"
    local hit=0
    case "$hay" in *"$needle"*) hit=1 ;; esac
    if { [ "$mode" = present ] && [ "$hit" = 1 ]; } ||
       { [ "$mode" = absent ] && [ "$hit" = 0 ]; }; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        printf '  FAIL  %s\n        wanted %s: %s\n        got: %s\n' \
               "$label" "$mode" "$needle" "$hay"
    fi
}

# --- the generic config surface -------------------------------------------

run IMP_SET="a.b=1 c.d=2" -- imp-server
check "IMP_SET: first pair"      '<--set><a.b=1>' present "$OUT"
check "IMP_SET: second pair"     '<--set><c.d=2>' present "$OUT"

run IMP_SET="$(printf 'a.b=1\nc.d=2\n')" -- imp-server
check "IMP_SET: newline separated" '<--set><a.b=1><--set><c.d=2>' present "$OUT"

run IMP_CONFIG=/etc/imp/imp.conf -- imp-server
check "IMP_CONFIG" '<--config></etc/imp/imp.conf>' present "$OUT"

# An unset or empty IMP_SET must not produce an empty --set: `--set ""` is a
# parse error at the far end, so this would break every existing deployment.
run -- imp-server
check "no IMP_SET means no --set"    '<--set>'    absent "$OUT"
check "no IMP_CONFIG means no --config" '<--config>' absent "$OUT"
run IMP_SET="" IMP_CONFIG="" -- imp-server
check "empty IMP_SET means no --set" '<--set>'    absent "$OUT"

# The unquoted expansion is wanted for word splitting, not for globbing. Run
# this one from a directory holding files that would match, so an unguarded
# loop hands imp filenames instead of the value.
GLOB_DIR="$STUB_DIR/glob"
mkdir -p "$GLOB_DIR" && : > "$GLOB_DIR/server.model_swap=a" && : > "$GLOB_DIR/server.model_swap=b"
OUT=$(cd "$GLOB_DIR" && env -i PATH="$STUB_DIR:/usr/bin:/bin" IMP_SET='server.model_swap=*' \
          bash "$ENTRYPOINT" imp-server 2>/dev/null)
check "IMP_SET value is not glob-expanded" '<--set><server.model_swap=*>' present "$OUT"
check "no filename leaked into --set"      'model_swap=a'                 absent  "$OUT"

# imp-cli takes the same two flags, and the entrypoint serves it too.
run IMP_SET="a.b=1" -- imp-cli
check "IMP_SET reaches imp-cli" '<--set><a.b=1>' present "$OUT"

# --- the frozen legacy names still translate -------------------------------

run IMP_KV_FP8=1 -- imp-server
check "IMP_KV_FP8=1"    '<--kv-fp8>' present "$OUT"
run IMP_KV_FP8=true -- imp-server
check "IMP_KV_FP8=true" '<--kv-fp8>' present "$OUT"
run IMP_KV_FP8=0 -- imp-server
check "IMP_KV_FP8=0"    '<--kv-fp8>' absent  "$OUT"

run IMP_MODEL=/models/m.gguf IMP_PORT=9090 IMP_THINK_BUDGET=0 -- imp-server
check "IMP_MODEL"        '<--model></models/m.gguf>' present "$OUT"
check "IMP_PORT"         '<--port><9090>'            present "$OUT"
check "IMP_THINK_BUDGET" '<--think-budget><0>'       present "$OUT"

run IMP_DECODE_NVFP4=2 -- imp-server
check "IMP_DECODE_NVFP4=2" '<--decode-nvfp4-only>' present "$OUT"
run IMP_DECODE_NVFP4=0 -- imp-server
check "IMP_DECODE_NVFP4=0" '<--no-nvfp4>' present "$OUT"

# Default host inside a container is 0.0.0.0 for imp-server only: the
# container's loopback is not reachable through a published port (#1619).
run -- imp-server
check "imp-server default host" '<--host><0.0.0.0>' present "$OUT"
run -- imp-cli
check "imp-cli gets no host"    '<--host>'          absent  "$OUT"

# --- silent precedence gets a line ----------------------------------------

# --kv-fp8 sets the dtype enum directly; the engine reads kv_cache.dtype only
# while that enum is still FP16. So the legacy name wins in either order and
# the IMP_SET pair is inert.
run IMP_KV_FP8=1 IMP_SET="kv_cache.dtype=nvfp4" -- imp-server
check "conflict is reported" 'override kv_cache.dtype' present "$ERR"
check "conflict still passes both through" '<--kv-fp8>' present "$OUT"

run IMP_SET="kv_cache.dtype=nvfp4" -- imp-server
check "no conflict without the legacy name" 'override kv_cache.dtype' absent "$ERR"
run IMP_KV_FP8=1 IMP_SET="runtime.max_batch_size=8" -- imp-server
check "no conflict on an unrelated key" 'override kv_cache.dtype' absent "$ERR"

# --- command dispatch ------------------------------------------------------

# A leading flag is meant for the default command, not exec'd as one.
run IMP_SET="a.b=1" -- --version
check "leading flag prepends imp-server" '<--set><a.b=1><--version>' present "$OUT"

# Anything that is not imp-server/imp-cli execs untouched - no env translation.
run IMP_MODEL=/models/m.gguf IMP_SET="a.b=1" -- echo hello
check "foreign command is not rewritten" 'hello' present "$OUT"
check "foreign command gets no flags"    '--set' absent  "$OUT"

echo "entrypoint: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
