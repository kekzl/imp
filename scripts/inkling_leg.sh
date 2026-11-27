#!/usr/bin/env bash
# inkling_leg.sh run|decide|--selftest: the Inkling leg of `make verify-fast` (B-142). It runs the
# whole `make verify-inkling` recipe (serve smoke + perf row) when the diff against the merge base
# with origin/main touches a path below; otherwise it prints one SKIP line naming the rule.
# Env (read here only): IMP_VERIFY_INKLING=1 forces it on, IMP_VERIFY_SKIP_INKLING=1 off,
# IMP_VERIFY_MODELS / IMP_VERIFY_MODEL_STORE locate Inkling-2-Mini (else $HOME/models).
set -uo pipefail

RULE="src/*inkling*, tests/*inkling*, scripts/inkling_*; KV, BlockPool, scheduler, src/memory/"

# inkling_path <path>: 0 + a reason when the path is Inkling's or a shared path that broke it
# (#3270: a KV/BlockPool change left Inkling serving 0 of 9 requests with every gate green).
inkling_path() {
    case "$1" in
        src/*inkling* | tests/*inkling* | scripts/inkling_*) echo "Inkling path $1" ;;
        src/memory/*) echo "allocation path $1" ;;
        src/*/kv_* | src/*/engine_kv_* | src/*/executor_kv_*) echo "KV path $1" ;;
        src/*block_pool* | src/*scheduler*) echo "BlockPool / scheduler path $1" ;;
        *) return 1 ;;
    esac
}

# decide: changed paths on stdin. Prints "on: <reason>" (exit 0) or "off: <reason>" (exit 1).
decide() {
    local p why
    if [ "${IMP_VERIFY_INKLING:-0}" = "1" ]; then
        echo "on: IMP_VERIFY_INKLING=1"; return 0
    fi
    if [ "${IMP_VERIFY_SKIP_INKLING:-0}" = "1" ]; then
        echo "off: IMP_VERIFY_SKIP_INKLING=1"; return 1
    fi
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        if why="$(inkling_path "$p")"; then echo "on: $why"; return 0; fi
    done
    echo "off: no changed path matches ($RULE)"
    return 1
}

# Changed paths: tracked edits against the merge base (committed or not) plus untracked files.
changed_paths() {
    git diff --name-only "$1"
    git ls-files --others --exclude-standard
}

# Model store to mount: the parent of the resolved Inkling-2-Mini dir (a symlink farm resolves to the store).
model_store() {
    local c d
    for c in "${IMP_VERIFY_MODELS:-}" "${IMP_VERIFY_MODEL_STORE:-}" "$HOME/models"; do
        [ -n "$c" ] && [ -d "$c/Inkling-2-Mini" ] || continue
        d="$(readlink -f "$c/Inkling-2-Mini")" && dirname "$d" && return 0
    done
    return 1
}

selftest() {
    local pass=0 bad=0
    expect() {  # expect <on|off> <label> <env assignment or -> <paths...>
        local want="$1" label="$2" env="$3" got
        shift 3
        if [ "$env" = "-" ]; then
            got="$(printf '%s\n' "$@" | env -u IMP_VERIFY_INKLING -u IMP_VERIFY_SKIP_INKLING bash "$0" decide)"
        else
            got="$(printf '%s\n' "$@" | env -u IMP_VERIFY_INKLING -u IMP_VERIFY_SKIP_INKLING "$env" bash "$0" decide)"
        fi
        if [ "${got%%:*}" = "$want" ]; then
            pass=$((pass + 1))
        else
            bad=$((bad + 1)); echo "selftest FAIL: $label: want $want, got '$got'"
        fi
    }
    expect on  "executor"          - docs/README.md src/exec/executor_inkling.cu
    expect on  "inkling kernel"    - src/compute/inkling_attn.cu
    expect on  "engine"            - src/runtime/engine_inkling.cpp
    expect on  "hf config"         - src/model/hf_config/inkling2.cpp
    expect on  "test"              - tests/test_inkling_slab_layout.cpp
    expect on  "bench test"        - tests/bench/test_inkling_kernels_bench.cu
    expect on  "leg scripts"       - scripts/inkling_perf.sh
    expect on  "block pool"        - src/memory/block_pool.cpp
    expect on  "allocation"        - src/memory/device_alloc_all.cpp
    expect on  "kv manager"        - src/memory/kv_cache_manager.cpp
    expect on  "kv init"           - src/runtime/engine_kv_cache_init.cpp
    expect on  "kv write"          - src/exec/executor_kv_write.cu
    expect on  "kv config"         - src/core/config/kv_cache.h
    expect on  "scheduler"         - src/runtime/scheduler.cpp
    expect on  "engine scheduler"  - src/runtime/engine_scheduler.cpp
    expect off "docs only"         - docs/ROADMAP.md docs/internals/BENCHMARKING.md README.md
    expect off "other kernel"      - src/compute/gemm.cu src/runtime/engine.cpp
    expect off "analysis tool"     - tools/analysis/inkling_kld_oracle.py
    expect off "empty diff"        -
    expect on  "forced"            IMP_VERIFY_INKLING=1 docs/ROADMAP.md
    expect off "skipped"           IMP_VERIFY_SKIP_INKLING=1 src/exec/executor_inkling.cu
    echo "selftest: $pass/$((pass + bad))"
    [ "$bad" -eq 0 ]
}

run() {
    local base verdict store t0 rc
    cd "$(dirname "$(readlink -f "$0")")/.." || return 1
    # No merge base: the leg runs, a gate that cannot see the diff does not go quiet.
    if [ "${IMP_VERIFY_SKIP_INKLING:-0}" != "1" ] && ! base="$(git merge-base HEAD origin/main 2>/dev/null)"; then
        verdict="on: no merge base with origin/main"
    # Not a pipe: decide returns at the first match, and pipefail then reads the writer's SIGPIPE as off.
    elif ! verdict="$(decide <<< "$(changed_paths "${base:-HEAD}")")"; then
        echo "SKIP Inkling leg (verify-inkling): ${verdict#off: }; IMP_VERIFY_INKLING=1 forces it"
        return 0
    fi
    if ! store="$(model_store)"; then
        echo "SKIP Inkling leg (verify-inkling): ${verdict#on: }, but no Inkling-2-Mini under" \
            "IMP_VERIFY_MODELS, IMP_VERIFY_MODEL_STORE or $HOME/models"
        return 0
    fi
    echo "Inkling leg (verify-inkling): ${verdict#on: }; model store $store"
    t0=$SECONDS
    ${MAKE:-make} --no-print-directory verify-inkling INKLING_MODELS="$store"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "PASS Inkling leg (verify-inkling, $((SECONDS - t0)) s)"
    else
        echo "FAIL Inkling leg (verify-inkling rc $rc, $((SECONDS - t0)) s)"
    fi
    return "$rc"
}

case "${1:-}" in
    run) run ;;
    decide) decide ;;
    --selftest) selftest ;;
    *) echo "usage: inkling_leg.sh run|decide|--selftest" >&2; exit 2 ;;
esac
