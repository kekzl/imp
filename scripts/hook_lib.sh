# Sourced by scripts/pre-commit.hook and scripts/pre-push.hook. Needs $ROOT and $HOOK set.
# CPU step: make preflight (Docker, ~20 s). GPU step: only with IMP_HOOK_GPU=1.
# IMP_HOOK_DRYRUN=1 prints the GPU calls (free-card check, lock, command) instead of running them.

# make preflight, timed. Exit 1 on any FAIL row or when Docker is unreachable.
hook_preflight() {
    local t0=$SECONDS rc
    if ! docker info >/dev/null 2>&1; then
        echo "$HOOK: Docker daemon unreachable ('docker info' failed, DOCKER_HOST=${DOCKER_HOST:-unset})." >&2
        echo "$HOOK: make preflight runs in imp:lint; start Docker and retry." >&2
        return 1
    fi
    # pre-commit checks the staged tree; pre-push the working tree.
    local staged=0
    [ "$HOOK" = pre-commit ] && staged=1
    echo "$HOOK: make preflight (CPU, Docker)"
    PREFLIGHT_STAGED=$staged make -s -C "$ROOT" preflight; rc=$?
    echo "$HOOK: preflight exit $rc, $((SECONDS - t0)) s wall (incl. lint image check)"
    [ "$rc" -eq 0 ] || { echo "$HOOK: preflight failed (see table above), commit/push refused" >&2; return 1; }
}

# hook_gpu <command string>: runs it under require_free_gpu + gpu_lock when IMP_HOOK_GPU=1,
# else prints the command for an explicit run and returns 0.
hook_gpu() {
    local cmd="$1"
    if [ "${IMP_HOOK_GPU:-0}" != "1" ]; then
        echo "$HOOK: GPU step skipped (IMP_HOOK_GPU unset), run it yourself: $cmd"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "$HOOK: IMP_HOOK_GPU=1 but no nvidia-smi on this host" >&2
        return 1
    fi
    if [ "${IMP_HOOK_DRYRUN:-0}" = "1" ]; then
        echo "$HOOK: dry-run: $ROOT/scripts/require_free_gpu.sh '$HOOK (GPU)'"
        echo "$HOOK: dry-run: bash $ROOT/scripts/gpu_lock.sh acquire --pid $$ $HOOK"
        echo "$HOOK: dry-run: (cd $ROOT && $cmd)"
        echo "$HOOK: dry-run: bash $ROOT/scripts/gpu_lock.sh release --pid $$"
        return 0
    fi
    # Guarded: an older tree may lack these scripts and must still run.
    if [ -x "$ROOT/scripts/require_free_gpu.sh" ]; then
        "$ROOT/scripts/require_free_gpu.sh" "$HOOK (GPU)" || return 1
    fi
    # Lock holder = this hook's pid, released on exit; refused if another worktree holds it.
    if [ -f "$ROOT/scripts/gpu_lock.sh" ]; then
        bash "$ROOT/scripts/gpu_lock.sh" acquire --pid $$ "$HOOK" >/dev/null || return 1
        # shellcheck disable=SC2064 # expand ROOT and $$ now
        trap "bash '$ROOT/scripts/gpu_lock.sh' release --pid $$ >/dev/null" EXIT
    fi
    echo "$HOOK: GPU step: $cmd"
    (cd "$ROOT" && bash -c "$cmd")
}
