#!/bin/bash
# Advisory GPU lock: one holder per card, owner = worktree. Hooks, make GPU targets and
# require_free_gpu.sh refuse while another owner holds it; nothing waits.
# Usage:
#   gpu_lock.sh acquire [--pid PID] <purpose>  hold the card (no --pid: detached keeper, TTL)
#   gpu_lock.sh release [--pid PID] [--force]  drop it (--pid: only if PID is the holder)
#   gpu_lock.sh status                         print the holder line or "free"
#   gpu_lock.sh check                          exit 1 + holder line if another owner holds it
#   gpu_lock.sh run <purpose> -- <cmd...>      hold it for the duration of cmd
# Env: IMP_GPU_LOCK (file, default GPU_LOCK_FILE, else /tmp/gpu-lock), IMP_GPU_LOCK_OWNER
# (default: git toplevel of $PWD), IMP_GPU_LOCK_TTL (keeper lifetime in s, default 14400),
# IMP_GPU_LOCK_LEGACY (old per-repo lock, read only, default /tmp/imp-gpu.lock).
# Exit: 0 ok, 1 held by another owner, 2 usage error.
#
# Same file, mutex and record format as ~/.claude/skills/gpu-stats/gpu-lock.sh:
# key=value lines; a record with pid= is held while that pid (same start time) lives,
# one without pid= until started_epoch + expected_minutes*60. Mutex: flock on $LOCK.mutex.
set -uo pipefail

LOCK="${IMP_GPU_LOCK:-${GPU_LOCK_FILE:-/tmp/gpu-lock}}"
LEGACY="${IMP_GPU_LOCK_LEGACY:-/tmp/imp-gpu.lock}"
HIST="$LOCK.history"
TTL="${IMP_GPU_LOCK_TTL:-14400}"
OWNER="${IMP_GPU_LOCK_OWNER:-$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)}"

die() { echo "gpu_lock: $*" >&2; exit 2; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Field 22 of /proc/<pid>/stat (start time in ticks): a recycled pid does not match it.
# A zombie (state Z, unreaped under a PID 1 that never waits) counts as gone.
start_of() {
    local s
    s="$(sed 's/.*) //' "/proc/$1/stat" 2>/dev/null)" || return 1
    [ -n "$s" ] && [ "${s%% *}" != Z ] || return 1
    awk '{print $20}' <<<"$s"
}

field_of() { sed -n "s/^$2=//p" "$1" 2>/dev/null | head -1; }
field() { field_of "$LOCK" "$1"; }

line_of() {  # line_of <file>
    printf 'pid=%s worktree=%s purpose="%s" since=%s holder=%s\n' \
        "$(field_of "$1" pid)" "$(field_of "$1" worktree)" "$(field_of "$1" purpose)" \
        "$(field_of "$1" since)" "$(field_of "$1" holder)"
}
holder_line() { line_of "$LOCK"; }

# 0 = the record in <file> is held. pid= set: that pid alive with the same start time.
# No pid=: now < started_epoch + expected_minutes*60; unparseable = expired.
held_in() {
    local f="$1" pid start cur ep mins
    [ -s "$f" ] || return 1
    pid="$(field_of "$f" pid)"
    if [ -n "$pid" ]; then
        case "$pid" in *[!0-9]*) return 1 ;; esac
        start="$(field_of "$f" start)"
        cur="$(start_of "$pid")" || return 1
        [ -z "$start" ] || [ "$start" = "$cur" ]
        return
    fi
    ep="$(field_of "$f" started_epoch)"; mins="$(field_of "$f" expected_minutes)"
    case "$ep" in '' | *[!0-9]*) return 1 ;; esac
    case "$mins" in '' | *[!0-9]*) return 1 ;; esac
    [ "$(date +%s)" -lt $((ep + mins * 60)) ]
}

# 0 = the read-only legacy file holds a live pid of another worktree.
legacy_held() {
    [ "$LEGACY" != "$LOCK" ] && [ -n "$(field_of "$LEGACY" pid)" ] && held_in "$LEGACY" &&
        [ "$(field_of "$LEGACY" worktree)" != "$OWNER" ]
}

# Serialises every read-modify-write of $LOCK; held for milliseconds only.
with_mutex() {
    local rc
    exec 9>"$LOCK.mutex" || die "cannot open $LOCK.mutex"
    flock -w 10 9 || die "mutex $LOCK.mutex busy for 10 s"
    "$@"; rc=$?
    exec 9>&-
    return $rc
}

log_history() {  # log_history <event>
    echo "[$(now) $1] holder=$(field holder) worktree=$(field worktree) purpose=$(field purpose) $(field history)" \
        >> "$HIST" 2>/dev/null
}

# Removes a dead or expired holder. Caller holds the mutex.
clear_stale() {
    [ -s "$LOCK" ] || return 0
    held_in "$LOCK" && return 0
    echo "gpu_lock: cleared stale lock (holder gone or expired): $(holder_line)" >&2
    log_history "cleared stale by $OWNER"
    rm -f "$LOCK"
}

do_acquire() {  # do_acquire <pid|""> <purpose>
    local pid="$1" purpose="$2" keeper=0 tmp start ts
    clear_stale
    if legacy_held; then
        echo "gpu_lock: GPU held by another session (legacy $LEGACY): $(line_of "$LEGACY")" >&2
        return 1
    fi
    if [ -s "$LOCK" ]; then
        if [ "$(field worktree)" = "$OWNER" ]; then
            echo "gpu_lock: already held by this worktree: $(holder_line)"
            return 0
        fi
        echo "gpu_lock: GPU held by another session: $(holder_line)" >&2
        return 1
    fi
    if [ -z "$pid" ]; then
        # Detached keeper: survives the calling shell, dies after TTL (then the lock is stale).
        tmp="$(mktemp)"
        # shellcheck disable=SC2016 # $$ expands in the keeper, not here
        setsid sh -c 'echo $$ > "$1"; exec sleep "$2"' _ "$tmp" "$TTL" \
            </dev/null >/dev/null 2>&1 9>&- &
        for _ in $(seq 1 50); do [ -s "$tmp" ] && break; sleep 0.02; done
        pid="$(cat "$tmp")"; rm -f "$tmp"
        [ -n "$pid" ] || die "keeper did not start"
        keeper=1
    fi
    start="$(start_of "$pid")" || die "pid $pid is not running"
    ts="$(now)"
    printf 'pid=%s\nstart=%s\nkeeper=%s\nworktree=%s\nholder=%s\npurpose=%s\nsince=%s\nstarted=%s\nstarted_epoch=%s\nexpected_minutes=%s\nvram_gb=?\nunterbrechbar=fragen\nhistory=[%s acquired by %s]\n' \
        "$pid" "$start" "$keeper" "$OWNER" "$OWNER" "$purpose" "$ts" "$ts" "$(date +%s)" \
        "$((TTL / 60))" "$ts" "$OWNER" > "$LOCK.tmp" && mv -f "$LOCK.tmp" "$LOCK"
    echo "gpu_lock: acquired: $(holder_line)"
}

do_release() {  # do_release <pid|""> <force 0|1>
    local pid="$1" force="$2"
    clear_stale
    [ -s "$LOCK" ] || return 0
    if [ -n "$pid" ]; then
        [ "$(field pid)" = "$pid" ] || return 0
    elif [ "$force" != "1" ] && [ "$(field worktree)" != "$OWNER" ]; then
        echo "gpu_lock: not released, held by another worktree: $(holder_line)" >&2
        echo "          (--force releases it anyway)" >&2
        return 1
    fi
    [ "$(field keeper)" = "1" ] && held_in "$LOCK" && kill "$(field pid)" 2>/dev/null
    echo "gpu_lock: released: $(holder_line)"
    log_history "released by $OWNER"
    rm -f "$LOCK"
}

do_status() {
    clear_stale
    if [ -s "$LOCK" ]; then echo "held: $(holder_line)"
    elif legacy_held; then echo "held (legacy $LEGACY): $(line_of "$LEGACY")"
    else echo "free"; fi
}

do_check() {
    clear_stale
    if legacy_held; then
        echo "GPU held by another session (legacy $LEGACY): $(line_of "$LEGACY")" >&2
        return 1
    fi
    [ -s "$LOCK" ] || return 0
    [ "$(field worktree)" = "$OWNER" ] && return 0
    echo "GPU held by another session: $(holder_line)" >&2
    echo "  (scripts/gpu_lock.sh status; the holder releases with scripts/gpu_lock.sh release)" >&2
    return 1
}

CMD="${1:-}"
[ $# -gt 0 ] && shift
case "$CMD" in
    acquire)
        PID=""
        if [ "${1:-}" = "--pid" ]; then PID="${2:?--pid needs a value}"; shift 2; fi
        [ $# -ge 1 ] || die "usage: acquire [--pid PID] <purpose>"
        with_mutex do_acquire "$PID" "$*" ;;
    release)
        PID="" FORCE=0
        while [ $# -gt 0 ]; do
            case "$1" in
                --pid) PID="${2:?--pid needs a value}"; shift 2 ;;
                --force) FORCE=1; shift ;;
                *) die "usage: release [--pid PID] [--force]" ;;
            esac
        done
        with_mutex do_release "$PID" "$FORCE" ;;
    status) with_mutex do_status ;;
    check) with_mutex do_check ;;
    run)
        [ $# -ge 3 ] && [ "$2" = "--" ] || die "usage: run <purpose> -- <cmd...>"
        PURPOSE="$1"; shift 2
        with_mutex do_acquire "$$" "$PURPOSE" >/dev/null || exit 1
        trap 'with_mutex do_release "$$" 0 >/dev/null' EXIT
        trap 'exit 130' INT TERM
        "$@" ;;
    *) die "usage: gpu_lock.sh acquire|release|status|check|run (see header)" ;;
esac
