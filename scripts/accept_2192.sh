#!/usr/bin/env bash
# GPU acceptance for #2192: stub GGUF tests must not persist warm-cache files into the
# imp-test-cache volume. Exit 0 only if every criterion passes. Needs a GPU and the worktree
# image (make build).
# Usage: bash scripts/accept_2192.sh
# Env: IMP_TEST_IMG, IMP_TEST_CACHE_VOL (default imp-test-cache),
#      IMP_ACCEPT_2192_FULL=1 runs `make test-gpu` instead of the stub lane.
# Criteria:
#   C1 volume warm/ file count (all files, and imp_stub_* files) unchanged after the run
#   C2 log shows a stub persisted into /tmp/imp_stub_*/warm/ (scratch dir is exercised)
#   C3 no /tmp/imp_stub_* dir left in the test container after the run (cleanup works)
#   C4 gtest run exit 0
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

if [ -z "${IMP_ACCEPT_2192_LOCKED:-}" ]; then
    bash scripts/require_free_gpu.sh "accept_2192" || exit 1
    IMP_ACCEPT_2192_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2192" -- bash "$0" "$@"
fi

IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
VOL="${IMP_TEST_CACHE_VOL:-imp-test-cache}"
LOG="$(mktemp "${TMPDIR:-/tmp}/accept_2192.XXXXXX.log")"
FAIL=0

verdict() {  # verdict <id> <PASS|FAIL> <detail>
    echo "$2 $1: $3"
    [ "$2" = PASS ] || FAIL=1
}

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }
docker volume inspect "$VOL" >/dev/null 2>&1 || docker volume create "$VOL" >/dev/null
echo "volume $VOL mountpoint: $(docker volume inspect -f '{{.Mountpoint}}' "$VOL")"

# Counted inside a container: the mountpoint is root-owned under /var/lib/docker.
count_warm() {  # count_warm <find-name-glob>
    docker run --rm --entrypoint sh -v "$VOL:/c:ro" "$IMG" \
        -c "mkdir -p /c/warm 2>/dev/null; find /c/warm -type f -name '$1' | wc -l"
}

all_before="$(count_warm '*')"
stub_before="$(count_warm 'imp_stub_*')"
echo "before: all=$all_before imp_stub_*=$stub_before"

if [ "${IMP_ACCEPT_2192_FULL:-}" = 1 ]; then
    make test-gpu >"$LOG" 2>&1
    rc=$?
    left=n/a
else
    # Same mounts as Makefile DOCKER_RUN; the stub tests live in test-e2e (EndToEndTest.*, StubModelTest.*).
    # The unit binary set filters out non-matching tests, so only the stub tests run.
    docker run --rm --gpus all --entrypoint sh -v "$HOME/models:/models" -v "$VOL:/home/imp/.cache/imp" \
        "$IMG" -c 'imp-tests --gtest_filter="EndToEndTest.*:StubModelTest.*"; rc=$?;
                   echo "STUB_DIRS_LEFT=$(find /tmp -maxdepth 1 -name "imp_stub_*" | wc -l)"; exit $rc' \
        >"$LOG" 2>&1
    rc=$?
    left="$(sed -n 's/^STUB_DIRS_LEFT=//p' "$LOG" | tail -1)"
fi

all_after="$(count_warm '*')"
stub_after="$(count_warm 'imp_stub_*')"
echo "after:  all=$all_after imp_stub_*=$stub_after (log: $LOG)"

if [ "$all_after" = "$all_before" ] && [ "$stub_after" = "$stub_before" ]; then
    verdict C1 PASS "delta 0 (all $all_before -> $all_after, imp_stub_* $stub_before -> $stub_after)"
else
    verdict C1 FAIL "all $all_before -> $all_after, imp_stub_* $stub_before -> $stub_after"
fi

persisted="$(grep -cE 'Warm cache: persisted .* to /tmp/imp_stub_[A-Za-z0-9]+/warm/' "$LOG")"
if [ "$persisted" -gt 0 ]; then
    verdict C2 PASS "$persisted stub persist line(s) into /tmp/imp_stub_*/warm/"
else
    verdict C2 FAIL "no 'Warm cache: persisted ... /tmp/imp_stub_*/warm/' line in $LOG"
fi

if [ "$left" = 0 ] || [ "$left" = n/a ]; then
    verdict C3 PASS "stub dirs left in container: $left"
else
    verdict C3 FAIL "stub dirs left in container: '${left}'"
fi

if [ "$rc" -eq 0 ]; then
    verdict C4 PASS "test run exit 0"
else
    verdict C4 FAIL "test run exit $rc"
fi

exit "$FAIL"
