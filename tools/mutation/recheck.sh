#!/usr/bin/env bash
# Targeted re-check for survivors whose would-be oracle was red in the baseline.
#
# The full `test-e2e` run loads three large models in one process; on WSL2 a
# process never gets its peak VRAM back, so ~38 tests fail for capacity alone.
# run.py correctly discounts those as pre-existing — but that also means those
# tests cannot *kill* a mutant, so a "SURVIVED" verdict from that lane says
# nothing. This re-runs one mutant against one small, isolated filter that is
# green on the clean tree in a fresh process.
#
# Usage: recheck.sh <MID> <binary> <gtest_filter>
set -uo pipefail
cd "$(dirname "$0")/../.."
MID=$1 BIN=$2 FILTER=$3 REPEATS=${4:-1}
MODELS=${IMP_MODELS_DIR:-$HOME/models}
CAT=tools/mutation/catalogue.json
OUT=loop/evidence/recheck
mkdir -p "$OUT"

run() {
  docker run --rm --gpus all -v "$PWD":/src -v "$MODELS":/models \
    -w /src/build-dev \
    -e IMP_TEST_MODEL=/models/Qwen3-8B-Q8_0.gguf \
    -e IMP_TEST_MOE_MODEL=/models/gpt-oss-20b-mxfp4.gguf \
    -e IMP_TEST_MODEL_QWEN4B=/models/Qwen3-4B-Instruct-2507-Q8_0.gguf \
    -e IMP_TEST_MODEL_LLAMA=/models/Llama-3.2-3B-Instruct-Q8_0.gguf \
    imp:toolchain ./"$BIN" --gtest_filter="$FILTER" 2>&1
}
build() { docker run --rm -v "$PWD":/src -w /src imp:toolchain ninja -C /src/build-dev 2>&1; }

python3 - "$MID" <<'PY'
import json, os, sys, pathlib
mid = sys.argv[1]
m = [x for x in json.load(open('tools/mutation/catalogue.json'))['mutants'] if x['id'] == mid][0]
p = pathlib.Path(m['file']); t = p.read_text()
assert t.count(m['find']) == 1, f'{mid}: anchor not unique'
orig = pathlib.Path(os.environ.get('TMPDIR', '/tmp')) / 'imp_recheck_orig'
orig.write_text(t)
p.write_text(t.replace(m['find'], m['replace'], 1))
print(f'{mid} applied to {m["file"]}')
PY

build > "$OUT/$MID-build.log" 2>&1
# Repeat the run: DetEvalE2ETest is flaky on the clean tree (see
# loop/evidence/P0-deteval-flake.log), so a single red run is not a kill and a
# single green run is not a survival.
for i in $(seq 1 "$REPEATS"); do
  run > "$OUT/$MID-mutant-r$i.log" 2>&1
done
cp "$OUT/$MID-mutant-r1.log" "$OUT/$MID-mutant.log"

python3 - "$MID" <<'PY'
import json, os, sys, pathlib
mid = sys.argv[1]
m = [x for x in json.load(open('tools/mutation/catalogue.json'))['mutants'] if x['id'] == mid][0]
orig = pathlib.Path(os.environ.get('TMPDIR', '/tmp')) / 'imp_recheck_orig'
pathlib.Path(m['file']).write_text(orig.read_text())
print(f'{mid} reverted')
PY
build > "$OUT/$MID-rebuild.log" 2>&1

echo "--- $MID against $BIN [$FILTER] × $REPEATS ---"
for i in $(seq 1 "$REPEATS"); do
  printf '  run %d: ' "$i"
  grep -E '^\[  (PASSED|FAILED)  \] [0-9]+ test' "$OUT/$MID-mutant-r$i.log" | tr '\n' ' '
  grep -E '^\[  FAILED  \] [A-Za-z]' "$OUT/$MID-mutant-r$i.log" | sort -u | sed 's/\[  FAILED  \] //' | tr '\n' ' '
  echo
done
echo "--- tree ---"
git status --porcelain | grep -v '^??' || echo CLEAN
