#!/usr/bin/env bash
# Does the performance gate see a mutant that only costs throughput?
#
# Usage: perf_ab.sh <MID> [model] [rounds]
#
# The real perf gate (tests/perf_baseline.json via make verify-fast) needs a full make build per
# mutant; this measures the delta directly on the incremental build-dev binary instead.
# Defensible for a delta only (not for pinning a baseline): both arms come from the same tree
# with one line different, alternating so drift cancels; checked against the gate's 8% decode threshold.
set -uo pipefail
cd "$(dirname "$0")/../.."
MID=${1:-M29}
MODEL=${2:-$HOME/models/Qwen3-4B-Instruct-2507-Q8_0.gguf}
ROUNDS=${3:-3}
OUT=${IMP_AB_LOG:-/tmp/imp-perf-ab.log}
: >"$OUT"

# Same invocation scripts/verify.sh uses for the decode gate (:386).
bench() {
  docker run --rm --gpus all -v "$PWD":/src -v "$(dirname "$MODEL")":/models \
    -w /src/build-dev imp:toolchain ./imp-cli \
    --model "/models/$(basename "$MODEL")" --bench --bench-pp 512 --bench-reps 3 \
    --prefill-chunk-size 0 --max-tokens 128 --temperature 0 \
    --set speculative.ngram=false 2>&1 \
    | tee -a "$OUT" | grep -oP '^tg\s+128\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' | head -1
}

apply() { IMP_AB_MID="$MID" python3 - "$1" <<'PY'
import json, os, pathlib, sys
mode = sys.argv[1]
m = [x for x in json.load(open('tools/mutation/catalogue.json'))['mutants'] if x['id'] == os.environ['IMP_AB_MID']][0]
p = pathlib.Path(m['file']); t = p.read_text()
if mode == 'on':
    assert t.count(m['find']) == 1
    p.write_text(t.replace(m['find'], m['replace'], 1))
else:
    assert t.count(m['replace']) == 1
    p.write_text(t.replace(m['replace'], m['find'], 1))
PY
}
build() { docker run --rm -v "$PWD":/src -w /src imp:toolchain ninja -C /src/build-dev >/dev/null 2>&1; }

for r in $(seq 1 "$ROUNDS"); do
  apply off 2>/dev/null || true; build; base=$(bench clean)
  apply on;  build; mut=$(bench m29)
  apply off; build
  echo "round $r: clean=$base  $MID=$mut"
done
echo "--- tree ---"
git status --porcelain | grep -v '^??' || echo CLEAN
