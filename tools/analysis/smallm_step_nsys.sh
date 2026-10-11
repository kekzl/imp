#!/usr/bin/env bash
# Small-M GEMM class per decode step at CONC streams (#2414): imp-server under nsys with graph-node
# trace, a CONC-stream burst of PLEN-filler prompts, then per step (interval between consecutive
# sampler launches) the summed small-M kernel time; median over steps with the full batch.
# Usage: tools/analysis/smallm_step_nsys.sh [OUTDIR]. Env: MODEL, CONC (32), WAVES (2), PLEN, BIN_DIR
# (dir holding imp-server, default build-dev), FLOOR_MS (4.56, the resident-bandwidth floor).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$HOME/imp-ab/smallm_step}"
MODEL=${MODEL:-/models/Qwen3-14B-NVFP4}
CONC=${CONC:-32}
WAVES=${WAVES:-2}
PLEN=${PLEN:-900}
PORT=${PORT:-8095}
BIN_DIR=${BIN_DIR:-$ROOT/build-dev}
NSYS="${NSYS:-$(ls -d /opt/nvidia/nsight-systems/*/ | sort | tail -1)}"
mkdir -p "$OUT" && chmod 777 "$OUT"
"$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" > /dev/null || { echo "GPU busy" >&2; exit 1; }
docker rm -f imp-smallm >/dev/null 2>&1
docker run -d --name imp-smallm --gpus all -v "$HOME/models":/models:ro -v "$BIN_DIR":/bin_arm:ro -v "$OUT":/out \
    -v "$NSYS":/nsys -w /tmp -p ${PORT}:${PORT} --entrypoint /nsys/target-linux-x64/nsys imp:toolchain \
    profile --sample=none --cpuctxsw=none --backtrace=none -t cuda --cuda-graph-trace=node \
    -o /out/serving --force-overwrite=true \
    /bin_arm/imp-server --model "$MODEL" --port $PORT --host 0.0.0.0 --max-concurrent $CONC \
    --set runtime.max_batch_size=32 --set runtime.max_seq_len=4096 --set kv_cache.max_blocks=2387 >/dev/null
for _ in $(seq 1 240); do
    sleep 2
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    [ -z "$(docker ps -q -f name=imp-smallm)" ] && { echo "server died"; docker logs imp-smallm 2>&1 | tail -20; exit 3; }
done
docker run --rm --network host -e MODEL_NAME="$(basename "$MODEL")" -v "$ROOT/tools/analysis:/h:ro" python:3.13-slim \
    python -I /h/conc_client.py $PORT $CONC $WAVES smallm $PLEN 2>&1 | tee "$OUT/client.log"
docker kill --signal=SIGINT imp-smallm >/dev/null 2>&1
for _ in $(seq 1 90); do
    sleep 2
    [ -z "$(docker ps -q -f name=imp-smallm)" ] && break
done
docker logs imp-smallm > "$OUT/server.log" 2>&1
docker rm -f imp-smallm >/dev/null 2>&1
nsys export --type sqlite --force-overwrite=true -o "$OUT/serving.sqlite" "$OUT/serving.nsys-rep" >/dev/null 2>&1
docker run --rm -v "$OUT":/d:ro -e FLOOR_MS="${FLOOR_MS:-4.56}" python:3.13-slim python -I -c "
import os, sqlite3, statistics
rows = sqlite3.connect('/d/serving.sqlite').execute(
    'SELECT k.start, k.end, k.gridX, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '
    'JOIN StringIds s ON s.id = k.shortName ORDER BY k.start').fetchall()
marks = [r for r in rows if 'sampl' in r[3].lower() or 'argmax' in r[3].lower()]
steps = []
j = 0
for a, b in zip(marks, marks[1:]):
    while j < len(rows) and rows[j][0] < a[1]:
        j += 1
    k, t, n = j, 0, 0
    while k < len(rows) and rows[k][0] < b[0]:
        if 'smallm' in rows[k][3]:
            t += rows[k][1] - rows[k][0]; n += 1
        k += 1
    steps.append((b[0] - a[1], t, n))
full = [s for s in steps if s[2] > 0]
top = max(s[2] for s in full)
sel = [s for s in full if s[2] == top]
print(f'steps with small-M launches {len(full)}, launches per step {top}, steps at that count {len(sel)}')
print(f'small-M class per step median {statistics.median(s[1] for s in sel)/1e6:.3f} ms '
      f'(min {min(s[1] for s in sel)/1e6:.3f}, max {max(s[1] for s in sel)/1e6:.3f}), '
      f'step period median {statistics.median(s[0] for s in sel)/1e6:.3f} ms, floor {os.environ[\"FLOOR_MS\"]} ms')
"
