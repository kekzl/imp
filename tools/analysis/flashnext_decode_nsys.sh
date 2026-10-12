#!/usr/bin/env bash
# Per-token decode kernel classes for a host-expert MoE (Qwen3.8-Flash-Next): imp-cli under nsys with
# graph-node trace on a real prompt, decode window = last tg_ms of the run, per-class ms/token,
# busy union vs wall. Usage: tools/analysis/flashnext_decode_nsys.sh [OUTDIR] [extra imp-cli args...]
# Env: MODEL, IMG (imp:test), BIN (path inside IMG, default imp-cli), TOKENS (256), BIN_DIR (mounted at /bin_arm).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$HOME/imp-ab/flashnext_nsys}"
shift || true
MODEL=${MODEL:-/models/Qwen3.8-Flash-Next-NVFP4}
IMG=${IMG:-imp:test}
BIN=${BIN:-imp-cli}
TOKENS=${TOKENS:-256}
NSYS="${NSYS:-$(ls -d /opt/nvidia/nsight-systems/*/ | sort | tail -1)}"
PROMPT=${PROMPT:-"Implement a lock-free single-producer single-consumer ring buffer in C++20 with std::atomic, explain the memory orderings, and add a small benchmark main()."}
mkdir -p "$OUT" && chmod 777 "$OUT"
"$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" > /dev/null || { echo "GPU busy" >&2; exit 1; }
MNT=()
[ -n "${BIN_DIR:-}" ] && MNT=(-v "$BIN_DIR":/bin_arm:ro)
docker run --rm --gpus all -v "$HOME/models":/models:ro -v "$OUT":/out -v "$NSYS":/nsys "${MNT[@]}" -w /tmp \
    --entrypoint /nsys/target-linux-x64/nsys "$IMG" \
    profile --sample=none --cpuctxsw=none --backtrace=none -t cuda --cuda-graph-trace=node \
    -o /out/run --force-overwrite=true \
    "$BIN" --model "$MODEL" --prompt "$PROMPT" --max-tokens "$TOKENS" --seed 1 "$@" > "$OUT/run.log" 2>&1
grep -E "^tg |^pp |hit rate|hits" "$OUT/run.log" | tail -4
docker run --rm -v "$OUT":/out -v "$NSYS":/nsys --entrypoint /nsys/target-linux-x64/nsys "$IMG" \
    export --type sqlite --force-overwrite=true -o /out/run.sqlite /out/run.nsys-rep > /dev/null 2>&1
TG_MS=$(sed -nE 's/^tg +[0-9]+ tokens in +([0-9.]+) ms.*/\1/p' "$OUT/run.log" | tail -1)
docker run --rm -v "$OUT":/d:ro -e TG_MS="$TG_MS" -e TOKENS="$TOKENS" python:3.13-slim python -I -c "
import os, sqlite3, re
from collections import defaultdict
db = sqlite3.connect('/d/run.sqlite')
rows = db.execute('SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                  'JOIN StringIds s ON s.id = k.shortName ORDER BY k.start').fetchall()
tg = float(os.environ['TG_MS']) * 1e6
n_tok = int(os.environ['TOKENS'])
end = rows[-1][1]
win = [r for r in rows if r[0] >= end - tg]
cls = defaultdict(lambda: [0, 0])
for a, b, name in win:
    cls[name][0] += b - a
    cls[name][1] += 1
busy, cur_a, cur_b = 0, None, None
for a, b, _ in win:
    if cur_b is None or a > cur_b:
        if cur_b is not None:
            busy += cur_b - cur_a
        cur_a, cur_b = a, b
    else:
        cur_b = max(cur_b, b)
busy += cur_b - cur_a
tot = sum(v[0] for v in cls.values())
print(f'decode window {tg/1e6:.1f} ms, {n_tok} tok: wall {tg/n_tok/1e6:.3f} ms/tok, kernel sum {tot/n_tok/1e6:.3f}, busy union {busy/n_tok/1e6:.3f}')
for name, (t, n) in sorted(cls.items(), key=lambda x: -x[1][0])[:25]:
    print(f'{t/n_tok/1e3:9.1f} us/tok {n/n_tok:6.1f} launches/tok  {name[:90]}')
"
