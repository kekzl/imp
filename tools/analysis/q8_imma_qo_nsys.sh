#!/usr/bin/env bash
# Per-layer q_proj / o_proj GEMM-class time at M=2048 under nsys (#2416): quantize or dequant plus the
# GEMM pieces after the attention rmsnorm (q) and after attention (o), median over the layers of a run.
# Usage: tools/analysis/q8_imma_qo_nsys.sh BINDIR on|off TAG [MODEL]; on = IMMA, off = FP16 route.
set -euo pipefail
dir="$1" imma="$2" tag="$3" model="${4:-Qwen3-8B-Q8_0.gguf}"
out="${OUT:-$HOME/imp-ab/nsys}"
NSYS="${NSYS:-$(ls -d /opt/nvidia/nsight-systems/*/ | sort | tail -1)}"
mkdir -p "$out" && chmod 777 "$out"
[ "$imma" = on ] && val=true || val=false
"$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" > /dev/null || { echo "GPU busy" >&2; exit 1; }
docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v "$HOME/models":/models:ro -v "$dir":/arm:ro \
    -v "$out":/out -v "$NSYS":/nsys --entrypoint /nsys/target-linux-x64/nsys imp:toolchain \
    profile --sample=none --cpuctxsw=none --backtrace=none -t cuda -o "/out/$tag" --force-overwrite=true \
    /arm/imp-cli --model "/models/$model" --bench --bench-pp 2048 --bench-reps 3 --max-tokens 1 \
    --temperature 0 --seed 42 --set speculative.ngram=false --set speculative.mtp_k=0 \
    --set "gemm.q8_imma_enabled=$val" > "$out/$tag.log" 2>&1 || true
nsys export --type sqlite -o "$out/$tag.sqlite" --force-overwrite=true "$out/$tag.nsys-rep" > /dev/null
docker run --rm -v "$out":/d:ro python:3.13-slim python -I -c "
import sqlite3, statistics
rows = sqlite3.connect('/d/$tag.sqlite').execute(
    'SELECT k.start, k.end, k.gridX, k.gridY, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '
    'JOIN StringIds s ON s.id = k.shortName ORDER BY k.start').fetchall()
def piece(win):  # prep before the first GEMM + that GEMM + its BM=32 tail
    t, first = 0, None
    for w in win:
        n = w[4]
        if ('quantize_act' in n or 'dequant_q8_0' in n) and first is None:
            t += w[1] - w[0]
        elif any(g in n for g in ('mmq_imma', 'nvjet', 'cutlass', 'gemm')):
            if first is None:
                first = w; t += w[1] - w[0]
            elif 'mmq_imma' in n and w[2] == first[2] and w[3] == 4:
                t += w[1] - w[0]; break
            else:
                break
        elif first is not None:
            break
    return t / 1e3 if first else None
q, o = [], []
for i, r in enumerate(rows):
    if 'rmsnorm' in r[4] and any('qknorm' in w[4] for w in rows[i + 1:i + 14]):
        v = piece(rows[i + 1:i + 8]); q += [v] if v else []
    if 'fmha_sm120' in r[4]:
        v = piece(rows[i + 1:i + 7]); o += [v] if v else []
q = [v for v in q if v > 150]; o = [v for v in o if v > 150]
print(f'$tag q_proj n={len(q)} median {statistics.median(q):.1f} us | o_proj n={len(o)} median {statistics.median(o):.1f} us')
"
