#!/usr/bin/env bash
# B-85: which activation sites carry Flint's W4A4 loss. CPU reference (flint-ref:local), BF16 forward as the
# reference (logits cached in $OUT_DIR/ref_logits.pt), the NVFP4 repack scored at W4A16 and with NVFP4
# activations at one site group per arm; two containers in parallel. One arm ~10 min on an idle 16-core host.
# Usage: tools/analysis/flint_a4_sites.sh [MAX_TOKENS]   -> $OUT_DIR/a4_sites_{1,2}.log
set -euo pipefail
N=${1:-4096}
OUT_DIR=${OUT_DIR:-/mnt/data/scratch/flint_a4}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
Q=/models/flint-nvfp4-official
mkdir -p "$OUT_DIR"
run() { docker run --rm -e OMP_NUM_THREADS="${THREADS:-8}" -v "$HOME/models:/models:ro" -v "$REPO/tools/analysis:/t:ro" \
    -v "$OUT_DIR:/o" -w /o --entrypoint python3 flint-ref:local -I "$@"; }
ev() { run /t/flint_quant_eval.py --model /models/flint --ids /o/ids.txt --max-tokens "$N" --ref-cache /o/ref_logits.pt "$@"; }
[ -f "$OUT_DIR/ids.txt" ] || run /t/flint_tokenize.py /models/flint /t/ppl_corpus_45k.txt /o/ids.txt "$N"
[ -f "$OUT_DIR/ref_logits.pt" ] || THREADS=16 ev 2>&1 | tee "$OUT_DIR/ref.log"
ev --ckpt "a4_all:$Q:." --ckpt "a4_attn:$Q:attn\." --ckpt "a4_dense:$Q:dense\." --ckpt "a4_routed_down:$Q:routed\.down" \
    > "$OUT_DIR/a4_sites_1.log" 2>&1 &
ev --ckpt "a4_routed:$Q:routed\." --ckpt "a4_not_routed:$Q:attn|dense|shared" --ckpt "a4_shared:$Q:shared\." \
    --ckpt "a16:$Q" > "$OUT_DIR/a4_sites_2.log" 2>&1 &
wait
grep -h '^ARM' "$OUT_DIR"/a4_sites_[12].log
