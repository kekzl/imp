#!/usr/bin/env bash
# imp-quantize reproducibility (#2481): calibrate once, export twice with the same binary and
# recipe, require every output file byte-identical; a third export with one flag changed must
# record a different recipe.
# Usage: make test-quantize-repro. Env: IMP_QUANT_MODEL (default Qwen3-0.6B), IMP_MODELS_DIR
# (default $HOME/models), IMP_TEST_IMG (default imp:test), IMP_REPRO_LOG_DIR (default
# $HOME/.cache/imp/q9).
set -euo pipefail
MODEL=${IMP_QUANT_MODEL:-Qwen3-0.6B}
MODELS_DIR=${IMP_MODELS_DIR:-$HOME/models}
IMG=${IMP_TEST_IMG:-imp:test}
LOG_DIR=${IMP_REPRO_LOG_DIR:-$HOME/.cache/imp/q9}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$LOG_DIR"
if [ ! -f "$MODELS_DIR/$MODEL/model.safetensors" ]; then
    echo "test-quantize-repro: SKIP, no BF16 checkpoint at $MODELS_DIR/$MODEL" >&2
    exit 0
fi
# Named volume: the image runs as uid 1001 and cannot write a host directory; removed on exit.
VOL="imp-quantize-repro-$$"
docker volume create "$VOL" >/dev/null
trap 'docker volume rm -f "$VOL" >/dev/null 2>&1' EXIT
docker run --rm --user 0 -v "$VOL:/out" --entrypoint chmod "$IMG" 0777 /out
RUN=(docker run --rm --gpus all -v "$MODELS_DIR:/models:ro" -v "$ROOT/tools/analysis:/corpus:ro" -v "$VOL:/out" "$IMG")

echo "== calibrate $MODEL =="
"${RUN[@]}" imp-cli --model "/models/$MODEL" --perplexity /corpus/ppl_corpus.txt --calibrate /out/calib.bin \
    > "$LOG_DIR/calibrate.log" 2>&1
for arm in a b; do
    echo "== export $arm (--calib) =="
    "${RUN[@]}" imp-quantize --model "/models/$MODEL" --out "/out/$arm" --calib /out/calib.bin \
        > "$LOG_DIR/export_$arm.log" 2>&1
done
echo "== export c (--calib --calib-weight sq) =="
"${RUN[@]}" imp-quantize --model "/models/$MODEL" --out /out/c --calib /out/calib.bin --calib-weight sq \
    > "$LOG_DIR/export_c.log" 2>&1

sums() {  # sha256 of every file in /out/<arm>, path relative to the arm
    docker run --rm -v "$VOL:/out" --entrypoint sh "$IMG" -c "cd /out/$1 && find . -type f | sort | xargs sha256sum"
}
sums a > "$LOG_DIR/sha256_a.txt"
sums b > "$LOG_DIR/sha256_b.txt"
sums c > "$LOG_DIR/sha256_c.txt"
cat "$LOG_DIR/sha256_a.txt"
docker run --rm -v "$VOL:/out" --entrypoint cat "$IMG" /out/a/hf_quant_config.json > "$LOG_DIR/hf_quant_config_a.json"
docker run --rm -v "$VOL:/out" --entrypoint cat "$IMG" /out/c/hf_quant_config.json > "$LOG_DIR/hf_quant_config_c.json"

fail=0
if ! diff -u "$LOG_DIR/sha256_a.txt" "$LOG_DIR/sha256_b.txt"; then
    echo "test-quantize-repro: FAIL, two exports of one recipe differ" >&2
    fail=1
fi
if ! grep -q '"recipe_version"' "$LOG_DIR/hf_quant_config_a.json" || ! grep -q '"file_sha256"' "$LOG_DIR/hf_quant_config_a.json"; then
    echo "test-quantize-repro: FAIL, hf_quant_config.json carries no calibrated recipe" >&2
    fail=1
fi
if cmp -s "$LOG_DIR/hf_quant_config_a.json" "$LOG_DIR/hf_quant_config_c.json" \
    || ! grep -q '"weight": "sq"' "$LOG_DIR/hf_quant_config_c.json"; then
    echo "test-quantize-repro: FAIL, --calib-weight sq left the recipe unchanged" >&2
    fail=1
fi
[ "$fail" = 0 ] || exit 1
echo "test-quantize-repro: PASS ($MODEL: $(wc -l < "$LOG_DIR/sha256_a.txt") files byte-identical across two --calib exports; --calib-weight sq recipe differs; logs $LOG_DIR)"
