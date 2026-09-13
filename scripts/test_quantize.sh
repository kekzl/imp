#!/usr/bin/env bash
# imp-quantize round trip (AUDIT_arch_2026 I-6): quantize a small BF16 SafeTensors checkpoint
# to NVFP4, load through imp-cli, require a coherent greedy answer.
# Usage: make test-quantize (or scripts/test_quantize.sh). Env: IMP_QUANT_MODEL (default
# Qwen3-0.6B), IMP_MODELS_DIR (default $HOME/models), IMP_TEST_IMG (default imp:test),
# IMP_QUANT_FORMAT (default modelopt, or vllm).
set -euo pipefail
MODEL=${IMP_QUANT_MODEL:-Qwen3-0.6B}
MODELS_DIR=${IMP_MODELS_DIR:-$HOME/models}
IMG=${IMP_TEST_IMG:-imp:test}
FORMAT=${IMP_QUANT_FORMAT:-modelopt}
# A named volume, not a bind mount: the image runs as uid 1001 and cannot
# write into a host directory owned by the operator.
VOL="imp-quantize-test-$$"
docker volume create "$VOL" >/dev/null
trap 'docker volume rm -f "$VOL" >/dev/null 2>&1' EXIT
# A fresh volume is root-owned; open it for the image's uid once.
docker run --rm --user 0 -v "$VOL:/out" --entrypoint chmod "$IMG" 0777 /out

if [ ! -f "$MODELS_DIR/$MODEL/model.safetensors" ]; then
    echo "test-quantize: SKIP, no BF16 checkpoint at $MODELS_DIR/$MODEL" >&2
    exit 0
fi

echo "== imp-quantize $MODEL -> NVFP4 ($FORMAT) =="
docker run --rm --gpus all -v "$MODELS_DIR:/models:ro" -v "$VOL:/out" "$IMG" \
    imp-quantize --model "/models/$MODEL" --out /out/q --format "$FORMAT"
docker run --rm -v "$VOL:/out" --entrypoint ls "$IMG" -la /out/q | head -20

echo "== load the result and answer greedily =="
# Raw completion (no chat template: Qwen3 would spend the budget on a think
# block) and the JSON document on stdout, so the answer is one grep.
ANSWER=$(docker run --rm --gpus all -v "$VOL:/out" "$IMG" \
    imp-cli --model /out/q --prompt "The capital of France is" --max-tokens 8 --temperature 0 \
            --chat-template none --json 2>/dev/null)
echo "$ANSWER" | cut -c1-400
if ! grep -q "Paris" <<< "$ANSWER"; then
    echo "test-quantize: FAIL, the quantized $MODEL did not answer Paris" >&2
    exit 1
fi
echo "test-quantize: PASS ($MODEL -> NVFP4 $FORMAT round trip answers Paris)"
