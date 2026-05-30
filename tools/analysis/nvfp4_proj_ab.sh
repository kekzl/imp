#!/bin/bash
# A/B the recipe-excluded BF16 GDN/attn projection NVFP4 lever on native-NVFP4
# hybrids. 4 configs: off / attn-only / gdn-only / both.
# Decode tg256 (10-rep), perplexity (teacher-forced), + coherence sample.
set -uo pipefail

IMG=imp:test
MODELS=/home/kekz/models
CORPUS=/work/tools/analysis/ppl_corpus.txt
REPO=/home/kekz/github.com/kekzl/imp
PROMPT="Explain why the sky is blue, then say why sunsets are red."

run() {  # $1=model $2=attn $3=gdn  -> extra args
  docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "$MODELS":/models -v "$REPO":/work "$IMG" "$@"
}

decode() { # model attn gdn
  run "$1" imp-cli --model "/models/$(basename $1)" --bench --bench-pp 16 \
      --bench-reps 10 --max-tokens 256 --temperature 0 \
      --set gemm.nvfp4_attn_proj="$2" --set gemm.nvfp4_gdn_proj="$3" 2>&1 \
    | grep -E '^tg|NVFP4 attn proj|NVFP4 GDN proj|gemv_fp16'
}

ppl() { # model attn gdn
  run "$1" imp-cli --model "/models/$(basename $1)" --perplexity "$CORPUS" \
      --temperature 0 \
      --set gemm.nvfp4_attn_proj="$2" --set gemm.nvfp4_gdn_proj="$3" 2>&1 \
    | grep -E '^perplexity'
}

cohere() { # model attn gdn
  run "$1" imp-cli --model "/models/$(basename $1)" --prompt "$PROMPT" \
      --max-tokens 80 --temperature 0 \
      --set gemm.nvfp4_attn_proj="$2" --set gemm.nvfp4_gdn_proj="$3" 2>&1 \
    | grep -oE '\][^[]*' | sed -E 's/^\] ?//' | tr -d '\n'
}

MODEL="$1"   # full path
echo "######## MODEL: $(basename $MODEL) ########"
# config: label attn gdn
for cfg in "off:false:false" "attn:true:false" "gdn:false:true" "both:true:true"; do
  lab=${cfg%%:*}; rest=${cfg#*:}; A=${rest%%:*}; G=${rest#*:}
  echo "===== cfg=$lab (attn=$A gdn=$G) ====="
  echo "--- decode ---"; decode "$MODEL" "$A" "$G" | tail -3
  echo "--- ppl ---";    ppl    "$MODEL" "$A" "$G"
  echo "--- text ---";   cohere "$MODEL" "$A" "$G"; echo
  echo "[cooldown 90s]"; sleep 90
done
echo "DONE $(basename $MODEL)"
