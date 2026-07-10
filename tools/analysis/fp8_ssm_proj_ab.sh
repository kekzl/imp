#!/bin/bash
# A/B the FP8 E4M3 decode sidecar for native-precision GDN/SSM projections
# (gemm.fp8_ssm_proj) on NVFP4 hybrids: decode tg256 (10-rep, spec off for the
# kernel-level signal + defaults-on for the user-visible number), perplexity
# (teacher-forced, same-corpus delta), and a coherence sample.
set -uo pipefail

IMG=${IMG:-imp:test}
MODELS=/home/kekz/models
CORPUS=/work/tools/analysis/ppl_corpus.txt
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PROMPT="Explain why the sky is blue, then say why sunsets are red."

run() {
  docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "$MODELS":/models -v "$REPO":/work "$IMG" "$@"
}

decode() { # model flag spec
  local spec_args=""
  if [ "$3" = "nospec" ]; then
    spec_args="--set speculative.suffix=false --set speculative.ngram=false --set speculative.moe=false"
  fi
  run imp-cli --model "/models/$(basename "$1")" --bench --bench-pp 16 \
      --bench-reps 10 --max-tokens 256 --temperature 0 \
      --set gemm.fp8_ssm_proj="$2" $spec_args 2>&1 \
    | grep -E '^tg|fp8_ssm_proj'
}

ppl() { # model flag
  run imp-cli --model "/models/$(basename "$1")" --perplexity "$CORPUS" \
      --temperature 0 --set gemm.fp8_ssm_proj="$2" 2>&1 \
    | grep -E '^perplexity|fp8_ssm_proj'
}

cohere() { # model flag
  run imp-cli --model "/models/$(basename "$1")" --prompt "$PROMPT" \
      --max-tokens 80 --temperature 0 --set gemm.fp8_ssm_proj="$2" 2>&1 \
    | grep -oE '\][^[]*' | sed -E 's/^\] ?//' | tr -d '\n'
}

MODEL="$1"
echo "######## MODEL: $(basename "$MODEL") ########"
for flag in false true; do
  echo "===== fp8_ssm_proj=$flag ====="
  echo "--- decode (spec off) ---"; decode "$MODEL" "$flag" nospec
  echo "--- decode (defaults) ---"; decode "$MODEL" "$flag" default
  echo "--- ppl ---";              ppl    "$MODEL" "$flag"
  echo "--- text ---";             cohere "$MODEL" "$flag"; echo
done
echo "DONE $(basename "$MODEL")"
