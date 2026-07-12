#!/bin/bash
# PPL A/B for the batched-decode CUTLASS NVFP4 LM head (gemm.nvfp4_lm_head_cutlass):
# teacher-forced perplexity over a ~13.5k-token corpus, one fresh process per arm.
# The perplexity harness routes its LM head through the same CUTLASS
# NVFP4-activation path as batched decode when the flag is on, so the printed
# delta IS the batched-serving quality trade. Identical PPL across arms means
# the path did not engage for that model (LM head not in the NVFP4 decode
# cache, e.g. tied-embedding Gemma GGUF).
#
# 2026-07-12 default-on sweep (this script): MoE/hybrid +1.9-2.1%
# (Coder-30B 10.19->10.38, Modelopt-30B 11.65->11.88, Qwen3.6-35B 13.39->13.66),
# dense +0.2-0.5% -- inside the +-0.3-0.5% run-to-run spread (cuBLAS algo
# selection at process start perturbs the prefill hidden states; run 3+ trials
# per arm before trusting a sub-1% delta).
set -uo pipefail

IMG=${IMG:-imp:test}
MODELS=$HOME/models
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CORPUS_REL=tools/analysis/ppl_corpus_45k.txt

# Corpus: ~45 KB of repo docs (~13.5k tokens). Regenerated deterministically
# from the working tree so no blob needs to be checked in.
if [ ! -f "$REPO/$CORPUS_REL" ]; then
  cat "$REPO"/docs/architecture.md "$REPO"/docs/sm120.md "$REPO"/docs/BENCHMARKING.md \
      "$REPO"/README.md "$REPO"/docs/GOAL.md | head -c 45000 > "$REPO/$CORPUS_REL"
fi

run() {
  docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "$MODELS":/models -v "$REPO":/work "$IMG" "$@"
}

MODEL="$1"
TRIALS=${TRIALS:-1}
echo "######## MODEL: $(basename "$MODEL") ########"
for t in $(seq 1 "$TRIALS"); do
  for flag in false true; do
    echo "===== trial=$t nvfp4_lm_head_cutlass=$flag ====="
    run imp-cli --model "/models/$(basename "$MODEL")" --perplexity "/work/$CORPUS_REL" \
        --max-seq-len 16384 --temperature 0 \
        --set gemm.nvfp4_lm_head_cutlass="$flag" 2>&1 \
      | grep -E "^perplexity:|LM head: CUTLASS"
  done
done
echo "DONE $(basename "$MODEL")"
