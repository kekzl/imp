#!/usr/bin/env bash
# Re-run only the captures still missing after the first pass aborted on
# Qwen3.5-GDN W1 (known chunked_prefill bug — skipped intentionally).
#
# Pipefail removed; each capture wrapped in `|| echo failed` so one bad
# config doesn't kill the whole batch.
set -uo pipefail

REPO="$REPO"
PROFILES="${REPO}/profiles/baselines"
NSYS=/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nsys

DOCKER_BASE="docker run --rm --gpus all \
  --user $(id -u):$(id -g) -w /tmp \
  -v ${REPO}/models:/models \
  -v ${PROFILES}:/profiles \
  -v /opt/nvidia:/opt/nvidia:ro \
  --entrypoint=${NSYS} imp:profile"

NSYS_FLAGS=(
  profile --trace=cuda,nvtx,osrt,cublas
  --cuda-memory-usage=true --force-overwrite=true --stats=false
)

run_one() {
  local tag="$1" file="$2" wl="$3" pp="$4" tg="$5" graphs="$6"
  local out="/profiles/${tag}_${wl}_${graphs}"
  local extra=()
  [ "${graphs}" = "ng" ] && extra+=(--no-cuda-graphs)

  echo "=== ${tag} | ${wl} (pp=${pp} tg=${tg}) | graphs=${graphs} ==="
  ${DOCKER_BASE} \
    "${NSYS_FLAGS[@]}" -o "${out}" \
    /usr/local/bin/imp-cli \
      --model "/models/${file}" \
      --bench --bench-pp "${pp}" --bench-reps 1 \
      --max-tokens "${tg}" --temperature 0 --seed 42 \
      "${extra[@]}" \
    2>&1 | tail -4 || echo "  (capture errored — see profile if generated)"
}

# Missing captures (GDN W1 omitted — chunked_prefill bug at pp=8192)
run_one qwen3-4b-q8       Qwen3-4B-Instruct-2507-Q8_0.gguf W1 8192 64   ng
run_one qwen3-4b-q8       Qwen3-4B-Instruct-2507-Q8_0.gguf W2 256  2048 ng
run_one qwen35-4b-gdn-q8  Qwen3.5-4B-Q8_0.gguf             W2 256  2048 ng
run_one qwen3-4b-q8       Qwen3-4B-Instruct-2507-Q8_0.gguf W2 256  2048 g

echo ""; echo "=== Done ==="
ls -la "${PROFILES}/"
