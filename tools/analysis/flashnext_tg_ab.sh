#!/usr/bin/env bash
# Flash-Next decode A/B on real prompts: arm A vs arm B imp-cli args, alternating A B B A per prompt,
# prints tg tok/s and the device expert-cache hit rate per run.
# Usage: tools/analysis/flashnext_tg_ab.sh OUTDIR "<arm A args>" "<arm B args>"
# Env: MODEL, IMG_A / IMG_B (imp:test), BIN_A / BIN_B (imp-cli), BIN_DIR_A / BIN_DIR_B (mounted at /bin_arm),
#      TOKENS (512), ROUNDS (1 = A B B A).
set -uo pipefail
OUT="$1"; ARGS_A="$2"; ARGS_B="$3"
MODEL=${MODEL:-/models/Qwen3.8-Flash-Next-NVFP4}
TOKENS=${TOKENS:-512}
ROUNDS=${ROUNDS:-1}
mkdir -p "$OUT" && chmod 777 "$OUT"
declare -A P
P[prose]="Write a detailed essay about the history of the Hanseatic League, its trade routes, its decline, and its legacy in northern European cities."
P[code]="Implement a lock-free single-producer single-consumer ring buffer in C++20 with std::atomic, explain the memory orderings, and add a small benchmark main()."
P[chat]="Hey, I'm planning a 5 day trip to Japan in November with my partner. We like food, hiking and quiet places. Can you suggest an itinerary and some tips?"
run() {  # arm prompt idx
    local arm=$1 k=$2 i=$3 img bin dir args
    if [ "$arm" = A ]; then img=${IMG_A:-imp:test}; bin=${BIN_A:-imp-cli}; dir=${BIN_DIR_A:-}; args=$ARGS_A
    else img=${IMG_B:-imp:test}; bin=${BIN_B:-imp-cli}; dir=${BIN_DIR_B:-}; args=$ARGS_B; fi
    local mnt=()
    [ -n "$dir" ] && mnt=(-v "$dir":/bin_arm:ro)
    "$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" > /dev/null || { echo "GPU busy" >&2; exit 1; }
    # shellcheck disable=SC2086
    docker run --rm --gpus all -v "$HOME/models":/models:ro "${mnt[@]}" --entrypoint "$bin" "$img" \
        --model "$MODEL" --prompt "${P[$k]}" --max-tokens "$TOKENS" --seed 1 $args > "$OUT/$arm.$k.$i.log" 2>&1
    local tg hit slots
    tg=$(sed -nE 's/^tg +[0-9]+ tokens in .*\( *([0-9.]+) tok\/s\)/\1/p' "$OUT/$arm.$k.$i.log" | tail -1)
    hit=$(sed -nE 's/.*Device expert cache stats: .*\(([0-9.]+)% hit rate\).*/\1/p' "$OUT/$arm.$k.$i.log" | tail -1)
    slots=$(sed -nE 's/.*Device expert cache: .*\(([0-9]+) slots\/layer.*/\1/p' "$OUT/$arm.$k.$i.log" | tail -1)
    echo "$arm $k $i tg=${tg:-FAIL} hit=${hit:-?} slots=${slots:-?}"
}
for i in $(seq 1 "$ROUNDS"); do
    for k in prose code chat; do
        run A $k $i; run B $k $i; run B $k $((i + 100)); run A $k $((i + 100))
    done
done
