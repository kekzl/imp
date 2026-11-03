#!/usr/bin/env bash
# imp vs vLLM gap sweep on the same SafeTensors checkpoints: one fresh server per (model, engine,
# trial), engines alternate per trial. Per arm: decode tg128 (1 stream, TPOT), TTFT at pp512 and
# pp4096 (max_tokens=1, unique prompt heads), aggregate tok/s at conc 4 and 16 (conc_client.py,
# 300-token greedy, ignore_eos), nvidia-smi VRAM/clock samples every 500 ms.
# imp runs spec-off (speculative.mtp_k=0, speculative.ngram=false); vLLM runs its defaults.
# Usage: bash tools/analysis/vllm_gap_sweep.sh [MODELS="dir1 dir2"] [TRIALS=3] [CELLS="base long"] [OUT=/mnt/data/scratch/vllmgap]
# Table: docker run --rm -v $OUT:/o -v tools/analysis:/t:ro imp:toolchain python3 /t/vllm_gap_table.py /o
# imp A/B: IMG_BASE=<image> replaces the vLLM arm by a "base" imp arm (same flags); table with REF=base.
# IMP_SET="k=v ...": extra imp-server --set pairs for both imp arms (e.g. kv_cache.dtype=fp8).
set -u
EXTRA=()
for kv in ${IMP_SET:-}; do EXTRA+=(--set "$kv"); done
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS=${MODELS:-"Qwen3-8B-NVFP4-cortecs Qwen3-8B-FP8 Qwen3-30B-A3B-NVFP4-Modelopt Gemma-4-26B-A4B-it-NVFP4 Nemotron-3-Nano-30B-A3B-NVFP4"}
TRIALS=${TRIALS:-3}
OUT=${OUT:-/mnt/data/scratch/vllmgap}
IMG_IMP=${IMG_IMP:-imp:test}
IMG_VLLM=${IMG_VLLM:-vllm/vllm-openai:v0.31.0}
IMG_BASE=${IMG_BASE:-}
REF=vllm
[ -n "$IMG_BASE" ] && REF=base
CLIENT=${CLIENT:-imp:toolchain}
PORT=${PORT:-8097}
MAXC=${MAXC:-16}
CELLS=${CELLS:-base}  # base: tg128, pp512/4096, conc 4/16; long: pp8192/15000, conc 16 x 1000-token prompts
PPS=""
[[ " $CELLS " == *" base "* ]] && PPS="512 4096"
[[ " $CELLS " == *" long "* ]] && PPS="$PPS 8192 15000"
NAME=vllmgap_srv
mkdir -p "$OUT"
LOG="$OUT/sweep.log"
{
    echo "start $(date -Is) commit=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null)"
    echo "harness md5=$(md5sum < "$0" | cut -d' ' -f1)"
    for img in "$IMG_IMP" "${IMG_BASE:-$IMG_VLLM}" "$CLIENT"; do
        echo "image $img id=$(docker image inspect --format '{{.Id}}' "$img" 2>/dev/null | cut -c8-19)" \
             "digest=$(docker image inspect --format '{{index .RepoDigests 0}}' "$img" 2>/dev/null)" \
             "tree=$(docker image inspect --format '{{index .Config.Labels "imp.tree"}}' "$img" 2>/dev/null)"
    done
} | tee -a "$LOG"

wait_free() {  # previous arm's VRAM can take seconds to return on WSL2: 6 checks, 10 s apart
    local load max_load
    for _ in 1 2 3 4 5 6; do
        if ~/.claude/skills/gpu-stats/gpu-busy-check.sh >/dev/null; then
            load=$(cut -d' ' -f1 /proc/loadavg); max_load=$(( $(nproc) / 4 ))
            if awk -v l="$load" -v m="$max_load" 'BEGIN { exit !(l <= m) }'; then return 0; fi
        fi
        sleep 10
    done
    ~/.claude/skills/gpu-stats/gpu-busy-check.sh | tee -a "$LOG"
    echo "BUSY or host load $(cut -d' ' -f1 /proc/loadavg): abort" | tee -a "$LOG"
    return 1
}

wait_healthy() {  # $1 = seconds
    for _ in $(seq 1 $(( $1 / 2 ))); do
        sleep 2
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
        [ -n "$(docker ps -q -f name=$NAME)" ] || return 1
    done
    return 1
}

client() {  # docker client run, host network, tools mounted read-only
    docker run --rm --network host -e IGNORE_EOS=1 -e MODEL_NAME="$MODEL" -e GEN="${GEN:-300}" -v "$HERE":/t:ro \
        -v "$D":/o "$CLIENT" "$@"
}

run_arm() {  # $1 = engine imp|vllm, $2 = model dir, $3 = trial
    local eng=$1 t=$3
    MODEL=$2
    D="$OUT/$MODEL/$eng-t$t"
    mkdir -p "$D"; chmod 777 "$D"
    wait_free || exit 2
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$D/base_mib"
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw \
        --format=csv,noheader,nounits -lms 500 > "$D/smi.csv" &
    local smi=$!
    docker rm -f $NAME >/dev/null 2>&1
    local common=(-d --name $NAME --gpus all -v "$MODELS_DIR":/models:ro -p "$PORT:$PORT")
    local t0; t0=$(date +%s)
    case "$eng" in
        imp|base) docker run "${common[@]}" "$([ "$eng" = base ] && echo "$IMG_BASE" || echo "$IMG_IMP")" \
                 imp-server --model "/models/$MODEL" --host 0.0.0.0 \
                 --port "$PORT" --max-concurrent "$MAXC" --set "runtime.max_batch_size=$MAXC" \
                 --set speculative.mtp_k=0 --set speculative.ngram=false "${EXTRA[@]}" >/dev/null ;;
        vllm) docker run "${common[@]}" --ipc=host -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 "$IMG_VLLM" \
                 --model "/models/$MODEL" --served-model-name "$MODEL" --host 0.0.0.0 --port "$PORT" \
                 --max-model-len 16384 --max-num-seqs "$MAXC" >/dev/null ;;
    esac
    if ! wait_healthy 1500; then
        echo "== $MODEL $eng t$t: SERVER FAILED" | tee -a "$LOG"
        docker logs $NAME > "$D/server.log" 2>&1
        tail -5 "$D/server.log" | tee -a "$LOG"
        docker rm -f $NAME >/dev/null 2>&1; kill $smi
        return 1
    fi
    echo "$(( $(date +%s) - t0 ))" > "$D/ready_s"
    echo "== $MODEL $eng t$t ready after $(cat "$D/ready_s") s" | tee -a "$LOG"
    GEN=32 client python3 /t/conc_client.py "$PORT" 1 1 "warm-$eng-$t" 0 > "$D/warm.txt" 2>&1
    if [[ " $CELLS " == *" base "* ]]; then
        client python3 /t/serving_kpi.py --url "http://127.0.0.1:$PORT" --model "$MODEL" \
            --endpoint completions --levels 1 --max-tokens 128 --ignore-eos --requests-per-level 5 \
            --warmup 0 --no-power --tag "dec-$eng-$t" --json /o/dec.json > "$D/dec.txt" 2>&1
    fi
    for pp in $PPS; do
        client python3 /t/serving_kpi.py --url "http://127.0.0.1:$PORT" --model "$MODEL" \
            --endpoint completions --levels 1 --max-tokens 1 --prompt-tokens "$pp" \
            --requests-per-level "$(( pp > 4096 ? 4 : 8 ))" --warmup 0 --no-power --tag "pp$pp-$eng-$t" \
            --json "/o/pp$pp.json" > "$D/pp$pp.txt" 2>&1
    done
    GEN=64 client python3 /t/conc_client.py "$PORT" "$MAXC" 1 "cwarm-$eng-$t" 0 > "$D/cwarm.txt" 2>&1
    if [[ " $CELLS " == *" base "* ]]; then
        for c in 4 16; do
            client python3 /t/conc_client.py "$PORT" "$c" 3 "c$c-$eng-$t" 0 > "$D/conc$c.txt" 2>&1
        done
    fi
    if [[ " $CELLS " == *" long "* ]]; then  # 16 streams x 1000-token prompts (prefill + decode mix)
        client python3 /t/conc_client.py "$PORT" 16 3 "c16L-$eng-$t" 1000 > "$D/conc16L.txt" 2>&1
    fi
    docker logs $NAME > "$D/server.log" 2>&1
    docker rm -f $NAME >/dev/null 2>&1
    kill $smi
    echo "== $MODEL $eng t$t done: $(grep -h MEDIAN "$D"/conc*.txt | tr '\n' ' ')" | tee -a "$LOG"
    sleep 5
}

for m in $MODELS; do
    [ -r "$MODELS_DIR/$m/config.json" ] || { echo "== $m: NOMODEL" | tee -a "$LOG"; continue; }
    vllm_ok=1
    for t in $(seq 1 "$TRIALS"); do
        if [ $((t % 2)) -eq 1 ]; then order="imp $REF"; else order="$REF imp"; fi
        for e in $order; do
            if [ "$e" = vllm ] && [ $vllm_ok -eq 0 ]; then continue; fi
            run_arm "$e" "$m" "$t" || { [ "$e" = vllm ] && vllm_ok=0; }
        done
    done
done
echo "end $(date -Is)" | tee -a "$LOG"
