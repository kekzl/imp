#!/usr/bin/env bash
# MTP on a DEFAULT imp-server (no --max-batch): 1 stream real chats (seq_stream_client.py, 6 prompts x 400 greedy
# tokens x ROUNDS) then c4 and c32 bursts (conc_client.py, WAVES waves, wave 0 dropped), fresh server per arm,
# arms alternate PAIRS times. Per arm: tok/s, spec counters, KV blocks, resolved batch, own peak VRAM.
# Usage: ARMS="label|image|k=v,k=v ..." [PAIRS=3] [ROUNDS=1] [WAVES=4] [LEVELS="4 32"] OUT=/mnt/data/scratch/dir \
#        [GEN=300 burst tokens] tools/analysis/mtp_concurrent_ab.sh
set -uo pipefail
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
ARMS=${ARMS:?ARMS="label|image|k=v,... ..."}
PAIRS=${PAIRS:-3}
ROUNDS=${ROUNDS:-1}
WAVES=${WAVES:-4}
LEVELS=${LEVELS:-"4 32"}
NVFP4=${NVFP4:-Qwen3.8-27B-NVFP4-vllm}
PORT=${PORT:-8210}
CTR=${CTR:-mtpconc-$PORT}  # not NAME: login shells export NAME=<hostname>
OUT=${OUT:?OUT=dir for logs}
mkdir -p "$OUT"
ctr() {  # $1 = metrics text, $2 = metric name: sum of its samples
    grep -E "^$2(\{|[[:space:]])" <<< "$1" | awk '{s += $NF} END {printf "%.0f", s}'
}
echo "harness: md5=$(md5sum "$0" | cut -c1-12)" | tee -a "$OUT/summary.txt"
for pair in $(seq 1 "$PAIRS"); do
    for arm in $ARMS; do
        IFS='|' read -r label img sets <<< "$arm"
        tag="p${pair}_${label}"
        docker rm -f "$CTR" >/dev/null 2>&1; sleep 5
        bash "$HERE/../../scripts/require_free_gpu.sh" "mtp_concurrent_ab $tag" >/dev/null || { echo "GPU BUSY before $tag"; exit 2; }
        docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models -p "$PORT:$PORT" "$img" imp-server \
            --model "/models/$NVFP4" --host 0.0.0.0 --port "$PORT" \
            $(for s in ${sets//,/ }; do printf -- "--set %s " "$s"; done) >/dev/null
        for _ in $(seq 1 600); do sleep 2; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
            [ -n "$(docker ps -q -f name="$CTR")" ] || { echo "$tag: server died"; docker logs "$CTR" 2>&1 | tail -5; exit 3; }; done
        curl -s "http://127.0.0.1:$PORT/health" > "$OUT/$tag.health.json"
        docker run --rm --network host -e IGNORE_EOS=1 -e MODEL_NAME="$NVFP4" -v "$HERE":/t:ro imp:toolchain \
            python3 /t/conc_client.py "$PORT" 1 1 warm >/dev/null 2>&1
        m0=$(curl -s "http://127.0.0.1:$PORT/metrics")
        docker run --rm --network host -e ROUNDS="$ROUNDS" -e MODEL_NAME="$NVFP4" -v "$HERE":/t:ro imp:toolchain \
            python3 /t/seq_stream_client.py "$PORT" "$tag" > "$OUT/$tag.c1.txt" 2>&1
        m1=$(curl -s "http://127.0.0.1:$PORT/metrics")
        d=$(( $(ctr "$m1" imp_spec_drafted_total) - $(ctr "$m0" imp_spec_drafted_total) ))
        a=$(( $(ctr "$m1" imp_spec_accepted_total) - $(ctr "$m0" imp_spec_accepted_total) ))
        c1=$(awk '/MEDIAN/ {print $4}' "$OUT/$tag.c1.txt")
        line="$tag c1 $c1 drafted $d accepted $a"
        for c in $LEVELS; do
            docker run --rm --network host -e MODEL_NAME="$NVFP4" -e GEN="${GEN:-300}" -v "$HERE":/t:ro imp:toolchain \
                python3 /t/conc_client.py "$PORT" "$c" "$WAVES" "$tag" > "$OUT/$tag.c$c.txt" 2>&1
            med=$(grep -E '^wave[1-9]' "$OUT/$tag.c$c.txt" | sed -E 's/.*= ([0-9.]+) tok\/s.*/\1/' | sort -n |
                  awk '{v[NR]=$1} END {print NR ? v[int((NR+1)/2)] : 0}')
            line="$line c$c $med"
        done
        m2=$(curl -s "http://127.0.0.1:$PORT/metrics"); echo "$m2" > "$OUT/$tag.metrics.txt"
        peak=$(ctr "$m2" imp_vram_own_peak_bytes); blocks=$(ctr "$m2" imp_kv_blocks_total)
        line="$line own_peak_mib $(( peak / 1048576 )) kv_blocks $blocks"
        echo "$line" | tee -a "$OUT/summary.txt"
        docker logs "$CTR" > "$OUT/$tag.server.log" 2>&1
        docker rm -f "$CTR" >/dev/null 2>&1
    done
done
