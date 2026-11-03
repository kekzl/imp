#!/usr/bin/env bash
# Multi-turn chat A/B on a DEFAULT imp-server (no --max-batch): per arm a fresh server, then the
# conversations in MODES (mtp_multiturn_probe.py, TURNS turns each), then the server log's MTP unbind
# reasons per request (--parse). Arms alternate PAIRS times. Lines per arm: probe SUMMARY rows +
# UNBIND_TOTAL + own peak VRAM.
# MODES: "label:TEMP:SIDE:CONVS ..." (default "greedy:0:0:1 sampled:0.7:0:1"; SIDE=1 a UI task request
# between turns, CONVS=2 two chats alternate turns, CONVS=ov two chats with overlapping turns via
# mtp_overlap_probe.py, its per-turn reasons in <tag>_<label>.reasons.txt).
# Usage: ARMS="label|image|k=v,k=v ..." [PAIRS=3] [TURNS=6] [GEN=1500] [MODES=...] OUT=/mnt/data/scratch/dir \
#        tools/analysis/mtp_multiturn_ab.sh
set -uo pipefail
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
ARMS=${ARMS:?ARMS="label|image|k=v,... ..."}
PAIRS=${PAIRS:-3}
TURNS=${TURNS:-6}
GEN=${GEN:-1500}
MODES=${MODES:-"greedy:0:0:1 sampled:0.7:0:1"}
NVFP4=${NVFP4:-Qwen3.8-27B-NVFP4-vllm}
PORT=${PORT:-8220}
CTR=${CTR:-mtpmt-$PORT}  # not NAME: login shells export NAME=<hostname>
OUT=${OUT:?OUT=dir for logs}
mkdir -p "$OUT"
probe() {  # $1 tag, $2 temperature, $3 side, $4 convs
    if [ "$4" = ov ]; then
        docker run --rm --network host -e MODEL_NAME="$NVFP4" -e TURNS="$TURNS" -e GEN="$GEN" -e OUT_JSON="/out/$1.json" \
            -v "$HERE":/t:ro -v "$OUT":/out imp:toolchain python3 /t/mtp_overlap_probe.py "$PORT" "$1" > "$OUT/$1.txt" 2>&1
        return
    fi
    docker run --rm --network host -e MODEL_NAME="$NVFP4" -e TURNS="$TURNS" -e GEN="$GEN" -e TEMP="$2" \
        -e SIDE="$3" -e CONVS="$4" -e OUT_JSON="/out/$1.json" -v "$HERE":/t:ro -v "$OUT":/out imp:toolchain \
        python3 /t/mtp_multiturn_probe.py "$PORT" "$1" > "$OUT/$1.txt" 2>&1
}
echo "harness: md5=$(md5sum "$0" "$HERE/mtp_multiturn_probe.py" "$HERE/mtp_overlap_probe.py" | cut -c1-12 | tr '\n' ' ')" | tee -a "$OUT/summary.txt"
for pair in $(seq 1 "$PAIRS"); do
    for arm in $ARMS; do
        IFS='|' read -r label img sets <<< "$arm"
        tag="p${pair}_${label}"
        docker rm -f "$CTR" >/dev/null 2>&1; sleep 5
        bash "$HERE/../../scripts/require_free_gpu.sh" "mtp_multiturn_ab $tag" >/dev/null || { echo "GPU BUSY before $tag"; exit 2; }
        docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models -p "$PORT:$PORT" "$img" imp-server \
            --model "/models/$NVFP4" --host 0.0.0.0 --port "$PORT" \
            $(for s in ${sets//,/ }; do printf -- "--set %s " "$s"; done) >/dev/null
        for _ in $(seq 1 600); do sleep 2; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
            [ -n "$(docker ps -q -f name="$CTR")" ] || { echo "$tag: server died"; docker logs "$CTR" 2>&1 | tail -5; exit 3; }; done
        for mode in $MODES; do
            IFS=':' read -r m temp side convs <<< "$mode"
            probe "${tag}_$m" "$temp" "$side" "$convs"
        done
        curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/$tag.metrics.txt"
        peak=$(grep -E '^imp_vram_own_peak_bytes' "$OUT/$tag.metrics.txt" | awk '{s += $NF} END {printf "%.0f", s}')
        docker logs "$CTR" > "$OUT/$tag.server.log" 2>&1
        docker rm -f "$CTR" >/dev/null 2>&1
        docker run --rm -v "$HERE":/t:ro -v "$OUT":/out imp:toolchain \
            python3 /t/mtp_multiturn_probe.py --parse "/out/$tag.server.log" > "$OUT/$tag.unbind.txt" 2>&1
        for mode in $MODES; do
            IFS=':' read -r m _ _ convs <<< "$mode"
            [ "$convs" = ov ] || continue
            docker run --rm -v "$HERE":/t:ro -v "$OUT":/out imp:toolchain python3 /t/mtp_overlap_probe.py --parse \
                "/out/$tag.server.log" "/out/${tag}_$m.json" > "$OUT/${tag}_$m.reasons.txt" 2>&1
            echo "${tag}_$m $(grep REASONS_TOTAL "$OUT/${tag}_$m.reasons.txt")" | tee -a "$OUT/summary.txt"
        done
        for mode in $MODES; do grep SUMMARY "$OUT/${tag}_${mode%%:*}.txt" | tee -a "$OUT/summary.txt"; done
        echo "$tag $(grep UNBIND_TOTAL "$OUT/$tag.unbind.txt") own_peak_mib $(( peak / 1048576 ))" | tee -a "$OUT/summary.txt"
    done
done
