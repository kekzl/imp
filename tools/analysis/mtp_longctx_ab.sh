#!/usr/bin/env bash
# MTP at long context on a DEFAULT imp-server: per arm a fresh server, then mtp_longctx_client.py (repo docs +
# source prompts at each CTX, greedy, GEN tokens) with CONC streams; arms alternate PAIRS times. Per arm: client
# lines, the server's MTP enable line (kv_cap x slots) and why drafting stopped, own peak VRAM, KV blocks.
# Usage: ARMS="label|image|k=v,k=v ..." OUT=/mnt/data/scratch/dir [PAIRS=1] [CTX="8000 20000 40000 80000"]
#        [CORPUS=dir holding docs/internals + src/ (default this tree; pin it: an edited tree changes the prompts)]
#        [CONC=1] [REPS=1 session headers per context] [GEN=384] [MTP_K=] [THINK_BUDGET= (1 = closer never forced)]
#        [STAGGER_MS=0 (> 0: stream i starts that long after i - 1)] tools/analysis/mtp_longctx_ab.sh
set -uo pipefail
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ARMS=${ARMS:?ARMS="label|image|k=v,... ..."}
PAIRS=${PAIRS:-1}
NVFP4=${NVFP4:-Qwen3.8-27B-NVFP4-vllm}
PORT=${PORT:-8211}
CTR=${CTR:-mtplong-$PORT}
OUT=${OUT:?OUT=dir for logs}
mkdir -p "$OUT"
ctr() { grep -E "^$2(\{|[[:space:]])" <<< "$1" | awk '{s += $NF} END {printf "%.0f", s}'; }
echo "harness: md5=$(md5sum "$0" | cut -c1-12) client_md5=$(md5sum "$HERE/mtp_longctx_client.py" | cut -c1-12)" |
    tee -a "$OUT/summary.txt"
for pair in $(seq 1 "$PAIRS"); do
    for arm in $ARMS; do
        IFS='|' read -r label img sets <<< "$arm"
        tag="p${pair}_${label}"
        docker rm -f "$CTR" >/dev/null 2>&1; sleep 5
        bash "$REPO/scripts/require_free_gpu.sh" "mtp_longctx_ab $tag" >/dev/null || { echo "GPU BUSY before $tag"; exit 2; }
        # image "dev" = this tree's build-dev/imp-server inside imp:toolchain (exploration, not an A/B arm)
        if [ "$img" = dev ]; then run=(-v "$REPO":/src imp:toolchain /src/build-dev/imp-server); else run=("$img" imp-server); fi
        docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models -p "$PORT:$PORT" "${run[@]}" \
            --model "/models/$NVFP4" --host 0.0.0.0 --port "$PORT" \
            $(for s in ${sets//,/ }; do printf -- "--set %s " "$s"; done) >/dev/null
        for _ in $(seq 1 600); do sleep 2; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
            [ -n "$(docker ps -q -f name="$CTR")" ] || { echo "$tag: server died"; docker logs "$CTR" 2>&1 | tail -5; exit 3; }; done
        mkdir -p "$OUT/$tag"
        docker run --rm --network host -e CTX="${CTX:-8000 20000 40000 80000}" -e GEN="${GEN:-384}" \
            -e CONC="${CONC:-1}" -e REPS="${REPS:-1}" -e MTP_K="${MTP_K:-}" -e STAGGER_MS="${STAGGER_MS:-0}" -e THINK_BUDGET="${THINK_BUDGET:-}" \
            -e OUT_DIR=/out -v "$OUT/$tag":/out \
            -e REPO_DIR=/corpus -v "${CORPUS:-$REPO}":/corpus:ro \
            -v "$REPO":/repo:ro imp:toolchain python3 /repo/tools/analysis/mtp_longctx_client.py "$PORT" "$tag" \
            2>&1 | tee "$OUT/$tag.client.txt"
        m=$(curl -s "http://127.0.0.1:$PORT/metrics")
        docker logs "$CTR" > "$OUT/$tag.server.log" 2>&1
        en=$(grep -o 'kv_cap=[0-9]* x [0-9]* slot(s) = [0-9.]* MiB' "$OUT/$tag.server.log" | head -1)
        echo "$tag own_peak_mib $(( $(ctr "$m" imp_vram_own_peak_bytes) / 1048576 )) kv_blocks $(ctr "$m" imp_kv_blocks_total) $en" |
            tee -a "$OUT/summary.txt"
        grep -E '^(REQ|AGG)' "$OUT/$tag.client.txt" >> "$OUT/summary.txt"
        grep -E 'mtp-spec|unbind|head KV' "$OUT/$tag.server.log" | sort | uniq -c | sort -rn | head -8 >> "$OUT/summary.txt"
        docker rm -f "$CTR" >/dev/null 2>&1
    done
done
