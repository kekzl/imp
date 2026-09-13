#!/usr/bin/env bash
# Alternating N-arm A/B at CONC concurrent streams, fresh server per arm, identical work per
# arm (ignore_eos: every request emits exactly GEN tokens, so an early-stopping arm doesn't run
# its wave tail at lower concurrency).
# ARMS "name=--set k=v ...;..." per-arm sets on top of COMMON; CONC WAVES TRIALS MBS SEQ =
# client/server geometry.
set -u
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=${MODEL:-/models/Qwen3.8-27B-NVFP4-vllm}
PORT=8090
CONC=${CONC:-24}
WAVES=${WAVES:-3}
TRIALS=${TRIALS:-3}
MBS=${MBS:-32}
SEQ=${SEQ:-4096}
COMMON=${COMMON:-"--set speculative.verify_smallm=true --set speculative.mtp_k=2 --set speculative.ngram=false"}
ARMS=${ARMS:-'OFF=--set speculative.batch_verify=false;ON=--set speculative.batch_verify=true --set speculative.factored_spare=true'}
LOG="${TMPDIR:-/tmp}/ab_arms_${CONC}.log"
: > "$LOG"

start_server() {  # $1 = extra --set args
    docker rm -f imp-ab >/dev/null 2>&1
    # shellcheck disable=SC2086
    docker run -d --name imp-ab --gpus all -v ${MODELS_DIR}:/models \
        -p ${PORT}:${PORT} imp:test imp-server --model $MODEL --port $PORT \
        --host 0.0.0.0 --max-concurrent $CONC --think-budget 0 \
        --set runtime.max_batch_size=$MBS --set runtime.max_seq_len=$SEQ \
        $COMMON $1 >/dev/null
    for _ in $(seq 1 240); do
        sleep 2
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        if [ -z "$(docker ps -q -f name=imp-ab)" ]; then
            echo "server died:" | tee -a "$LOG"
            docker logs imp-ab 2>&1 | tail -20 | tee -a "$LOG"
            return 1
        fi
    done
    echo "server never became healthy" | tee -a "$LOG"
    return 1
}

run_arm() {  # $1 = arm name, $2 = extra sets, $3 = trial
    ~/.claude/skills/gpu-stats/gpu-busy-check.sh >/dev/null || {
        echo "GPU BUSY before $1 - aborting" | tee -a "$LOG"; exit 2; }
    start_server "$2" || exit 3
    echo "== arm $1 trial $3 ==" | tee -a "$LOG"
    IGNORE_EOS=1 python3 "$HERE/conc_client.py" $PORT $CONC $WAVES "$1$3" 2>&1 | tee -a "$LOG"
    # Proof the arm ran what it claims: the plan it got and the verify's own
    # counters (drafted per verify step says how many requests really drafted).
    docker stop imp-ab >/dev/null 2>&1
    docker logs imp-ab 2>&1 | grep -E "clamped|KV: live pass|\[spec-ngram\] verify_steps" | tail -3 | tee -a "$LOG"
    docker rm -f imp-ab >/dev/null 2>&1
    sleep 3
}

for t in $(seq 1 $TRIALS); do
    (IFS=';'; for arm in $ARMS; do
        echo "${arm%%=*}|${arm#*=}"
    done) | while IFS='|' read -r name sets; do
        run_arm "$name" "$sets" "$t"
    done
done
echo "=== summary ===" | tee -a "$LOG"
grep -H "MEDIAN\|== arm\|verify_steps" "$LOG" | tail -40
