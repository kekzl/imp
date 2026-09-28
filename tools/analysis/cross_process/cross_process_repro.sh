#!/bin/bash
# Cross-process determinism repro (#2168): N fresh server processes, one identical request each,
# IMP_DETERMINISTIC=1, prefix cache off. Prints the first-token top-2 logprobs per process.
# Exit 0 = all processes identical, 1 = more than one distinct value, 2 = setup error.
#
# usage: cross_process_repro.sh --model <dir under $MODELS> [--image IMG | --tree DIR] [--n 6] [--build DIR]
#                               [--dump-root DIR] [--set k=v ...]
#   --image  released image (default imp:test); --tree  a repo checkout with a build dir
#   --build  build dir under --tree (default build-dev; e.g. an IMP_ALLOC_POISON_BYTE build)
#   --dump-root  per-process diagnostics.dump_hidden_dir=<root>/p<i> (all layers), for dump_first_diff.py
# env: MODELS (default $HOME/models), PORT (default 8091)
set -u
MODELS=${MODELS:-$HOME/models}; PORT=${PORT:-8091}
IMG=imp:test; TREE=""; BUILD=build-dev; N=6; MODEL=""; DUMP_ROOT=""; SETS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --image) IMG=$2; shift 2 ;;
        --tree) TREE=$(cd "$2" && pwd); shift 2 ;;
        --build) BUILD=$2; shift 2 ;;
        --n) N=$2; shift 2 ;;
        --dump-root) DUMP_ROOT=$2; shift 2 ;;
        --set) SETS+=("$2"); shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done
[ -n "$MODEL" ] || { echo "--model required" >&2; exit 2; }
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
CORPUS=$ROOT/tools/analysis/ppl_corpus_45k.txt
[ -f "$CORPUS" ] || "$ROOT/tools/analysis/make_ppl_corpus.sh" "$CORPUS" >/dev/null || exit 2
PROMPT="Summarize the following text in three sentences.

$(head -c 1400 "$CORPUS")"
BODY_FILE=$(mktemp); OUT=$(mktemp -d); trap 'rm -rf "$BODY_FILE" "$OUT"; docker rm -f imp-xproc >/dev/null 2>&1' EXIT
NAME=imp-xproc

for i in $(seq 1 "$N"); do
    SET="server.prefix_cache=false ${SETS[*]}"
    VOL=()
    if [ -n "$DUMP_ROOT" ]; then
        mkdir -p "$DUMP_ROOT/p$i"
        VOL=(-v "$(cd "$DUMP_ROOT/p$i" && pwd):/dump")
        SET="$SET diagnostics.dump_hidden_dir=/dump"
    fi
    docker rm -f $NAME >/dev/null 2>&1
    if [ -n "$TREE" ]; then
        docker run -d --init --name $NAME --user "$(id -u):$(id -g)" --gpus all -v "$MODELS:/models" -v "$TREE:/src" \
            "${VOL[@]}" -p "127.0.0.1:$PORT:8080" -e HOME=/tmp -e IMP_DETERMINISTIC=1 -e IMP_SET="$SET" \
            --entrypoint bash imp:toolchain \
            -c "export PATH=/src/$BUILD:\$PATH; exec /src/docker-entrypoint.sh imp-server --model /models/$MODEL" >/dev/null
    else
        docker run -d --init --name $NAME --user "$(id -u):$(id -g)" --gpus all -v "$MODELS:/models" "${VOL[@]}" \
            -p "127.0.0.1:$PORT:8080" -e HOME=/tmp -e IMP_DETERMINISTIC=1 -e IMP_SET="$SET" \
            "$IMG" --model "/models/$MODEL" >/dev/null
    fi
    for _ in $(seq 1 200); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
        docker ps -q -f name=$NAME | grep -q . || { echo "process $i: server died"; docker logs $NAME 2>&1 | tail -5; exit 2; }
        sleep 3
    done
    M=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].id')
    jq -n --arg m "$M" --arg p "$PROMPT" '{model:$m,messages:[{role:"user",content:$p}],max_tokens:1,temperature:0,
        logprobs:true,top_logprobs:5,chat_template_kwargs:{enable_thinking:false}}' > "$BODY_FILE"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$BODY_FILE" > "$OUT/p$i.json"
    V=$(jq -c '[.choices[0].logprobs.content[0].top_logprobs[:2][] | [.token, (.logprob*10000|round/10000)]]' "$OUT/p$i.json")
    [ -n "$V" ] && [ "$V" != "null" ] || { echo "process $i: no logprobs"; head -c 300 "$OUT/p$i.json"; exit 2; }
    echo "$V" > "$OUT/p$i.val"
    printf "process %d: %s\n" "$i" "$V"
done
docker rm -f $NAME >/dev/null 2>&1

DISTINCT=$(cat "$OUT"/p*.val | sort -u | wc -l)
echo "distinct values: $DISTINCT of $N processes"
cat "$OUT"/p*.val | sort | uniq -c | sed 's/^/  /'
[ "$DISTINCT" -eq 1 ]
