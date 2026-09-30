#!/usr/bin/env bash
# Tokenizer parity: imp token ids vs HF tokenizers (tokenizer.json) on a fixed 1196-string corpus.
# CPU only. Needs a dev build of the tree under test (make dev -> build-dev/libimp.a).
#
#   tools/tokenizer_parity/run.sh [--tree DIR] [--chat] [--diff] MODEL...
#
# MODEL: a SafeTensors dir with tokenizer.json, or FILE.gguf=DIR (DIR's tokenizer.json is the
# reference). --tree: the imp checkout whose build-dev/libimp.a is measured (default: this one).
# --chat: ChatTemplate::apply vs transformers apply_chat_template on 3 conversations.
# --diff: print the first 20 differing records.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(cd "$HERE/../.." && pwd)"
CHAT=0
DIFF=0
while [ $# -gt 0 ]; do
    case "$1" in
        --tree) TREE="$(cd "$2" && pwd)"; shift 2 ;;
        --chat) CHAT=1; shift ;;
        --diff) DIFF=1; shift ;;
        *) break ;;
    esac
done
[ $# -gt 0 ] || { sed -n '2,11p' "$0"; exit 2; }

PY_IMG=python:3.12-slim
PY_PKGS="tokenizers==0.23.2 transformers==5.17.0 jinja2==3.1.6"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$HERE/parity.py" "$WORK/"
docker volume create imp-tokparity-pip >/dev/null

py() {  # py <args...>: parity.py in the pinned container, models mounted read-only at their paths
    docker run --rm -v "$WORK":/w -v imp-tokparity-pip:/root/.cache/pip -v /home:/home:ro -w /w "$PY_IMG" \
        bash -c "pip -q install $PY_PKGS >/dev/null 2>&1; python parity.py $*"
}

docker run --rm -v "$TREE":/src -v "$HERE":/h:ro -v "$WORK":/w -w /src --entrypoint bash imp:toolchain -c \
    "g++ -O2 -std=gnu++23 -I/src/src -I/src/tools -I/src/include \
       -isystem /usr/local/cuda/targets/x86_64-linux/include /h/tok_dump.cpp -o /w/tok_dump \
       -Wl,-rpath,/usr/local/cuda/targets/x86_64-linux/lib build-dev/libimp.a \
       -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
       -lcublas -lcublasLt -lculibos -lcudadevrt -lcudart_static -lrt -lpthread -ldl"
echo "tree: $TREE @ $(git -C "$TREE" rev-parse --short HEAD)"
py corpus corpus.bin

for spec in "$@"; do
    model="${spec%%=*}"
    ref="${spec#*=}"
    [ "$ref" = "$spec" ] && ref="$model"
    name="$(basename "$model")"
    imp_env=()
    if [ "$CHAT" = 1 ]; then
        imp_env=(-e CHAT=1)
        py chat "$ref" > "$WORK/hf.txt" 2>"$WORK/hf.err" || { printf '%-44s ERROR hf: %s\n' "$name" "$(tail -1 "$WORK/hf.err")"; continue; }
    else
        py hf "$ref/tokenizer.json" > "$WORK/hf.txt" 2>"$WORK/hf.err" || { printf '%-44s ERROR hf: %s\n' "$name" "$(tail -1 "$WORK/hf.err")"; continue; }
    fi
    docker run --rm -v "$WORK":/w -v /home:/home:ro -w /w "${imp_env[@]}" --entrypoint /w/tok_dump imp:toolchain \
        "$model" corpus.bin > "$WORK/imp.raw" 2>&1 || { printf '%-44s ERROR imp: %s\n' "$name" "$(tail -1 "$WORK/imp.raw")"; continue; }
    grep '^@@ ' "$WORK/imp.raw" | cut -c4- > "$WORK/imp.txt" || true
    extra=""
    [ "$DIFF" = 1 ] && extra="$ref/tokenizer.json"
    printf '%-44s %s\n' "$name" "$(py compare imp.txt hf.txt $extra)"
done
