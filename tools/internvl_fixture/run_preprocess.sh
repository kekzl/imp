#!/usr/bin/env bash
# Regenerates tests/fixtures/internvl/{synth_300x200.png,pixels_448.u8,prompt.txt} from the pinned HF
# InternVL processor. CPU only. MODEL_DIR needs the processor/tokenizer files, no weights are read.
#   tools/internvl_fixture/run_preprocess.sh [MODEL_DIR]   (default $HOME/models/InternVL3_5-2B-HF)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(cd "$HERE/../.." && pwd)"
MODEL="$(cd "${1:-$HOME/models/InternVL3_5-2B-HF}" && pwd)"
OUT="$TREE/tests/fixtures/internvl"
PY_IMG=python:3.12-slim
PY_PKGS="transformers==5.17.0 torch==2.14.1 torchvision==0.29.1 numpy==2.5.3 pillow==12.3.0 safetensors==0.8.0"
PY_INDEX="--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple"

mkdir -p "$OUT"
docker volume create imp-qwen3vl-video-pip >/dev/null
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PIP_CACHE_DIR=/pip \
    -v imp-qwen3vl-video-pip:/pip -v "$HERE":/h:ro -v "$MODEL":/model:ro -v "$OUT":/out "$PY_IMG" \
    bash -c "pip -q install --user $PY_INDEX $PY_PKGS >/dev/null 2>&1 && python /h/gen_preprocess.py /model /out"
