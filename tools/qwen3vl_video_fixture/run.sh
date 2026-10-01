#!/usr/bin/env bash
# Regenerates tests/fixtures/qwen3vl_video/ from the pinned HF Qwen3-VL processor. CPU only.
#   tools/qwen3vl_video_fixture/run.sh [MODEL_DIR]   (default $HOME/models/Qwen3-VL-4B-Instruct)
# MODEL_DIR needs tokenizer + config + preprocessor_config.json; weights are never read.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(cd "$HERE/../.." && pwd)"
MODEL="$(cd "${1:-$HOME/models/Qwen3-VL-4B-Instruct}" && pwd)"
OUT="$TREE/tests/fixtures/qwen3vl_video"
PY_IMG=python:3.12-slim
PY_PKGS="transformers==5.17.0 torch==2.14.1 torchvision==0.29.1 numpy==2.5.3"
PY_INDEX="--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple"

mkdir -p "$OUT"
docker volume create imp-qwen3vl-video-pip >/dev/null
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PIP_CACHE_DIR=/pip \
    -v imp-qwen3vl-video-pip:/pip -v "$HERE":/h:ro -v "$MODEL":/model:ro -v "$OUT":/out "$PY_IMG" \
    bash -c "pip -q install --user $PY_INDEX $PY_PKGS >/dev/null 2>&1 && python /h/gen.py /model /out"
