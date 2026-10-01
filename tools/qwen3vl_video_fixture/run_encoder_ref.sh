#!/usr/bin/env bash
# Regenerates tests/fixtures/qwen3vl_video/red_panda/encoder_ref*: HF vision tower (BF16, FP16, FP32, CPU) on
# the committed frames. CPU only, ~2 min.
#   tools/qwen3vl_video_fixture/run_encoder_ref.sh [MODEL_DIR]   (default $HOME/models/Qwen3-VL-4B-Instruct)
# Frames: 8 of red-panda.mp4 (OpenGVLab/InternVL3_5-2B-HF examples/, sha256 d921c07b...), indices
# 0 128 255 383 511 639 766 894 of 895 at 30 fps, ffmpeg scale=192:320:flags=bicubic, rgb24 PNG.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(cd "$HERE/../.." && pwd)"
MODEL="$(cd "${1:-$HOME/models/Qwen3-VL-4B-Instruct}" && pwd)"
OUT="$TREE/tests/fixtures/qwen3vl_video/red_panda"
PY_IMG=python:3.12-slim
PY_PKGS="transformers==5.17.0 torch==2.14.1 torchvision==0.29.1 numpy==2.5.3 pillow==12.3.0 safetensors==0.8.0"
PY_INDEX="--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple"

docker volume create imp-qwen3vl-video-pip >/dev/null
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PIP_CACHE_DIR=/pip \
    -v imp-qwen3vl-video-pip:/pip -v "$HERE":/h:ro -v "$MODEL":/model:ro -v "$OUT":/out "$PY_IMG" \
    bash -c "pip -q install --user $PY_INDEX $PY_PKGS >/dev/null 2>&1 && python /h/gen_encoder_ref.py /model /out /out"
