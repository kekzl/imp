#!/usr/bin/env bash
# Regenerates tests/fixtures/internvl/test_cat_pil_{projector_fp32.f16,pixels_448.u8,ref.txt}: the real InternVL3.5-2B
# tower + projector in HF FP32 on CPU for ~/models/gemma-3-4b-vl/test_cat_pil.png (sha256 in the txt; Pillow decode of test_cat.jpg, #2381).
#   tools/internvl_fixture/run_real_ref.sh [MODEL_DIR] [IMAGE]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(cd "$HERE/../.." && pwd)"
MODEL="$(cd "${1:-$HOME/models/InternVL3_5-2B-HF}" && pwd)"
IMAGE="$(realpath "${2:-$HOME/models/gemma-3-4b-vl/test_cat_pil.png}")"
OUT="$TREE/tests/fixtures/internvl"
PY_IMG=python:3.12-slim
PY_PKGS="transformers==5.17.0 torch==2.14.1 torchvision==0.29.1 numpy==2.5.3 pillow==12.3.0 safetensors==0.8.0"
PY_INDEX="--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple"

docker volume create imp-qwen3vl-video-pip >/dev/null
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PIP_CACHE_DIR=/pip \
    -v imp-qwen3vl-video-pip:/pip -v "$HERE":/h:ro -v "$MODEL":/model:ro -v "$IMAGE":/img/"$(basename "$IMAGE")":ro \
    -v "$OUT":/out "$PY_IMG" \
    bash -c "pip -q install --user $PY_INDEX $PY_PKGS >/dev/null 2>&1 && python /h/gen_real_ref.py /model /img/$(basename "$IMAGE") /out"
