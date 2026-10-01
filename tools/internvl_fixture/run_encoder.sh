#!/usr/bin/env bash
# Regenerates tests/fixtures/internvl/tiny_{tower,stages}.st (safetensors format; *.safetensors is gitignored) from the pinned HF InternVL
# modules (random tiny tower, FP32, CPU). No checkpoint needed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(cd "$HERE/../.." && pwd)"
OUT="$TREE/tests/fixtures/internvl"
PY_IMG=python:3.12-slim
PY_PKGS="transformers==5.17.0 torch==2.14.1 torchvision==0.29.1 numpy==2.5.3 pillow==12.3.0 safetensors==0.8.0"
PY_INDEX="--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple"

mkdir -p "$OUT"
docker volume create imp-qwen3vl-video-pip >/dev/null
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PIP_CACHE_DIR=/pip \
    -v imp-qwen3vl-video-pip:/pip -v "$HERE":/h:ro -v "$OUT":/out "$PY_IMG" \
    bash -c "pip -q install --user $PY_INDEX $PY_PKGS >/dev/null 2>&1 && python /h/gen_encoder.py /out"
