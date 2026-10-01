#!/usr/bin/env bash
# Regenerates tests/fixtures/jpeg (synthetic JPEGs + Pillow RGB references) in a pinned
# Python container. Consumer: tests/test_image_decode.cpp.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(cd "$HERE/../.." && pwd)"
OUT="$TREE/tests/fixtures/jpeg"
PY_IMG=python:3.12-slim
PY_PKGS="pillow==12.3.0"

mkdir -p "$OUT"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$HERE":/h:ro -v "$OUT":/out "$PY_IMG" \
    bash -c "pip -q install --user $PY_PKGS >/dev/null 2>&1 && python /h/gen_jpeg_fixtures.py /out"
