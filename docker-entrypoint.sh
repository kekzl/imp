#!/bin/bash
set -e

# Translate environment variables to imp-server CLI flags.
# Usage: docker run ... -e IMP_MODEL=/models/model.gguf imp:latest

CMD="$1"
shift 2>/dev/null || true

# Default to imp-server if no command given
if [ -z "$CMD" ]; then
    CMD="imp-server"
fi

# If the command is not imp-server or imp-cli, exec directly (e.g. bash, sh)
case "$CMD" in
    imp-server|imp-cli) ;;
    *) exec "$CMD" "$@" ;;
esac

args=()

# Model path
if [ -n "$IMP_MODEL" ]; then
    args+=(--model "$IMP_MODEL")
fi

# Host — default 0.0.0.0 inside container
if [ -n "$IMP_HOST" ]; then
    args+=(--host "$IMP_HOST")
elif [ "$CMD" = "imp-server" ]; then
    args+=(--host "0.0.0.0")
fi

# Port
if [ -n "$IMP_PORT" ]; then
    args+=(--port "$IMP_PORT")
fi

# Max tokens
if [ -n "$IMP_MAX_TOKENS" ]; then
    args+=(--max-tokens "$IMP_MAX_TOKENS")
fi

# GPU layers
if [ -n "$IMP_GPU_LAYERS" ]; then
    args+=(--gpu-layers "$IMP_GPU_LAYERS")
fi

# Device ID
if [ -n "$IMP_DEVICE" ]; then
    args+=(--device "$IMP_DEVICE")
fi

# Chat template
if [ -n "$IMP_CHAT_TEMPLATE" ]; then
    args+=(--chat-template "$IMP_CHAT_TEMPLATE")
fi

# Boolean flags — accept 1 or true
is_true() { [ "$1" = "1" ] || [ "$1" = "true" ] || [ "$1" = "TRUE" ]; }

if is_true "$IMP_KV_FP8"; then
    args+=(--kv-fp8)
fi

if is_true "$IMP_KV_INT8"; then
    args+=(--kv-int8)
fi

if [ "$IMP_DECODE_NVFP4" = "1" ]; then
    args+=(--decode-nvfp4)
elif [ "$IMP_DECODE_NVFP4" = "2" ]; then
    args+=(--decode-nvfp4-only)
elif [ "$IMP_DECODE_NVFP4" = "0" ]; then
    args+=(--no-nvfp4)
fi

if is_true "$IMP_DECODE_NVFP4_ONLY"; then
    args+=(--decode-nvfp4-only)
fi

if is_true "$IMP_NO_NVFP4"; then
    args+=(--no-nvfp4)
fi

if is_true "$IMP_NO_CUDA_GRAPHS"; then
    args+=(--no-cuda-graphs)
fi

if is_true "$IMP_SSM_FP16"; then
    args+=(--ssm-fp16)
fi

# Vision encoder
if [ -n "$IMP_MMPROJ" ]; then
    args+=(--mmproj "$IMP_MMPROJ")
fi

# Prefill chunk size
if [ -n "$IMP_PREFILL_CHUNK_SIZE" ]; then
    args+=(--prefill-chunk-size "$IMP_PREFILL_CHUNK_SIZE")
fi

# Models directory
if [ -n "$IMP_MODELS_DIR" ]; then
    args+=(--models-dir "$IMP_MODELS_DIR")
elif [ "$CMD" = "imp-server" ] && [ -d "/models" ]; then
    args+=(--models-dir "/models")
fi

exec "$CMD" "${args[@]}" "$@"
