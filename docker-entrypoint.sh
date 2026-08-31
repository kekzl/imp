#!/bin/bash
set -e

# Translate environment variables to imp-server CLI flags.
# Usage: docker run ... -e IMP_MODEL=/models/model.gguf imp:latest

# If the first argument is a flag (e.g. --help, --model, --version), it is meant
# for the default command, not a command name. Prepend imp-server so the flag
# reaches the real binary instead of being exec'd as a command.
if [ "${1#-}" != "$1" ]; then
    set -- imp-server "$@"
fi

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

# Host — default 0.0.0.0 inside container.
#
# This one stays 0.0.0.0: the container's own loopback is not reachable through
# a published port, so binding it here would break every mapping rather than
# secure anything. What decides exposure is the HOST side of the port
# publication, and docker-compose.yml binds that to 127.0.0.1 by default
# (#1619).
if [ -n "$IMP_HOST" ]; then
    args+=(--host "$IMP_HOST")
elif [ "$CMD" = "imp-server" ]; then
    args+=(--host "0.0.0.0")
fi

# API key. The entrypoint translated fourteen env vars and not this one, so a
# compose deployment had no way to turn authentication on at all (#1619).
if [ -n "$IMP_API_KEY" ]; then
    args+=(--api-key "$IMP_API_KEY")
fi

# Trusted proxies for X-Forwarded-For (#1614). Behind a reverse proxy this is
# what makes per-client rate limiting work; without it the limit keys on the
# proxy's address, which is one bucket for everyone.
if [ -n "$IMP_TRUSTED_PROXY" ]; then
    args+=(--trusted-proxy "$IMP_TRUSTED_PROXY")
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

# Think budget
if [ -n "$IMP_THINK_BUDGET" ]; then
    args+=(--think-budget "$IMP_THINK_BUDGET")
fi

# Models directory
if [ -n "$IMP_MODELS_DIR" ]; then
    args+=(--models-dir "$IMP_MODELS_DIR")
elif [ "$CMD" = "imp-server" ] && [ -d "/models" ]; then
    args+=(--models-dir "/models")
fi

# The maintained configuration surface, bridged generically. Every name above is
# hand-written and frozen, so until this existed a compose deployment could not
# reach any imp.conf key that had not been given one - which is every key added
# since the config system landed (sparse attention, growable KV, MTP depth, the
# GDN state dtype, the batching knobs). Two names instead of one per key.
if [ -n "$IMP_CONFIG" ]; then
    args+=(--config "$IMP_CONFIG")
fi

# One `section.key=value` per whitespace or newline, so a compose YAML block
# scalar works. A value containing a space is not expressible here; pass that
# one in `command:` instead.
#
# `set -f` because the unquoted expansion is wanted for the word splitting and
# NOT for globbing: without it a value holding `*`, `?` or `[` expands against
# the container's working directory, so `--set` would carry filenames.
if [ -n "$IMP_SET" ]; then
    set -f
    for kv in $IMP_SET; do
        args+=(--set "$kv")
    done
    set +f
fi

# The legacy KV names do not lose to IMP_SET, they outrank it in either order:
# --kv-fp8 sets the dtype enum directly and the engine consults kv_cache.dtype
# only while that enum is still FP16. Silent precedence is what makes a stale
# name expensive, so say it rather than let the IMP_SET line look applied.
case "$IMP_SET" in
    *kv_cache.dtype=*)
        if is_true "$IMP_KV_FP8" || is_true "$IMP_KV_INT8"; then
            echo "imp: IMP_KV_FP8/IMP_KV_INT8 override kv_cache.dtype from IMP_SET;" \
                 "unset the legacy name for IMP_SET to take effect" >&2
        fi
        ;;
esac

exec "$CMD" "${args[@]}" "$@"
