#!/bin/sh
# CI CUDA compiler launcher. It ran the heavy CUTLASS TU (7.85 GB peak RSS) under a lock; that TU is gone
# since 2026-10-31 and every other TU stays <= 1.27 GB, so it only passes the command through.
exec "$@"
