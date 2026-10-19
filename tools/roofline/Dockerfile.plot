# Plot container for the roofline pipeline (host stays clean - no matplotlib
# on the host). Built on demand by `roofline plot`.
# Base image and deps are content-pinned (AUDIT_ARCH_2026 H-6): a tag is a
# mutable ref, and a bare `==` still lets the index serve different bytes.
# Digest: docker buildx imagetools inspect python:3.14-slim
FROM python:3.14-slim@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
