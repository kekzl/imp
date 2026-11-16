# Plot container for the roofline pipeline (host stays clean - no matplotlib
# on the host). Built on demand by `roofline plot`.
# Base image and deps are content-pinned (AUDIT_ARCH_2026 H-6): a tag is a
# mutable ref, and a bare `==` still lets the index serve different bytes.
# Digest: docker buildx imagetools inspect python:3.14-slim
FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
