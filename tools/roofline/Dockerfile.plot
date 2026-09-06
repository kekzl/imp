# Plot container for the roofline pipeline (host stays clean - no matplotlib
# on the host). Built on demand by `roofline plot`.
# Base image and deps are content-pinned (AUDIT_arch_2026 H-6): a tag is a
# mutable ref, and a bare `==` still lets the index serve different bytes.
# Digest: docker buildx imagetools inspect python:3.14-slim
FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
