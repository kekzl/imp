# Plot container for the roofline pipeline (host stays clean - no matplotlib
# on the host). Built on demand by `roofline plot`.
# Base image and deps are content-pinned (AUDIT_ARCH_2026 H-6): a tag is a
# mutable ref, and a bare `==` still lets the index serve different bytes.
# Digest: docker buildx imagetools inspect python:3.14-slim
FROM python:3.15-slim@sha256:94d3426f1def7db4a106a232ac121bfd975393e5dfd40a619b955fae45fe7913
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
