# syntax=docker/dockerfile:1

# =============================================================================
# Stage 1: Build imp from source
# =============================================================================
# Native CUDA 13.3 devel image: nvcc 13.3 (V13.3.33) is on PATH at
# /usr/local/cuda (-> /usr/local/cuda-13.3) out of the box — no apt toolkit
# install needed. The host driver (UMD 13.3) supports it. sm_120 gains the 13.3
# ptxas/PTX-ISA-9.3 codegen; no new tensor-core HW (still mma.sync, no
# tcgen05/wgmma).
FROM nvidia/cuda:13.3.0-devel-ubuntu24.04 AS builder

ARG CMAKE_BUILD_TYPE=Release

RUN { sed -i 's|archive.ubuntu.com|de.archive.ubuntu.com|g; s|security.ubuntu.com|de.archive.ubuntu.com|g' \
          /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true; } \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ git ninja-build ca-certificates python3 wget \
    && wget -qO /tmp/cmake.sh https://github.com/Kitware/CMake/releases/download/v4.3.1/cmake-4.3.1-linux-x86_64.sh \
    && sh /tmp/cmake.sh --skip-license --prefix=/usr/local \
    && rm /tmp/cmake.sh \
    && rm -rf /var/lib/apt/lists/*

# Hint CMake's find_package(CUDAToolkit) at the toolkit (nvcc is already on PATH).
ENV CUDA_HOME=/usr/local/cuda

# Pre-clone third-party deps into their own layer. Only invalidated when the
# pinned tags below change — code-only edits keep this layer cached, saving
# the FetchContent git-clone step (~30-60s) on every Docker rebuild.
# Tags must mirror the FetchContent_Declare entries in CMakeLists.txt.
RUN git clone --depth=1 --branch v1.17.0 https://github.com/google/googletest.git /deps/googletest \
 && git clone --depth=1 --branch v4.5.1  https://github.com/NVIDIA/cutlass.git    /deps/cutlass    \
 && git clone --depth=1 --branch v0.46.1 https://github.com/yhirose/cpp-httplib.git /deps/httplib  \
 && git clone --depth=1 --branch v3.12.0 https://github.com/nlohmann/json.git     /deps/json

WORKDIR /src
COPY . .

# Override -march=native with portable -march=x86-64-v3 for Docker portability
RUN sed -i 's/-march=native/-march=x86-64-v3/g' cmake/CompilerFlags.cmake

ARG IMP_BUILD_TESTS=OFF
ARG IMP_BUILD_BENCH=OFF
ARG IMP_EXTRA_CMAKE=

RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DIMP_BUILD_TESTS=${IMP_BUILD_TESTS} \
        -DIMP_BUILD_BENCH=${IMP_BUILD_BENCH} \
        -DIMP_BUILD_TOOLS=ON \
        -DIMP_BUILD_SERVER=ON \
        ${IMP_EXTRA_CMAKE} \
        -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/deps/googletest \
        -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/deps/cutlass \
        -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=/deps/httplib \
        -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/deps/json \
    && cmake --build build -j$(nproc) \
    && cp build/imp-server build/imp-cli /tmp/ \
    && if [ -f build/imp-tests ]; then \
           cp build/imp-tests build/imp-tests-unit /tmp/ \
           && for b in test-core test-text test-compute test-attention \
                       test-quant test-kv test-moe-gdn test-e2e; do \
                  [ -f "build/$b" ] && cp "build/$b" /tmp/; \
              done; \
       fi \
    && ([ -f build/imp-bench ] && cp build/imp-bench /tmp/ || true) \
    && ([ -f build/test-gdn ] && cp build/test-gdn /tmp/ || true)

# =============================================================================
# Stage 2: Minimal runtime image
# =============================================================================
# Native CUDA 13.3 runtime image already ships the matching cudart + cuBLAS
# (and transitive deps like libnvjitlink) at /usr/local/cuda; only the small
# entrypoint/healthcheck helpers need adding.
FROM nvidia/cuda:13.3.0-runtime-ubuntu24.04

# OCI image metadata — GHCR renders org.opencontainers.image.description on the
# package page (https://github.com/kekzl/imp/pkgs/container/imp). Hardcoded here
# (not only via docker/metadata-action) so local builds carry it too.
LABEL org.opencontainers.image.title="imp" \
      org.opencontainers.image.description="LLM inference engine in C++/CUDA for NVIDIA Blackwell sm_120 (RTX 5090/5080/5070 Ti, RTX PRO 6000). Native NVFP4 + GGUF, OpenAI/Anthropic-compatible server. Written entirely by Claude Code." \
      org.opencontainers.image.source="https://github.com/kekzl/imp" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Copy built binaries
COPY --from=builder /tmp/imp-server /usr/local/bin/imp-server
COPY --from=builder /tmp/imp-cli /usr/local/bin/imp-cli
COPY --from=builder /tmp/imp-test[s] /usr/local/bin/
COPY --from=builder /tmp/imp-tests-uni[t] /usr/local/bin/
COPY --from=builder /tmp/test-cor[e] /usr/local/bin/
COPY --from=builder /tmp/test-tex[t] /usr/local/bin/
COPY --from=builder /tmp/test-comput[e] /usr/local/bin/
COPY --from=builder /tmp/test-attentio[n] /usr/local/bin/
COPY --from=builder /tmp/test-quan[t] /usr/local/bin/
COPY --from=builder /tmp/test-k[v] /usr/local/bin/
COPY --from=builder /tmp/test-moe-gd[n] /usr/local/bin/
COPY --from=builder /tmp/test-e2[e] /usr/local/bin/
COPY --from=builder /tmp/imp-benc[h] /usr/local/bin/
COPY --from=builder /tmp/test-gd[n] /usr/local/bin/

# Copy entrypoint
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Non-root user with write access to /models
RUN useradd -m -s /bin/bash imp \
    && mkdir -p /models \
    && chown imp:imp /models

USER imp
WORKDIR /home/imp

EXPOSE 8080
VOLUME /models

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:${IMP_PORT:-8080}/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["imp-server"]
