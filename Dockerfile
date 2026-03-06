# syntax=docker/dockerfile:1

# =============================================================================
# Stage 1: Build imp from source
# =============================================================================
FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS builder

ARG CUDA_ARCHITECTURES="90a;100;120a"
ARG CMAKE_BUILD_TYPE=Release

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake g++ git ninja-build ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Override -march=native with portable -march=x86-64-v3 for Docker portability
RUN sed -i 's/-march=native/-march=x86-64-v3/g' cmake/CompilerFlags.cmake

# BuildKit cache mount persists the build directory across Docker builds.
# On code-only changes, Ninja recompiles only modified translation units
# instead of rebuilding from scratch (~30s vs ~5min for full CUDA rebuild).
RUN --mount=type=cache,target=/src/build \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DIMP_BUILD_TESTS=OFF \
        -DIMP_BUILD_BENCH=OFF \
        -DIMP_BUILD_TOOLS=ON \
        -DIMP_BUILD_SERVER=ON \
    && cmake --build build -j$(nproc) \
    && cp build/imp-server build/imp-cli /tmp/

# =============================================================================
# Stage 2: Minimal runtime image
# =============================================================================
FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
        libcublas-13-1 \
        curl \
        python3-pip \
    && pip3 install --no-cache-dir --break-system-packages huggingface-hub \
    && rm -rf /var/lib/apt/lists/*

# Copy built binaries
COPY --from=builder /tmp/imp-server /usr/local/bin/imp-server
COPY --from=builder /tmp/imp-cli /usr/local/bin/imp-cli

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
