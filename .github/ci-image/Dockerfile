# CI toolchain image for .github/workflows/ci.yml (Build, clang-tidy, PTX fallback, Sanitizers).
# nvidia/cuda:13.5.0-devel is 5007 MB compressed and cost a median 72 s "Initialize containers"
# per job (98 PR Builds, 2026-11-23); this is the base image plus the CUDA packages imp links.
# Published by .github/workflows/ci-image.yml; ci.yml pins the digest.
FROM nvidia/cuda:13.5.0-base-ubuntu26.04@sha256:2b8fc7b402eb045b809c3bb6296853e7d6e99a5acaded62b28f9494b357fbb4f

ARG CUDA_PKG=13-5
ENV DEBIAN_FRONTEND=noninteractive

# cuda-minimal-build: nvcc, ptxas, cudart headers, driver stub. libcublas-dev: cublasLt.
# cuda-nvtx: CUDA::nvtx. cuda-cuobjdump, cuda-cuxxfilt: the "Kernel resources" gate and the PTX lane.
# The static cuBLAS archives (960 MB) go: imp links the shared libs. libcudart_static.a stays, nvcc links it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-minimal-build-${CUDA_PKG} cuda-cuxxfilt-${CUDA_PKG} \
        libcublas-dev-${CUDA_PKG} cuda-nvtx-${CUDA_PKG} cuda-cuobjdump-${CUDA_PKG} \
        bash ca-certificates git python3 cmake ninja-build g++ ccache \
        libcurl4-openssl-dev libjpeg-turbo8-dev \
    && rm -f /usr/local/cuda/targets/x86_64-linux/lib/libcublas*_static.a \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

# clang-tidy for `Build`'s lint step: the per-run apt.llvm.org install was 12 s median, 41 s p90 on push.
# Pin and key fingerprint: scripts/install_llvm.sh (named context `scripts`, ci-image.yml).
COPY --from=scripts install_llvm.sh /tmp/install_llvm.sh
RUN bash /tmp/install_llvm.sh clang-tidy \
    && rm -rf /var/lib/apt/lists/* /tmp/install_llvm.sh

# FetchContent sources at the commits of cmake/imp-deps.cmake (named context `pins`), /deps/PINS =
# "<dir> <sha>" lines. scripts/ci_dep_dirs.sh points CMake here only where a pin matches, so a
# stale image fetches instead of building an old dependency.
COPY --from=pins imp-deps.cmake /tmp/imp-deps.cmake
RUN set -eu; mkdir -p /deps; \
    pin() { sha="$(sed -nE "s/^set\(IMP_DEP_$1_SHA[[:space:]]+([0-9a-f]{40}).*/\1/p" /tmp/imp-deps.cmake)"; \
            test -n "$sha"; mkdir -p "/deps/$2"; git -C "/deps/$2" init -q; \
            git -C "/deps/$2" fetch -q --depth=1 "$3" "$sha"; git -C "/deps/$2" checkout -q FETCH_HEAD; \
            echo "$2 $sha" >> /deps/PINS; }; \
    pin GOOGLETEST    googletest https://github.com/google/googletest.git; \
    pin HTTPLIB       httplib    https://github.com/yhirose/cpp-httplib.git; \
    pin NLOHMANN_JSON json       https://github.com/nlohmann/json.git; \
    pin PCRE2         pcre2      https://github.com/PCRE2Project/pcre2.git; \
    rm /tmp/imp-deps.cmake; cat /deps/PINS
