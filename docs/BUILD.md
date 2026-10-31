<!--
layer: L1
audience: operators
verified: 2026-10-31
commit: ed4422b7
-->

# Build from source

The published image `ghcr.io/kekzl/imp:latest` is the supported way to run imp ([README Quickstart](../README.md#quickstart-docker)). Build it yourself to run a tree that is not released yet or your own changes.

## Docker image (recommended)

Needs Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), no CUDA toolkit on the host: the compile runs in a CUDA 13.4.3 container.

```bash
git clone https://github.com/kekzl/imp && cd imp
make build                     # -> image imp:test (minutes)
docker run --gpus all -v ~/models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8080:8080 imp:test --model /models/Qwen3.8-27B-NVFP4-vllm
```

| Want | Command |
|---|---|
| image tagged `imp:latest` (what `docker-compose.yml` runs) | `docker compose build imp-server` |
| convert a model with your own build | `IMP_IMAGE=imp:test scripts/stage-model.sh <org>/<repo> <dir>` ([`QUICKSTART.md`](QUICKSTART.md#converting-a-source-that-is-not-nvfp4-yet)) |
| fast incremental compile while changing code | `make dev` (seconds), workflow in [`CONTRIBUTING.md`](../CONTRIBUTING.md) |

## Release tarball, no Docker at runtime

`make bundle` writes `dist/imp-<version>-linux-x86_64.tar.gz`: `imp-server`, `imp-cli`, `imp-quantize`, `imp-bench` with the CUDA libraries bundled; the host needs only the NVIDIA driver. Running and verifying it: [`DEPLOYMENT.md`](DEPLOYMENT.md#without-docker).

## Host build

For a machine with the CUDA toolkit installed natively.

| Needs | Version |
|---|---|
| CUDA toolkit | 13.2 or newer (CMake refuses older); tested: 13.4.3 |
| CMake | 3.25 or newer |
| host compiler | C++23: GCC 13+ or Clang 16+ |
| CUDA libraries linked | `cudart`, `cuda_driver`, `cublasLt` |

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
build/imp-server --model ~/models/Qwen3.8-27B-NVFP4-vllm --port 8080
```

Build options (tests, tools, sanitizers): [`CONTRIBUTING.md`](../CONTRIBUTING.md#build).
