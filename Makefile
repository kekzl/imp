# imp test runner targets
# Usage: make test-unit, make test-gpu, make test-all, make bench

# Fail loudly: a failure anywhere in a piped recipe (e.g. `cmd | tee`) must
# propagate, never get masked by the exit code of the last pipe stage.
SHELL := bash
.SHELLFLAGS := -o pipefail -c

DOCKER_IMG ?= imp:test
DOCKER_RUN = docker run --rm --gpus all -v $(PWD)/models:/models $(DOCKER_IMG)
BUILD_ARGS = --build-arg IMP_BUILD_TESTS=ON

.PHONY: build test-unit test-gpu test-fast test-all test-perf test-golden bench check-gpu verify verify-fast verify-chunked gen-perf-baseline install-hooks format format-check

# Check that no other process is using the GPU (games, other inference, etc.)
check-gpu:
	@GPU_PROCS=$$(nvidia-smi --query-compute-apps=pid,name,used_gpu_memory --format=csv,noheader 2>/dev/null | grep -v "^$$"); \
	if [ -n "$$GPU_PROCS" ]; then \
		echo "ERROR: GPU is in use — benchmarks will be unreliable:"; \
		echo "$$GPU_PROCS"; \
		echo "Close other GPU processes first (games, other inference, etc.)"; \
		exit 1; \
	fi; \
	GPU_UTIL=$$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1); \
	if [ "$$GPU_UTIL" -gt 5 ] 2>/dev/null; then \
		echo "WARNING: GPU utilization at $${GPU_UTIL}% — results may be noisy"; \
	fi; \
	echo "GPU is free (utilization: $${GPU_UTIL:-0}%)"

build:
	docker build $(BUILD_ARGS) -t $(DOCKER_IMG) .

# Unit tests: CPU-only, no GPU, no model, < 5s
# Mirrors `ctest -L unit`. Filter is sourced from CMakeLists.txt (_unit_e2e_filter).
test-unit: build
	$(DOCKER_RUN) imp-tests-unit

# GPU tests: everything including CUDA kernels, < 30s
test-gpu: build
	$(DOCKER_RUN) imp-tests

# Fast: unit tests only (no Docker GPU needed if built already)
test-fast: test-unit

# All tests including standalone GDN kernel test
test-all: build
	$(DOCKER_RUN) imp-tests
	$(DOCKER_RUN) test-gdn

# E2E model tests: load real models, generate, verify output
# Uses Qwen3-4B (dense) + Qwen3.5-4B (GDN hybrid) + Gemma-4-26B-A4B (MoE) from ./models/
test-e2e: build
	docker run --rm --gpus all -v $(PWD)/models:/models \
		-e IMP_TEST_MODEL=/models/Qwen3-4B-Instruct-2507-Q8_0.gguf \
		-e IMP_TEST_MODEL_GDN=/models/Qwen3.5-4B-Q8_0.gguf \
		-e IMP_TEST_MODEL_GEMMA4=/models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
		$(DOCKER_IMG) imp-tests --gtest_filter="PrimaryModelTest.*:GDNModelTest.*:EndToEndModelTest.*:Gemma4ModelTest.*:Gemma4GraphsTest.*"

# Full benchmark suite: all baseline models (requires GPU to be free)
bench: build check-gpu
	@echo "=== imp benchmark suite (RTX 5090) ==="
	@echo ""
	@echo "--- Qwen3-4B Q8_0 ---"
	$(DOCKER_RUN) imp-cli --model /models/Qwen3-4B-Instruct-2507-Q8_0.gguf --bench --bench-pp 512 --bench-reps 5 --max-tokens 256 --temperature 0
	@echo ""
	@echo "--- Qwen3-8B Q8_0 ---"
	$(DOCKER_RUN) imp-cli --model /models/Qwen3-8B-Q8_0.gguf --bench --bench-pp 512 --bench-reps 5 --max-tokens 256 --temperature 0
	@echo ""
	@echo "--- Qwen3.5-4B GDN Q8_0 ---"
	$(DOCKER_RUN) imp-cli --model /models/Qwen3.5-4B-Q8_0.gguf --bench --bench-pp 512 --bench-reps 5 --max-tokens 256 --temperature 0
	@echo ""
	@echo "--- Qwen3.5-9B GDN Q8_0 ---"
	$(DOCKER_RUN) imp-cli --model /models/Qwen3.5-9B-Q8_0.gguf --bench --bench-pp 512 --bench-reps 5 --max-tokens 256 --temperature 0
	@echo ""
	@echo "--- Qwen3-4B MXFP4 ---"
	$(DOCKER_RUN) imp-cli --model /models/qwen3-4b-instruct-2507-mxfp4.gguf --bench --bench-pp 512 --bench-reps 5 --max-tokens 256 --temperature 0

# Single model benchmark (quick check)
test-perf: build check-gpu
	$(DOCKER_RUN) imp-cli --model /models/Qwen3-8B-Q8_0.gguf --bench --bench-pp 512 --bench-reps 5 --max-tokens 256 --temperature 0

# Golden output comparison (greedy, temp=0)
test-golden: build
	@echo "Golden output tests require running server — use pytest tests/api/ instead"

# verify: full pre-merge gate (host build, ~5 min). build + ctest + perf + smoke.
verify:
	@scripts/verify.sh full

# verify-fast: pre-push gate (host build, ~90s). build + filtered tests + 1 smoke.
# Perf gate uses --prefill-chunk-size 0 to stay apples-to-apples with tests/perf_baseline.json.
verify-fast:
	@scripts/verify.sh fast

# verify-chunked: gates chunked-prefill path (chunk=512) against tests/perf_baseline_chunked.json.
# Looser thresholds (5%/8%) cover the gather + rect-attn per-chunk overhead.
verify-chunked:
	@IMP_VERIFY_BASELINE=tests/perf_baseline_chunked.json \
	 IMP_VERIFY_CHUNK_SIZE=512 \
	 scripts/verify.sh fast

# verify-north-star: gates the GOAL.md north-star model (Qwen3-14B Q6_K) against
# tests/perf_baseline_north_star.json. Same 3%/5% thresholds as perf_baseline.json.
# Requires Qwen3-14B-Q6_K.gguf in $(HOME)/models. Numbers were captured
# 2026-05-23 under the cold-median methodology (PR #376) — see
# memory/qwen3_14b_north_star_cold_median_2026_05_23.md for the raw samples
# (σ = 0.16 tok/s on tg128 @ ctx=2048, well inside the 3% threshold).
verify-north-star:
	@IMP_VERIFY_BASELINE=tests/perf_baseline_north_star.json \
	 scripts/verify.sh fast

# Regenerate tests/perf_baseline.json with the cold-median methodology (5 trials,
# 15s cooldown between, median of each metric). Resists cuBLAS-algo-state drift —
# see memory/bench_sustained_load_cublas_algo_drift_2026_05_23.md.
# Defaults to Qwen3-8B Q8_0; pass MODEL=… to override.
#
# Mounts the resolved model directory (defaults to ~/models so symlinks work)
# AND the repo root (so the script can write back to tests/perf_baseline.json).
# `-u` matches the host UID so the write succeeds.
MODEL ?= /models/Qwen3-8B-Q8_0.gguf
MODELS_DIR ?= $(HOME)/models
gen-perf-baseline: build
	@docker run --rm --gpus all \
		-v $(MODELS_DIR):/models \
		-v $(PWD):/src -w /src \
		-u $(shell id -u):$(shell id -g) \
		-e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
		--entrypoint bash $(DOCKER_IMG) scripts/gen_perf_baseline.sh "$(MODEL)"

# install the pre-push hook that runs verify-fast when source files change
install-hooks:
	@cp scripts/pre-push.hook .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "pre-push hook installed → runs 'make verify-fast' when src/, include/, or tools/ changes"

# clang-format settings live in .clang-format. Host has no clang-format
# installed (clean-host policy), so we run it in a throwaway container.
CLANG_FORMAT_IMG ?= silkeh/clang:18
CLANG_FORMAT_RUN = docker run --rm -v $(PWD):/work -w /work $(CLANG_FORMAT_IMG) clang-format
CLANG_FORMAT_FILES = $$(find src include tools tests -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh')

# Apply clang-format in place across src/, include/, tools/, tests/.
format:
	@$(CLANG_FORMAT_RUN) -i --style=file $(CLANG_FORMAT_FILES)
	@echo "clang-format applied"

# Check formatting without modifying files. Exits non-zero on violation.
format-check:
	@$(CLANG_FORMAT_RUN) --dry-run -Werror --style=file $(CLANG_FORMAT_FILES)
