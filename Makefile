# imp test runner targets
# Usage: make test-unit, make test-gpu, make test-all, make bench

# Fail loudly: a failure anywhere in a piped recipe (e.g. `cmd | tee`) must
# propagate, never get masked by the exit code of the last pipe stage.
SHELL := bash
.SHELLFLAGS := -o pipefail -c

DOCKER_IMG ?= imp:test
DOCKER_RUN = docker run --rm --gpus all -v $(PWD)/models:/models $(DOCKER_IMG)
BUILD_ARGS = --build-arg IMP_BUILD_TESTS=ON
# Dependency pins live once in cmake/imp-deps.cmake; inject them into the Docker
# build so the tags are not duplicated (bump that file only). Extraction is in a
# script — inlining the sed breaks make's $(shell ...) paren matching.
DEP_ARGS = $(shell scripts/dep_build_args.sh)

.PHONY: roofline-measure roofline-pin roofline-regress build test-unit test-gpu test-fast test-all test-e2e test-server test-vision test-perf test-golden test-agents test-agents-external test-niah test-rerank bench bench-agentic check-gpu verify verify-fast verify-chunked verify-north-star gen-perf-baseline install-hooks format format-check tidy sanitize asan coverage

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
	docker build $(BUILD_ARGS) $(DEP_ARGS) -t $(DOCKER_IMG) .

# Unit tests: CPU-only, no GPU, no model, < 5s
# Mirrors `ctest -L unit`. Filter is sourced from CMakeLists.txt (_unit_e2e_filter).
test-unit: build
	$(DOCKER_RUN) imp-tests-unit

# GPU tests: everything including CUDA kernels. ~4-5 min without models —
# 7 of 8 binaries finish in <11s, but test-attention alone is ~241s (the
# paged-/crosspath-oracle sweeps, TEST_AUDIT.md §8). The old "<30s" note was
# stale.
test-gpu: build
	$(DOCKER_RUN) imp-tests

# Stage 3 — the SERVER stage (local, GPU-only). Boots a real imp-server against
# a live model and GATES on the OpenAI+Anthropic wire batteries (endpoints,
# robustness #712, logprobs, /v1/messages stream, embed/chat interleave,
# 0-token #710). CI has no GPU runner, so this is the only place handlers.cpp /
# batching_engine run end-to-end. See the script header for env knobs.
test-server: build
	bash scripts/test_server.sh

# Measured gcov line coverage of tools/imp-server/ over an end-to-end GPU run
# (builds an instrumented imp-server, drives every endpoint + the manual server
# batteries, reports coverage). Needs a GPU + a local model. See the script header.
coverage:
	bash scripts/coverage_server.sh

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

# Vision GPU golden (R9 / #583): SigLIP + gemma4v encoder + projector tail.
# Mounts $(HOME)/models (symlink targets resolve) + the committed fixture.
# Set IMP_VISION_GOLDEN_DUMP=1 to regenerate goldens instead of asserting.
test-vision: build
	docker run --rm --gpus all -v $(HOME)/models:/models -v $(PWD)/tests/fixtures:/fixtures \
		-e IMP_TEST_MMPROJ=/models/gemma-3-4b-vl/mmproj-F16.gguf \
		-e IMP_TEST_MMPROJ_GEMMA4=/models/gemma-3-4b-vl/mmproj-gemma4-26b-bf16.gguf \
		-e IMP_VISION_TEST_IMAGE=/fixtures/vision_test_64.png \
		-e IMP_VISION_GOLDEN_DUMP=$(IMP_VISION_GOLDEN_DUMP) \
		$(DOCKER_IMG) imp-tests --gtest_filter="VisionGolden.*"

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

# Agentic benchmarks: boot a real imp-server and drive the two agent-shaped
# harnesses — concurrency TTFT/ITL (agent_bench.py) and growing-transcript
# per-turn TTFT with prefix cache on/off (agent_replay_bench.py). The metrics a
# coding agent actually feels (see docs/GOAL.md "Agentic surface"). Override the
# model with MODEL=<name-under-$HOME/models>.
AGENTIC_MODEL ?= Qwen3-8B-Q8_0.gguf
bench-agentic: build check-gpu
	@echo "=== imp agentic bench (RTX 5090) — model=$(AGENTIC_MODEL) ==="
	@docker rm -f imp-agentic-bench >/dev/null 2>&1 || true
	@docker run -d --name imp-agentic-bench --gpus all -p 8080:8080 \
		-v $(HOME)/models:/models $(DOCKER_IMG) \
		imp-server --host 0.0.0.0 --model /models/$(AGENTIC_MODEL) >/dev/null
	@echo "waiting for server..."; \
	for i in $$(seq 1 90); do curl -sf http://localhost:8080/health >/dev/null 2>&1 && break; sleep 2; done; \
	trap 'docker rm -f imp-agentic-bench >/dev/null 2>&1' EXIT; \
	echo "--- concurrency TTFT/ITL ---"; \
	python3 tools/agent_bench.py --url http://localhost:8080 --model $(AGENTIC_MODEL) --concurrency 1,4,16; \
	echo "--- multi-turn replay ---"; \
	python3 tools/agent_replay_bench.py --url http://localhost:8080 --model $(AGENTIC_MODEL) --turns 16

# Agent-harness E2E battery (#1007): boots a real imp-server and drives the
# wire patterns real agent harnesses generate — multi-turn tool loops in the
# Anthropic (/v1/messages), OpenAI chat and /v1/responses dialects, with
# cache_control reuse, FSM-enforced tool_choice and streaming delta assembly.
# Gates on every check (exit 1 on failure). GPU + local model, like test-server.
test-agents: build check-gpu
	@echo "=== imp agent-loop battery — model=$(AGENTIC_MODEL) ==="
	@docker rm -f imp-agent-suite >/dev/null 2>&1 || true
	@docker run -d --name imp-agent-suite --gpus all -p 8080:8080 \
		-v $(HOME)/models:/models $(DOCKER_IMG) \
		imp-server --host 0.0.0.0 --model /models/$(AGENTIC_MODEL) >/dev/null
	@echo "waiting for server..."; \
	for i in $$(seq 1 90); do curl -sf http://localhost:8080/health >/dev/null 2>&1 && break; sleep 2; done; \
	trap 'docker rm -f imp-agent-suite >/dev/null 2>&1' EXIT; \
	echo "--- stage 1: wire-conformance (forced flows, 3 dialects) ---"; \
	python3 tools/analysis/agent_loop_suite.py --url http://localhost:8080; \
	echo "--- stage 2: real model-driven loop (auto tool_choice, real tools) ---"; \
	python3 tools/analysis/agent_task_loop.py --url http://localhost:8080 --model $(AGENTIC_MODEL)

# Needle-in-a-haystack retrieval gate past 16K (#1022): boots a model that fits
# a long context and asserts a planted needle is retrieved at 16K and 32K across
# depths. A CORRECTNESS gate (retrieval success), independent of timing — safe on
# any host. The TTFT timing gates at 32K-64K need a verified-healthy host to pin
# their numbers (benchmarking contract) and are run separately.
# Rerank gate (#roadmap gap 9): boots a cross-encoder reranker and asserts the
# /v1/rerank contract, semantics (the relevant document wins) and stability.
# Pass COMPARE_URL=http://host:port to also diff against a reference reranking
# server (llama.cpp --reranking) running the SAME model file.
RERANK_MODEL ?= qwen3-reranker-0.6b-q8_0.gguf
test-rerank: build check-gpu
	@docker rm -f imp-rerank >/dev/null 2>&1 || true
	@docker run -d --name imp-rerank --gpus all -p 8080:8080 \
		-v $(HOME)/models:/models $(DOCKER_IMG) \
		imp-server --host 0.0.0.0 --model /models/$(RERANK_MODEL) >/dev/null
	@echo "waiting for server..."; \
	for i in $$(seq 1 90); do curl -sf http://localhost:8080/health >/dev/null 2>&1 && break; sleep 2; done; \
	trap 'docker rm -f imp-rerank >/dev/null 2>&1' EXIT; \
	python3 tools/analysis/rerank_check.py --url http://localhost:8080 \
		$(if $(COMPARE_URL),--compare $(COMPARE_URL),)

NIAH_MODEL ?= Qwen3-14B-Q6_K.gguf
test-niah: build check-gpu
	@docker rm -f imp-niah >/dev/null 2>&1 || true
	@docker run -d --name imp-niah --gpus all -p 8080:8080 \
		-v $(HOME)/models:/models $(DOCKER_IMG) \
		imp-server --host 0.0.0.0 --model /models/$(NIAH_MODEL) --set runtime.max_seq_len=40000 >/dev/null
	@echo "waiting for server..."; \
	for i in $$(seq 1 90); do curl -sf http://localhost:8080/health >/dev/null 2>&1 && break; sleep 2; done; \
	trap 'docker rm -f imp-niah >/dev/null 2>&1' EXIT; \
	python3 tools/analysis/niah_check.py --url http://localhost:8080 --model $(NIAH_MODEL) \
		--lengths 16000,32000 --depths 0.1,0.5,0.9

# #1007 stage-2 EXTERNAL gate (opt-in): REAL third-party agent binaries driving
# imp-server through a genuine edit loop — proves the whole loop survives an
# ACTUAL agent, not just our own driver. Two legs: aider over the OpenAI dialect
# and Claude Code over the Anthropic one (ANTHROPIC_BASE_URL), the latter being
# the demanding client — ~20K system prompt, 25 tools, cache_control, streaming.
# Heavier than `test-agents` (builds harness images, uses --network host), so it
# is a separate opt-in target rather than part of the default agent battery.
# GPU + local model.  Third arg selects a leg: all (default) | aider | claude-code
test-agents-external: build check-gpu
	bash tools/analysis/agent_external_smoke.sh $(AGENTIC_MODEL) 8080

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

# verify-north-star: gates the docs/GOAL.md north-star model (Qwen3-14B Q6_K) against
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

# Roofline pipeline (tools/roofline/): GPU measurement is local-only (CI has no
# GPU runner). `roofline-pin` runs the full sweep, pins the run as regression
# baseline (history/BASELINE) and leaves history ready to commit.
roofline-measure: check-gpu
	@tools/roofline/roofline measure

roofline-pin: check-gpu
	@RUN_ID=$$(tools/roofline/roofline measure | tail -1) && \
		echo "$$RUN_ID" > tools/roofline/history/BASELINE && \
		echo "pinned roofline baseline: $$RUN_ID (commit tools/roofline/history/)"

roofline-regress:
	@if [ -f tools/roofline/history/BASELINE ]; then \
		tools/roofline/roofline regress --baseline "$$(cat tools/roofline/history/BASELINE)" --run latest --threshold 5; \
	else echo "no pinned baseline — run 'make roofline-pin' first"; fi

# Install the local git hooks. Two-stage test gate:
#   Stage 1 — pre-commit (GPU): runs the full GPU suite (make test-gpu) when
#             staged sources change. CI has no GPU runner, so GPU correctness is
#             gated here, locally, before the commit lands.
#   pre-push: keeps the verify-fast perf+smoke regression gate.
#   Stage 2 — CI (CPU): ctest -L unit, in .github/workflows/ci.yml (no hook).
install-hooks:
	@cp scripts/pre-commit.hook .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@cp scripts/pre-push.hook .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "hooks installed:"
	@echo "  pre-commit → Stage 1 'make test-gpu' (full GPU suite) on staged src/tests changes"
	@echo "  pre-push   → 'make verify-fast' (perf + smoke regression) on source changes"
	@echo "  CI (Stage 2) runs 'ctest -L unit' — the CPU lane — automatically"

# clang-format settings live in .clang-format. Host has no clang-format
# installed (clean-host policy), so we run it in a throwaway container.
CLANG_FORMAT_IMG ?= silkeh/clang:18
CLANG_FORMAT_RUN = docker run --rm -v $(PWD):/work -w /work $(CLANG_FORMAT_IMG) clang-format
CLANG_FORMAT_FILES = $$(find src include tools tests -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh')

# compute-sanitizer (memcheck) over the GPU-numeric test binaries
# (TEST_AUDIT.md §4.4). Runs inside the BUILDER stage (the runtime image has
# no CUDA toolkit, hence no compute-sanitizer; the builder keeps build/).
#
# DOES NOT WORK ON WSL2: the WDDM driver model exposes no debugger interface,
# compute-sanitizer reports "Error: Failed to initialize" (verified
# 2026-06-04 on the WSL2 dev host). Run this target on a native-Linux GPU
# host (e.g. a future self-hosted CI runner). Listed here so the invocation
# is documented and ready, not because it runs on the dev box.
# Host-code ASan+UBSan over the CPU test binaries (test-core, test-text).
# Works on WSL2 (host-side sanitizers only — nvcc-compiled device code is NOT
# sanitized, see IMP_SANITIZERS in CMakeLists.txt). Suppressions live in
# tools/sanitizers/: vendored-stb unaligned stores (UBSan, #1047) and NVIDIA
# driver one-time allocations (LSan). Build tree persists in a named docker
# volume so re-runs are incremental.
asan:
	docker build --target builder $(BUILD_ARGS) $(DEP_ARGS) -t imp:builder .
	docker run --rm --gpus all -v $(PWD):/src -v imp-asan-build:/basan -w /src imp:builder bash -c '\
	  cmake -B /basan -S /src -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIMP_SANITIZERS=ON \
	        -DIMP_BUILD_TOOLS=OFF -DIMP_BUILD_BENCH=OFF -DIMP_BUILD_SERVER=OFF > /basan/configure.log && \
	  cmake --build /basan --target test-core test-text -j$$(nproc) && \
	  for b in test-core test-text; do \
	    echo "== ASan+UBSan: $$b =="; \
	    UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1:suppressions=/src/tools/sanitizers/ubsan.supp \
	    ASAN_OPTIONS=detect_leaks=1 \
	    LSAN_OPTIONS=suppressions=/src/tools/sanitizers/lsan.supp \
	    /basan/$$b || exit 1; \
	  done'

sanitize:
	docker build --target builder $(BUILD_ARGS) -t imp:sanitize .
	@for b in test-attention test-quant test-kv; do \
		echo "== compute-sanitizer memcheck: $$b =="; \
		docker run --rm --gpus all -v $(PWD)/models:/models imp:sanitize \
			/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
			/src/build/$$b || exit 1; \
	done

# Apply clang-format in place across src/, include/, tools/, tests/.
format:
	@$(CLANG_FORMAT_RUN) -i --style=file $(CLANG_FORMAT_FILES)
	@echo "clang-format applied"

# Check formatting without modifying files. Exits non-zero on violation.
format-check:
	@$(CLANG_FORMAT_RUN) --dry-run -Werror --style=file $(CLANG_FORMAT_FILES)

# clang-tidy over host C++ TUs (advisory — findings surface, do not fail). Runs in
# the CUDA builder image so the CUDA headers our .cpp files include are present;
# clang-tidy is apt-installed on the fly. .cu files are out of scope (need full
# nvcc flags). Configures first so build/compile_commands.json exists.
CLANG_TIDY_FILES = $$(find src tools -name '*.cpp')
tidy:
	@docker run --rm -v $(PWD):/work -w /work imp:builder bash -c '\
	  apt-get update -qq && apt-get install -y -qq clang-tidy >/dev/null 2>&1; \
	  test -f build/compile_commands.json || cmake --preset ci \
	      -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/deps/googletest \
	      -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/deps/cutlass \
	      -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=/deps/httplib \
	      -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/deps/json >/dev/null; \
	  clang-tidy -p build --warnings-as-errors= $(CLANG_TIDY_FILES) || true'
