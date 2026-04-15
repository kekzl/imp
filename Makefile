# imp test runner targets
# Usage: make test-unit, make test-gpu, make test-all, make bench

DOCKER_IMG ?= imp:test
DOCKER_RUN = docker run --rm --gpus all -v $(PWD)/models:/models $(DOCKER_IMG)
BUILD_ARGS = --build-arg IMP_BUILD_TESTS=ON

.PHONY: build test-unit test-gpu test-fast test-all test-perf test-golden bench check-gpu

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
test-unit: build
	$(DOCKER_RUN) imp-tests --gtest_filter="TensorTest.*:GgufLoaderTest.*:Tokenizer*:ChatTemplate*:HFChatTemplate*:BatchBuilder*:Scheduler*:Request*:EndToEndTest.*:StubModelTest.LoadStubModel:StubModelTest.TokenizeStub"

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
