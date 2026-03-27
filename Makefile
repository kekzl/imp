# imp test runner targets
# Usage: make test-unit, make test-gpu, make test-all

DOCKER_IMG ?= imp:test
DOCKER_RUN = docker run --rm --gpus all -v $(PWD)/models:/models $(DOCKER_IMG)
BUILD_ARGS = --build-arg IMP_BUILD_TESTS=ON

.PHONY: build test-unit test-gpu test-fast test-all test-perf test-golden

build:
	docker build $(BUILD_ARGS) -t $(DOCKER_IMG) .

# Unit tests: CPU-only, no GPU, no model, < 5s
test-unit: build
	$(DOCKER_RUN) imp-tests --gtest_filter="TensorTest.*:GgufLoaderTest.*:Tokenizer*:ChatTemplate*:BatchBuilder*:Scheduler*:Request*:EndToEndTest.*:StubModelTest.LoadStubModel:StubModelTest.TokenizeStub"

# GPU tests: everything including CUDA kernels, < 30s
test-gpu: build
	$(DOCKER_RUN) imp-tests

# Fast: unit tests only (no Docker GPU needed if built already)
test-fast: test-unit

# All tests including standalone GDN kernel test
test-all: build
	$(DOCKER_RUN) imp-tests
	$(DOCKER_RUN) test-gdn

# Performance regression (needs baseline in tests/perf_baseline.json)
test-perf: build
	$(DOCKER_RUN) imp-cli --model /models/Qwen3-8B-Q8_0.gguf --bench --bench-pp 512 --bench-reps 5 --max-tokens 128 --temperature 0

# Golden output comparison (greedy, temp=0)
test-golden: build
	@echo "Golden output tests require running server — use pytest tests/api/ instead"
