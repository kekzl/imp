#!/usr/bin/env bash
# Usage: ./run_mock_tests.sh. Env: IMP_MOCK_PORT (default 9099).
# Exit 0 if all tests pass, 1 otherwise.

set -euo pipefail
cd "$(dirname "$0")"

export IMP_USE_MOCK=1
export IMP_MOCK_PORT="${IMP_MOCK_PORT:-9099}"

echo "=== Running API tests against mock server (port ${IMP_MOCK_PORT}) ==="

# Run all tests except those that require a real model
python -m pytest \
    -v \
    -m "not perf and not tools" \
    --tb=short \
    "$@"

echo "=== Mock tests complete ==="
