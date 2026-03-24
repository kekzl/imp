#!/usr/bin/env bash
# Run API tests against the mock server (no GPU, no model required).
# Usage: ./run_mock_tests.sh
#
# Environment:
#   IMP_MOCK_PORT  - mock server port (default: 9099)
#
# Exit code: 0 if all tests pass, 1 otherwise.

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
