---
name: Bug report
about: Something is broken or produces wrong output
labels: bug
---

## What happened

<!-- One paragraph on what you did and what went wrong. -->

## Reproducer

```bash
# Exact command that fails
```

## Environment

- imp version (`./build/imp-cli --version` or commit SHA):
- GPU + driver (`nvidia-smi | head -3`):
- CUDA toolkit (`nvcc --version`):
- Build (Docker / host / which `CMAKE_BUILD_TYPE`):
- Model (file path or HF repo + quantization):

## Output

<!-- Paste the failing output, error messages, or degenerate text. -->

## Notes

<!-- Anything else: when it started, things you tried, related issues. -->
