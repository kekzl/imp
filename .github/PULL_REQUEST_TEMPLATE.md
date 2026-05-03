<!--
Thanks for the PR. Keep it short — what changed, why, and how it was tested.
For perf-sensitive changes, paste before/after numbers (model, quant,
tg256 / pp512, hardware).
-->

## Summary

<!-- 1–3 bullets on what this changes and why. -->

## Test plan

- [ ] `make verify-fast` (or `make verify`) green
- [ ] For perf changes: tg256 / pp512 before/after on at least one model
- [ ] For new model architectures: smoke prompt + degeneration check
- [ ] For C-API changes: every `include/imp/` caller updated
