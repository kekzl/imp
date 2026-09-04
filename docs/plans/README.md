# Plan records

Dated records, lint-exempt, no metadata header. One file per campaign,
`YYYY-MM-DD-<topic>.md`. Items are closed, never deleted: closure is
`~~strikethrough~~` plus `DONE <date> (#PR)` / `MEASURED` / `ANSWERED` /
`REFUTED` / `CLOSED <date>`, and a finished plan ends in a terminal
`## ROADMAP CLOSED (<date>)` block (#1786).

The verdict a plan produced lives in [`../roadmap.md`](../roadmap.md); the plan
holds the measurement narrative behind it. A record whose title says "detail"
is the other direction: text moved OUT of the roadmap.

| plan | state | standing record |
|---|---|---|
| [2026-05-28-q4k-mmq-kernel-design](2026-05-28-q4k-mmq-kernel-design.md) | CLOSED 2026-05-28, REFUTED before implementation | roadmap "Closed competitive records" |
| [2026-07-22-token-recycling-spec-tree](2026-07-22-token-recycling-spec-tree.md) | CLOSED 2026-08-19, neutral to -7%, default off | roadmap Closed, "speculation tree" |
| [2026-07-31-qwen3-vl-vision](2026-07-31-qwen3-vl-vision.md) | CLOSED 2026-08-11, shipped #1163-#1180 | roadmap Closed, "vision beyond Gemma" |
| [2026-08-15-imp-quantize-roadmap](2026-08-15-imp-quantize-roadmap.md) | **OPEN**: items 2 (embedding opt-in), 3 (blocked on a model), 4 (stacked experts) | roadmap Open 6, Open 7 |
| [2026-08-24-qwen38-port](2026-08-24-qwen38-port.md) | CLOSED 2026-08-27, all 8 items terminal | roadmap Closed, "Qwen3.8 port roadmap" |
| [2026-08-27-prefill-decode-overlap](2026-08-27-prefill-decode-overlap.md) | CLOSED 2026-08-27, NEUTRAL both shapes, default off | roadmap lever ledger |
| [2026-08-28-sparse-decode-attention](2026-08-28-sparse-decode-attention.md) | CLOSED 2026-08-30, SHIPPED opt-in (#1808, #1818, #1819) | roadmap Open 3 (the remainder) |
| [2026-08-29-qwen38-long-context-posture](2026-08-29-qwen38-long-context-posture.md) | CLOSED 2026-08-31, both open items answered | roadmap lever ledger |
| [2026-08-31-fp8-ssm-prefill](2026-08-31-fp8-ssm-prefill.md) | CLOSED 2026-08-31, REFUTED e2e, closed unmerged | roadmap prefill kernels |
| [2026-08-31-mtp-multicandidate-hybrid](2026-08-31-mtp-multicandidate-hybrid.md) | CLOSED 2026-08-31, built and measured, gate not met | roadmap Closed, "speculation tree" |
| [2026-08-31-roadmap-ledger-detail](2026-08-31-roadmap-ledger-detail.md) | record, moved out of the roadmap 2026-08-31 | - |
| [2026-09-04-lever-ledger-detail](2026-09-04-lever-ledger-detail.md) | record, moved out of the roadmap 2026-09-04 | - |
