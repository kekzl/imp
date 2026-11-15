# Plan records

Dated records, lint-exempt, no metadata header. One file per campaign,
`YYYY-MM-DD-<topic>.md`. Items are closed, never deleted: closure is
`~~strikethrough~~` plus `DONE <date> (#PR)` / `MEASURED` / `ANSWERED` /
`REFUTED` / `CLOSED <date>`, and a finished plan ends in a terminal
`## ROADMAP CLOSED (<date>)` block (#1786).

The verdict a plan produced lives in [`../ROADMAP.md`](../ROADMAP.md); the plan
holds the measurement narrative behind it. A record whose title says "detail"
is the other direction: text moved OUT of the roadmap.

A closed plan moves to [`../archive/plans/`](../archive/plans/) with its row
([index](../archive/README.md#plans-and-audit-records-moved-2026-11-15)).

| plan | state | standing record |
|---|---|---|
| [2026-08-15-imp-quantize-roadmap](2026-08-15-imp-quantize-roadmap.md) | **OPEN**: items 2 (embedding opt-in), 4 (stacked experts); item 3 superseded 2026-10-01 (#2359) | roadmap Open 6, Open 7 (closed, archive) |
| [2026-09-28-qwen4exp-mtp](2026-09-28-qwen4exp-mtp.md) | record, MTP draft-layer reference semantics; drafting shipped (#2272) | CHANGELOG `speculative.mtp_k` |
