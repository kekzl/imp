# tools/roofline — reproduzierbare Roofline- & Coverage-Pipeline

Misst Roofline-Nähe (%-Roofline, AI, achieved FLOPS/BW) und Kernel-Coverage
(Legacy-Fallback-Anteile) der imp-Hot-Path-Kernels auf der RTX 5090 (sm_120a),
mit append-only Historie pro Commit.

## Quickstart

```bash
tools/roofline/roofline measure                 # voller Sweep, 3 Restarts (~Stunden, GPU exklusiv)
tools/roofline/roofline measure --models q8-dense --shapes tg256 --restarts 1   # Smoke
tools/roofline/roofline plot --latest           # roofline.png + roofline_trend.png
tools/roofline/roofline plot --compare RUN_A RUN_B
tools/roofline/roofline report --run latest -o audit/roofline_$(date +%Y_%m_%d).md
tools/roofline/roofline regress --baseline <run_id|sha> --threshold 5
tools/roofline/roofline issues --run latest     # dry-run; --create legt GitHub-Issues an
tools/roofline/roofline ab --knob fa2           # unprofiled A/B (FA2 on vs never)
```

## Architektur

- **Messung** (`measure`): pro Zelle (Modell × Shape × Restart) zwei Pässe in
  frischen Docker-Containern (= Restart-Varianz by construction):
  1. **nsys** (volle Timeline): Kernel-Zeitanteile, cuBLAS-API-Attribution
     (batched GEMM ⇒ Legacy-Attention QK^T/PV), Phasen-Split (init/prefill/
     decode) und Kalibrierung des ncu `--launch-skip` (Steady-State-Fenster).
  2. **ncu** (gepinntes Counter-Set, `--clock-control base`): pro Launch
     Zeit, `dram__bytes`, Tensor-Pipe-FLOP-Counter, SM/DRAM-%, Occupancy.
- **Historie** (`history/`): append-only.
  - `runs/<run_id>.json` — geparster Run (committed). `run_id = <shortSHA>[-dirty]_<timestamp>`.
  - `index.jsonl` — eine Zeile pro Run für Trend-Queries (committed).
  - `raw/<run_id>/*.ncu_raw.csv.gz` + `*.nsys_extract.json` — Roh-Exporte,
    Re-Parse ohne Re-Measure (committed).
  - `raw/<run_id>/*.ncu-rep|*.nsys-rep|*.sqlite` — binäre Originale,
    **nur lokal** (gitignored; Release-Check verbietet sie im Repo).
- **Plots** (`plot`): matplotlib im Container `imp-roofline-plot`
  (Dockerfile.plot, Host bleibt clean) — rendert ausschließlich aus History.
- **Report** (`report`): Markdown mit Modul-1-Tabelle, Modul-2-Coverage-Matrix
  und priorisierter Lever-Liste; jede Zahl referenziert den Run (commit+ts).
- **Gate** (`regress`): exit≠0, wenn eine Kernel-Klasse (Zeitanteil ≥0.5%)
  im Median > Schwelle unter die Baseline fällt UND die Restart-Spannen
  disjunkt sind (sonst Varianz, kein Fail).

## Determinismus / Methodik

- Counter-Set, Shapes, Peaks und Klassifikations-Regexes liegen versioniert in
  `config.json` (`config_version` — Läufe nur innerhalb gleicher Version vergleichen).
- ncu lockt Clocks auf Base (`--clock-control base`); Compute-Peaks werden auf
  den **gemessenen** SM-Takt normiert (`gpc__cycles_elapsed.avg.per_second`),
  Ridge-Points auf Boost-Takt (2.407 GHz) — beides in config.json.
- AI = FLOPs / `dram__bytes.sum` (gemessener DRAM-Traffic, nicht geschätzt).
- **FLOP counting**: Tensor-Core-FLOPs aus `sm__ops_path_tensor_src_*`-Countern
  (Kalibrierung gegen bekannte GEMM-Shapes, siehe Report-Methodik). Non-TC:
  SASS-Thread-Instruktionen (ffma=2 FLOP; hfma als gepackt HFMA2=4 FLOP
  gezählt — obere Schranke, im Report als solche markiert).
- Profil-Läufe nutzen `--no-cuda-graphs` (Graph-Replay versteckt Kernel,
  Kernel-Mix ist identisch — siehe docs/MISSION_JOURNAL/memory).
- Prefill-Zellen messen `--bench-reps 3 --max-tokens 1`; Decode-Zellen
  `--bench-pp 64 --max-tokens 256`. pp-Restart-Varianz (bekannt bis 2.6×)
  wird als min/med/max ausgewiesen, nie weggemittelt.

## CI

GPU-Messung ist LOKAL (CI hat keinen GPU-Runner). Der Workflow
`.github/workflows/roofline.yml` prüft auf jedem PR nur Parser/Mathe gegen die
eingecheckte History (re-parse) und rendert Plots als Artefakt. Baseline-Pinning:
nach Merge auf main lokal `roofline measure` + Commit der History; `regress`
läuft im pre-push-Hook, wenn eine Baseline gepinnt ist (`history/BASELINE`).
