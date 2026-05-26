# quick_scripts/

Utility scripts to run benchmarks, perform ablation studies, post-process
results, and debug regressions for the TRP MaxSAT/SAT/MILP solvers in this
repo.

## Layout

```
quick_scripts/
├── activation.txt                       # env setup (conda + gurobi license)
├── run_all_txt_instances_limited.sh     # core OOM-safe per-instance runner
├── bench/        # benchmark drivers (the things you actually launch)
├── ablation/     # SC/AMO ablation experiments (research-only)
├── analyze/      # post-processing & plotting (Python)
└── debug/        # debugging utilities (audits, bisect, value-trace diff)
```

The single core runner `run_all_txt_instances_limited.sh` stays at the top
level because every `bench/` script invokes it with `bash quick_scripts/...`
from the repo root.

## bench/ — benchmark drivers

Most scripts produce JSON+CSV under
`2026-05-16-Verified-Result-For-Graduation-Thesis/<subdir>/`.

| Script | What it runs |
|---|---|
| `maxsat_baseline.sh` | `maxsat_ddd_ladder` (Croella2024 baseline), 3 objectives |
| `maxsat_default.sh` | `maxsat_ddd_ladder_sc` with default settings, 3 objectives |
| `maxsat_ablation.sh` | MaxSAT-OnlyPrec + MaxSAT-OnlySC ablation, 3 objectives |
| `maxsat_4config_clause_var.sh` | 4-config MaxSAT (Base/Default/SC/Prec) with CountingSolver |
| `maxsat_threshold_sweep.sh` | MaxSAT-Default at given `N=PAIRWISE_AMO_MAX_SIZE` |
| `puresat_default.sh` | `sat_ddd_sc_fresh_addclauses`, OOM-safe, 1+ objectives |
| `puresat_default_nothing.sh` | PureSAT Default + Nothing configs, 3 objectives |
| `incsat_totalizer_cont.sh` | `sat_ddd_sc_totalizer` on cont (default cfg) |
| `incsat_bit_totalizer_cont.sh` | `sat_ddd_sc` with bit_totalizer encoding, cont |
| `incsat_totalizer_default_nothing.sh` | Inc-Tot Default + Nothing, 3 objectives |
| `incsat_totalizer_verify.sh` | Parameterized verification (objectives × configs) |
| `incsat_verify.sh` | Generic incremental-SAT verify suite |
| `sat_default_nothing.sh` | Plain SAT Default + Nothing |
| `milp_objectives_separate.sh` | Big-M + MIP-DDD per objective |
| `milp_matrix.sh` | Big-M + MIP-DDD + SAT matrix |
| `single_instance.sh` | Run one instance through baseline + SC |
| `thesis_benchmark.sh` | Full thesis verification set (2 solvers × 3 obj) |
| `repeated_benchmark.sh` | Repeated runs for variance estimation |
| `post_fix_verify.sh` | Regression-verify after a fix |

### Common invocation patterns

```bash
# Activate env first
source quick_scripts/activation.txt

# Build
cargo build --release

# Single config × 3 objectives
bash quick_scripts/bench/maxsat_default.sh

# Threshold sweep (recompile binary with N first)
N=5 bash quick_scripts/bench/maxsat_threshold_sweep.sh

# Background batch
nohup bash quick_scripts/bench/maxsat_default.sh > maxsat_default.log 2>&1 &
tail -f maxsat_default.log
```

## ablation/ — SC/AMO ablation studies

Research-only scripts probing the impact of individual encoding knobs.
Outputs typically go to `results_*.json` at repo root.

| Script | Question it answers |
|---|---|
| `amo_isolate.sh` | Isolate effect of SC AMO encoding alone |
| `cont_isolate.sh` | Which SC feature causes slowdown on cont? |
| `sc_isolate.sh` | use_sc_amo vs use_interval_graph_conflicts |
| `sc_amo_tune.sh` | Benchmark a single (PAIRWISE_AMO_MAX_SIZE, LAZY) tuple |
| `sc_config_compare.sh` | 4-way config comparison on chosen objective |
| `sc_threshold10_all_methods.sh` | All 3 SC-affected methods at threshold=10 |
| `touched_amo_compare.sh` | Effect of touched-clique AMO accumulation |
| `3way_sc_compare.sh` | Ldr vs ScOnly vs ScFull |

## analyze/ — post-processing & plotting

| Script | Purpose |
|---|---|
| `json_to_csv_batch.py` | Convert JSON results → CSV (compact or trace) |
| `aggregate_repeated_runs.py` | Aggregate repeated benchmark JSONs |
| `summarize_scamo.py` | Summarise SCAMO log lines |
| `plot_4config_hard_groups.py` | Generate Chapter 4 ablation figures (PDF+PNG) |
| `plot_instance_scatter.py` | Per-instance scatter (Solver-A vs Solver-B time) |
| `mk_table.py` | Convert JSON results → LaTeX table |

### CSV conversion examples

```bash
# Compact (one row per solve, list/dict fields dropped)
python3 quick_scripts/analyze/json_to_csv_batch.py \
    results_maxsat_ladder_infsteps180.json \
    --format compact --overwrite

# Trace (one row per value_trace event)
python3 quick_scripts/analyze/json_to_csv_batch.py \
    results_sat_sc_totalizer_infsteps180_cost_fix.json \
    --format trace --overwrite

# Batch
python3 quick_scripts/analyze/json_to_csv_batch.py \
    results_repeated/*.json \
    --format compact --overwrite
```

## debug/ — debugging utilities

| Script | Purpose |
|---|---|
| `bisect_origA1_regression.sh` | Git-bisect around the origA1 regression |
| `compare_value_trace.py` | Diff two `value_trace` arrays between runs |
| `audit_dups.py` | Find duplicate `\label{}` entries in LaTeX |
| `audit_refs.py` | Find dangling `\ref{}`/`\eqref{}` entries |

## Conventions

- All `bench/` scripts honour these env vars where applicable:
  - `BIN` (default `./target/release/ddd`)
  - `OUT_DIR_BASE` / `OUT_DIR`
  - `OBJECTIVES` (space-separated, default `"finsteps123 infsteps180 cont"`)
  - `INSTANCE_TIMEOUT_SECS` (default 150s)
  - `RAM_LIMIT_KB` (default 15 GB)
- Outputs land under `2026-05-16-Verified-Result-For-Graduation-Thesis/`
  by default — change `OUT_DIR_BASE` to redirect.
- CSV is regenerated alongside each JSON via `analyze/json_to_csv_batch.py`.
