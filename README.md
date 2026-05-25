# Train Re-scheduling using Dynamic Discretization Discovery

This repository contains a Rust implementation of a train re-scheduling
algorithm based on the Dynamic Discretization Discovery (DDD) framework using
SAT/MaxSAT solvers. It also contains other solver implementations including
MILP baselines.

The codebase is the main artefact for the graduation thesis on Train
Rescheduling Problem (TRP) at UET, comparing 8 solver configurations across
72 problem instances and 3 delay-cost objectives.

## Solvers

| Solver | Paradigm | Role in thesis |
|---|---|---|
| **MaxSAT-Default** | MaxSAT core-guided (RC2) + 2 thesis contributions | Main proposed method |
| **MaxSAT-Base** | MaxSAT core-guided (RC2) | Baseline (Croella 2024) |
| **IncSAT-Default** / **Nothing** | SAT with binary search + incremental learning | SAT alternative |
| **PureSAT-Default** / **Nothing** | SAT with binary search + fresh rebuild | SAT alternative |
| **Big-M MILP** | Gurobi MILP with Big-M encoding | MILP baseline |
| **TI-MILP** | Gurobi MILP with time-indexed + DDD | MILP baseline |

Two thesis contributions are implemented in `MaxSAT-Default`:
1. **Sequential Counter AMO** encoding for resource conflict cliques of size ≥ 6
2. **Precedence graph preprocessing** via within-train chain propagation

## Problem instances

72 instances in `instances/`, split into 3 infrastructure models:

- `instances/original/` (24 files) — Croella 2024 baseline format
- `instances/addstationtime/` (24 files) — added dwell time at stations
- `instances/addtracktime/` (24 files) — added travel time on tracks

Each instance × 3 objectives (FiniteSteps123 / InfiniteSteps180 / Continuous) =
216 (instance, objective) pairs × 8 solvers = 1728 benchmark runs.

## Installation

### Prerequisites

- **Operating system**: Linux (Ubuntu 22.04 recommended). On Windows, install via WSL2.
- **Rust toolchain**: 1.83 or newer (`rustup install stable`)
- **Gurobi Optimizer**: version 12.0 with valid license (free academic license available)
- **Python 3**: for analysis scripts (3.10+ recommended)
- **GCC / build tools**: `build-essential` package

### Step 1: Clone external solver dependencies

The project links against 4 external SAT/MaxSAT solvers built from source. Clone
them as **siblings** of the main repo:

```bash
cd ~/projects   # or wherever you want
git clone https://github.com/NguyenTanUET/maxsattrainscheduling.git
git clone https://github.com/luteberget/salvers.git
git clone https://github.com/arminbiere/cadical.git
git clone https://github.com/marekpiotrow/uwrmaxsat.git
git clone https://github.com/biotomas/ipamir.git ipamir-rs
```

Final layout:

```
~/projects/
├── maxsattrainscheduling/    ← this repo
├── salvers/                   ← satcoder + IDL solver
├── cadical/                   ← CaDiCaL SAT solver
├── uwrmaxsat/                 ← UWrMaxSat MaxSAT solver
└── ipamir-rs/                 ← IPAMIR Rust bindings
```

### Step 2: Build external solvers

```bash
# CaDiCaL
cd ~/projects/cadical
./configure && make -j$(nproc)

# UWrMaxSat (requires CaDiCaL built first)
cd ~/projects/uwrmaxsat
make build_release -j$(nproc)

# IPAMIR Rust bindings
cd ~/projects/ipamir-rs
cargo build --release
```

### Step 3: Set up Gurobi

1. Download Gurobi 12.0 from [gurobi.com](https://www.gurobi.com/downloads/)
2. Install and acquire license (academic free)
3. Export environment variables:

```bash
# Add to ~/.bashrc
export GUROBI_HOME="/opt/gurobi1200/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
export GRB_LICENSE_FILE="$HOME/gurobi.lic"
```

### Step 4: Configure link paths

The build system links against external solver libraries via
`.cargo/config.toml`. Edit paths to match your installation:

```toml
[build]
rustflags = [
  "-L", "/path/to/ipamir-rs",
  "-L", "/path/to/uwrmaxsat/build/release/lib",
  "-L", "/path/to/cadical/build",
  "-L", "/path/to/gurobi/lib",
  "-l", "cadical",
]
```

Default paths in `.cargo/config.toml` assume `/mnt/d/github/...` (WSL2 mount).
**Update to your actual paths.**

### Step 5: Build the main project

```bash
cd ~/projects/maxsattrainscheduling
cargo build --release
```

The binary will be at `target/release/ddd`.

## Quick start

### Solve a single instance

```bash
# Solve with MaxSAT-Default solver (thesis main method)
./target/release/ddd \
    --solver MaxSatDddLadderSc \
    --delay-cost-type continuous \
    --timeout 120 \
    instances/original/InstanceA1.txt
```

### CLI options

| Flag | Values | Default |
|---|---|---|
| `--solver` | `MaxSatDddLadderSc`, `MaxSatDddLadderRC2`, `SatDddScTotalizer`, `SatDddScFreshAddClauses`, `BigMLazy`, `MipDdd`, ... | (required) |
| `--delay-cost-type` | `finite_steps123`, `infinite_steps180`, `continuous` | `continuous` |
| `--timeout` | seconds (e.g., `120`) | `120` |
| `--maxsat-ladder-sc-use-precedence-graph` | `true`/`false` | `true` |
| `--maxsat-ladder-sc-use-sc-amo` | `true`/`false` | `true` |
| `--maxsat-ladder-sc-use-touched-clique-amo` | `true`/`false` | `true` |

See `src/main.rs` for the full list of solver variants and flags.

### Run benchmark

Shell scripts in `quick_scripts/` automate batch runs:

```bash
# Example: run MaxSAT-Default on all 72 instances × 3 objectives
bash quick_scripts/bench/run_maxsat_default.sh
```

Results are written to CSV files (one row per (instance, run) with cost, lb,
ub, time, status, iterations).

## Repository structure

```
maxsattrainscheduling/
├── Cargo.toml                            # workspace manifest
├── .cargo/config.toml                    # external solver link paths
├── instances/                            # 72 TRP instances (3 infrastructure models)
├── src/
│   ├── main.rs                           # CLI + solver dispatch
│   ├── problem.rs                        # TRP problem definition (Problem, Train, Visit)
│   ├── parser.rs                         # Input file parser (.txt + .xml)
│   └── solvers/
│       ├── ddd/                          # 6 DDD solvers (MaxSAT + SAT variants)
│       ├── milp/                         # MILP solvers (Gurobi)
│       ├── legacy/                       # Pre-thesis solvers
│       └── util/                         # Heuristic + counting solver
├── crates/                               # Standalone Rust crates (heuristic, GUI)
├── docs/codebase/                        # Vietnamese technical documentation
├── quick_scripts/                        # Benchmark + analysis scripts
│   ├── bench/                            # Shell scripts for batch solver runs
│   ├── ablation/                         # Ablation study scripts
│   └── analyze/                          # Python scripts for table/plot generation
├── 2026-05-16-Verified-Result-For-Graduation-Thesis/
│                                         # Verified benchmark CSVs for thesis
└── chapter4_tables/                      # Generated tables for thesis Chapter 4
```

## Documentation

Detailed Vietnamese technical documentation is in [docs/codebase/](docs/codebase/):

- [`01-overview.md`](docs/codebase/01-overview.md) — Architecture overview
- [`02-dataset.md`](docs/codebase/02-dataset.md) — Instance format + dataset
- [`03-build-setup.md`](docs/codebase/03-build-setup.md) — Build environment details
- [`04-main-orchestration.md`](docs/codebase/04-main-orchestration.md) — CLI + dispatch
- [`05-ddd-solvers.md`](docs/codebase/05-ddd-solvers.md) — DDD solver internals
- [`07-shared-utils.md`](docs/codebase/07-shared-utils.md) — Shared utilities
- [`10-detailed-solver-walkthrough.md`](docs/codebase/10-detailed-solver-walkthrough.md) — Line-by-line solver walkthrough

## Generating thesis tables

After running benchmarks, generate the 7 tables for thesis Chapter 4:

```bash
python3 quick_scripts/analyze/build_chapter4_tables.py
```

Output: 7 CSV files in `chapter4_tables/` (one per table 4.4 through 4.10).
The script reads raw CSVs from `2026-05-16-Verified-Result-For-Graduation-Thesis/`.

## Troubleshooting

### Build fails with "cannot find -lcadical"

Make sure CaDiCaL is built (`make -j$(nproc)` in cadical directory) and
the `-L /path/to/cadical/build` line in `.cargo/config.toml` points to the
directory containing `libcadical.a`.

### Build fails with "cannot find Gurobi"

Verify `GUROBI_HOME` is exported and `$GUROBI_HOME/lib/libgurobi120.so` exists.
Add `-L $GUROBI_HOME/lib` to `.cargo/config.toml` if needed.

### Solver crashes with "license expired"

Gurobi requires an active license for the MILP solvers (Big-M, TI-MILP) and
the heuristic UB thread used by MaxSAT solvers. SAT-only solvers (`SatDdd*`,
`PureSat*`) work without Gurobi but won't have a heuristic UB.

### Out-of-memory on large instances

PureSAT variants are particularly memory-heavy on Continuous objective for
instances A11/A12. Use `quick_scripts/bench/*.sh` wrappers which set
`RAM_LIMIT_KB=15000000` (15GB) per subprocess for isolation.

## License + Citation

Source code released for academic research and reproducibility of the thesis
results. Original DDD-MaxSAT framework: Croella et al. 2024.
