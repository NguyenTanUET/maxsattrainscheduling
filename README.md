# Train Re-scheduling using Dynamic Discretization Discovery

This repository contains a Rust implementation of a train re-scheduling
algorithm based on the Dynamic Discretization Discovery (DDD) framework using
SAT/MaxSAT solvers. It also contains other solver implementations including
MILP baselines.

The codebase is the main artefact for the graduation thesis on Train
Rescheduling Problem (TRP) at UET, comparing 8 solver configurations across
72 problem instances and 3 delay-cost objectives.

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

