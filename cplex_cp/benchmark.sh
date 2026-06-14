#!/bin/bash
# Orchestrator: build → solve → verify cho TRP CP với OPL + oplrun CLI.
#
# Workflow (giống MILP pattern):
#   1. Rust binary export_opl → .mod (model) + .dat (data)
#   2. oplrun CLI: solve OPL → text output
#   3. Python verify.py: parse output + verify schedule
#
# Usage:
#   ./benchmark.sh <instance.txt> <cost_type> [timeout=120] [threads=4]
#
# Environment vars:
#   SKIP_VERIFY=1      bỏ qua verify
#   OPL_HOME=<path>    path đến CPLEX_Studio/opl (default tự detect)
#
# Examples:
#   ./benchmark.sh ../instances/original/InstanceA1.txt finsteps123 120 4
#   ./benchmark.sh ../instances/original/InstanceA1.txt cont 180 8

set -e

INSTANCE="$1"
COST_TYPE="$2"
TIMEOUT="${3:-120}"
THREADS="${4:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

if [ -z "$INSTANCE" ] || [ -z "$COST_TYPE" ]; then
    echo "Usage: $0 <instance.txt> <cost_type> [timeout=120] [threads=4]"
    echo ""
    echo "Cost types: finsteps123, infsteps180, cont, ..."
    exit 1
fi

if [ ! -f "$INSTANCE" ]; then
    echo "Error: instance file not found: $INSTANCE"
    exit 1
fi

# Resolve dirs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUST_BIN="$PROJECT_ROOT/target/release/export_opl"

# Detect oplrun
OPLRUN_BIN="oplrun"
if [ -n "$OPL_HOME" ]; then
    if [ -x "$OPL_HOME/bin/x86-64_linux/oplrun" ]; then
        OPLRUN_BIN="$OPL_HOME/bin/x86-64_linux/oplrun"
    fi
fi

# Common path: CPLEX_Studio.../opl/bin/x86-64_linux/oplrun
if [ ! -x "$OPLRUN_BIN" ] && ! command -v oplrun &> /dev/null; then
    for candidate in \
        /opt/ibm/ILOG/CPLEX_Studio222/opl/bin/x86-64_linux/oplrun \
        /opt/ibm/ILOG/CPLEX_Studio2210/opl/bin/x86-64_linux/oplrun \
        /opt/ibm/ILOG/CPLEX_Studio221/opl/bin/x86-64_linux/oplrun; do
        if [ -x "$candidate" ]; then
            OPLRUN_BIN="$candidate"
            export OPL_HOME="$(dirname $(dirname $(dirname $candidate)))"
            break
        fi
    done
fi

if ! command -v "$OPLRUN_BIN" &> /dev/null && [ ! -x "$OPLRUN_BIN" ]; then
    echo "Error: oplrun not found"
    echo "  Set OPL_HOME or add oplrun to PATH"
    exit 1
fi

# Setup LD_LIBRARY_PATH if OPL_HOME set
if [ -n "$OPL_HOME" ]; then
    export LD_LIBRARY_PATH="$OPL_HOME/bin/x86-64_linux:$LD_LIBRARY_PATH"
fi

mkdir -p "$OUTPUT_DIR"
BASE=$(basename "$INSTANCE" .txt)
OPL_BASE="${BASE}_cp_${COST_TYPE}"

MOD_FILE="$OUTPUT_DIR/${OPL_BASE}.mod"
DAT_FILE="$OUTPUT_DIR/${OPL_BASE}.dat"
OUT_FILE="$OUTPUT_DIR/${OPL_BASE}.out"
LOG_FILE="$OUTPUT_DIR/${OPL_BASE}.oplrun.log"
SUMMARY="$OUTPUT_DIR/${OPL_BASE}.summary.txt"

echo "========================================"
echo "Instance: $INSTANCE"
echo "Solver:   OPL + CP Optimizer"
echo "Cost:     $COST_TYPE"
echo "Timeout:  ${TIMEOUT}s"
echo "Threads:  $THREADS"
echo "Output:   $OUTPUT_DIR/"
echo "oplrun:   $OPLRUN_BIN"
echo "========================================"

# ===== BUOC 1: BUILD =====
echo ""
echo "[1/3] Building OPL model (Rust binary)..."
START_BUILD=$(date +%s.%N)

if [ ! -x "$RUST_BIN" ]; then
    echo "Error: Rust binary not found at $RUST_BIN"
    echo "Build with: cd $PROJECT_ROOT && cargo build --release --bin export_opl"
    exit 2
fi

# Pass solver parameters via env vars - Rust binary embeds them in .mod
CPO_TIME_LIMIT="$TIMEOUT" CPO_WORKERS="$THREADS" \
    "$RUST_BIN" "$INSTANCE" "$COST_TYPE" "$OUTPUT_DIR"

END_BUILD=$(date +%s.%N)
BUILD_TIME=$(echo "$END_BUILD - $START_BUILD" | bc -l)

if [ ! -f "$MOD_FILE" ] || [ ! -f "$DAT_FILE" ]; then
    echo "Error: OPL files not created"
    exit 2
fi

# ===== BUOC 2: SOLVE =====
echo ""
echo "[2/3] Solving with oplrun CLI..."
START_SOLVE=$(date +%s.%N)

# oplrun simple usage: oplrun model.mod data.dat
# Parameters (TimeLimit, Workers) embedded trong .mod qua execute SETTINGS
"$OPLRUN_BIN" "$MOD_FILE" "$DAT_FILE" 2>&1 | tee "$LOG_FILE" | tail -50

# Save full output
cp "$LOG_FILE" "$OUT_FILE"

END_SOLVE=$(date +%s.%N)
SOLVE_TIME=$(echo "$END_SOLVE - $START_SOLVE" | bc -l)

# ===== BUOC 3: VERIFY =====
if [ "$SKIP_VERIFY" = "1" ]; then
    echo ""
    echo "[3/3] Verify skipped (SKIP_VERIFY=1)"
    VERIFY_OUT=""
else
    echo ""
    echo "[3/3] Verifying solution..."
    python "$SCRIPT_DIR/verify.py" "$INSTANCE" "$COST_TYPE" "$OUT_FILE" | tee /tmp/_verify_out.txt
    VERIFY_OUT=$(cat /tmp/_verify_out.txt)
fi

# ===== SUMMARY =====
cat > "$SUMMARY" <<EOF
Instance: $INSTANCE
Solver: OPL + CP Optimizer
Cost type: $COST_TYPE
Timeout: ${TIMEOUT}s
Threads: $THREADS

Build time: ${BUILD_TIME}s
Solve time: ${SOLVE_TIME}s

Files:
  MOD: $MOD_FILE
  DAT: $DAT_FILE
  OUT: $OUT_FILE
  oplrun: $OPLRUN_BIN

$VERIFY_OUT
EOF

echo ""
echo "========================================"
echo "Build: ${BUILD_TIME}s | Solve: ${SOLVE_TIME}s"
echo "Summary: $SUMMARY"
echo "========================================"
