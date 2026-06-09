#!/bin/bash
# Orchestrator: build -> solve -> verify cho TRP MILP với CPLEX CLI.
#
# 3 bước:
#   1. Build model bằng Rust binary export_lp → file .lp
#   2. CPLEX CLI: read .lp + optimize + write .sol
#   3. Verify (Python): parse .sol + TI post-processing + compute cost
#
# Usage:
#   ./benchmark.sh <instance.txt> <bigm|ti> <cost_type> [timeout=120] [threads=4]
#
# Environment vars:
#   SKIP_VERIFY=1        bỏ qua bước verify Python
#   TI_INTERVAL=10       (chỉ cho TI) bước rời rạc thời gian
#   TI_BIG_M=900         (chỉ cho TI) khoảng range từ earliest
#
# Examples:
#   ./benchmark.sh ../instances/original/InstanceA1.txt bigm finsteps123 120 4
#   TI_INTERVAL=30 TI_BIG_M=600 \
#       ./benchmark.sh ../instances/original/InstanceA1.txt ti cont 180 8

set -e

INSTANCE="$1"
SOLVER="$2"
COST_TYPE="$3"
TIMEOUT="${4:-120}"
THREADS="${5:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

if [ -z "$INSTANCE" ] || [ -z "$SOLVER" ] || [ -z "$COST_TYPE" ]; then
    echo "Usage: $0 <instance.txt> <bigm|ti> <cost_type> [timeout=120] [threads=4]"
    exit 1
fi

if [ ! -f "$INSTANCE" ]; then
    echo "Error: instance file not found: $INSTANCE"
    exit 1
fi

# Resolve script dir to find Rust binary
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUST_BIN="$PROJECT_ROOT/target/release/export_lp"

# Check CPLEX CLI
if ! command -v cplex &> /dev/null; then
    echo "Error: cplex not in PATH. Set CPLEX_HOME + source ~/.bashrc."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
BASE=$(basename "$INSTANCE" .txt)

# Tính tên file output dựa trên solver + cost
if [ "$SOLVER" = "ti" ]; then
    INTERVAL="${TI_INTERVAL:-10}"
    BIG_M="${TI_BIG_M:-900}"
    LP_BASE="${BASE}_ti_${COST_TYPE}_i${INTERVAL}_m${BIG_M}"
else
    LP_BASE="${BASE}_bigm_${COST_TYPE}"
fi

LP_FILE="$OUTPUT_DIR/${LP_BASE}.lp"
SOL_FILE="$OUTPUT_DIR/${LP_BASE}.sol"
CPLEX_LOG="$OUTPUT_DIR/${LP_BASE}.cplex.log"
SUMMARY="$OUTPUT_DIR/${LP_BASE}.summary.txt"

echo "========================================"
echo "Instance: $INSTANCE"
echo "Solver:   $SOLVER"
echo "Cost:     $COST_TYPE"
echo "Timeout:  ${TIMEOUT}s"
echo "Threads:  $THREADS"
echo "Output:   $OUTPUT_DIR/"
echo "========================================"

# ===== BUOC 1: BUILD =====
echo ""
echo "[1/3] Building MILP model (Rust binary)..."
START_BUILD=$(date +%s.%N)

if [ ! -x "$RUST_BIN" ]; then
    echo "Error: Rust binary not found at $RUST_BIN"
    echo "Build with: cd $PROJECT_ROOT && cargo build --release --bin export_lp"
    exit 2
fi

if [ "$SOLVER" = "ti" ]; then
    TI_INTERVAL="$INTERVAL" TI_BIG_M="$BIG_M" \
        "$RUST_BIN" "$INSTANCE" "$SOLVER" "$COST_TYPE" "$OUTPUT_DIR"
else
    "$RUST_BIN" "$INSTANCE" "$SOLVER" "$COST_TYPE" "$OUTPUT_DIR"
fi

END_BUILD=$(date +%s.%N)
BUILD_TIME=$(echo "$END_BUILD - $START_BUILD" | bc -l)

if [ ! -f "$LP_FILE" ]; then
    echo "Error: .lp file not created: $LP_FILE"
    exit 2
fi

# ===== BUOC 2: SOLVE =====
echo ""
echo "[2/3] Solving with CPLEX CLI..."
START_SOLVE=$(date +%s.%N)

rm -f "$SOL_FILE"

cplex -c \
    "set logfile $CPLEX_LOG" \
    "set timelimit $TIMEOUT" \
    "set threads $THREADS" \
    "set mip tolerances mipgap 0.001" \
    "set mip display 2" \
    "read $LP_FILE" \
    "optimize" \
    "write $SOL_FILE" \
    "quit" 2>&1 | tail -30

END_SOLVE=$(date +%s.%N)
SOLVE_TIME=$(echo "$END_SOLVE - $START_SOLVE" | bc -l)

if [ ! -f "$SOL_FILE" ]; then
    echo "Error: .sol file not created. Check $CPLEX_LOG for details."
    exit 3
fi

# ===== BUOC 3: VERIFY =====
if [ "$SKIP_VERIFY" = "1" ]; then
    echo ""
    echo "[3/3] Verify skipped (SKIP_VERIFY=1)"
    VERIFY_OUT=""
else
    echo ""
    echo "[3/3] Verifying solution..."
    python "$SCRIPT_DIR/verify.py" "$INSTANCE" "$SOLVER" "$COST_TYPE" "$SOL_FILE" | tee /tmp/_verify_out.txt
    VERIFY_OUT=$(cat /tmp/_verify_out.txt)
fi

# ===== SUMMARY =====
cat > "$SUMMARY" <<EOF
Instance: $INSTANCE
Solver: $SOLVER
Cost type: $COST_TYPE
Timeout: ${TIMEOUT}s
Threads: $THREADS

Build time: ${BUILD_TIME}s (Rust binary: ${RUST_BIN##*/})
Solve time: ${SOLVE_TIME}s

Files:
  LP: $LP_FILE
  SOL: $SOL_FILE
  CPLEX log: $CPLEX_LOG

$VERIFY_OUT
EOF

echo ""
echo "========================================"
echo "Build: ${BUILD_TIME}s | Solve: ${SOLVE_TIME}s"
echo "Summary: $SUMMARY"
echo "========================================"
