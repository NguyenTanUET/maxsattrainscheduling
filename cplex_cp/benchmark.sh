#!/bin/bash
# Orchestrator: build → solve → verify cho TRP CP với cpoptimizer CLI.
#
# 3 bước:
#   1. Build model bằng Rust binary export_cp → file .cpo
#   2. cpoptimizer CLI: read .cpo + optimize + write output
#   3. Verify (Python): parse output + verify schedule
#
# Usage:
#   ./benchmark.sh <instance.txt> <cost_type> [timeout=120] [threads=4]
#
# Environment vars:
#   SKIP_VERIFY=1        bỏ qua bước verify Python
#   CPO_WORKERS=N        số worker threads cho CP (default = threads)
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
RUST_BIN="$PROJECT_ROOT/target/release/export_cp"

# Check cpoptimizer CLI
CPO_BIN="cpoptimizer"
if [ -n "$CPLEX_HOME" ]; then
    # CPLEX_HOME = /opt/ibm/.../cplex
    # cpoptimizer is at sibling /opt/ibm/.../cpoptimizer/bin/x86-64_linux/cpoptimizer
    CPO_CANDIDATE="$(dirname $(dirname $CPLEX_HOME))/cpoptimizer/bin/x86-64_linux/cpoptimizer"
    if [ -x "$CPO_CANDIDATE" ]; then
        CPO_BIN="$CPO_CANDIDATE"
    fi
fi

if ! command -v "$CPO_BIN" &> /dev/null && [ ! -x "$CPO_BIN" ]; then
    echo "Error: cpoptimizer not found"
    echo "  Tried: $CPO_BIN"
    echo "  Set CPLEX_HOME or add cpoptimizer to PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
BASE=$(basename "$INSTANCE" .txt)
CPO_BASE="${BASE}_cp_${COST_TYPE}"

CPO_FILE="$OUTPUT_DIR/${CPO_BASE}.cpo"
OUT_FILE="$OUTPUT_DIR/${CPO_BASE}.out"
CPO_LOG="$OUTPUT_DIR/${CPO_BASE}.cpo.log"
SUMMARY="$OUTPUT_DIR/${CPO_BASE}.summary.txt"

echo "========================================"
echo "Instance: $INSTANCE"
echo "Solver:   CP Optimizer"
echo "Cost:     $COST_TYPE"
echo "Timeout:  ${TIMEOUT}s"
echo "Threads:  $THREADS"
echo "Output:   $OUTPUT_DIR/"
echo "CPO bin:  $CPO_BIN"
echo "========================================"

# ===== BUOC 1: BUILD =====
echo ""
echo "[1/3] Building CP model (Rust binary)..."
START_BUILD=$(date +%s.%N)

if [ ! -x "$RUST_BIN" ]; then
    echo "Error: Rust binary not found at $RUST_BIN"
    echo "Build with: cd $PROJECT_ROOT && cargo build --release --bin export_cp"
    exit 2
fi

"$RUST_BIN" "$INSTANCE" "$COST_TYPE" "$OUTPUT_DIR"

END_BUILD=$(date +%s.%N)
BUILD_TIME=$(echo "$END_BUILD - $START_BUILD" | bc -l)

if [ ! -f "$CPO_FILE" ]; then
    echo "Error: .cpo file not created: $CPO_FILE"
    exit 2
fi

# ===== BUOC 2: SOLVE =====
echo ""
echo "[2/3] Solving with cpoptimizer CLI..."
START_SOLVE=$(date +%s.%N)

# Note: cpoptimizer CLI has -workers and TimeLimit settings
# Output: stdout có status, objective, solution
"$CPO_BIN" \
    -TimeLimit "$TIMEOUT" \
    -Workers "${CPO_WORKERS:-$THREADS}" \
    -OptimalityTolerance 0.001 \
    "$CPO_FILE" 2>&1 | tee "$CPO_LOG" | tail -50

# Save full output for verify
cp "$CPO_LOG" "$OUT_FILE"

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
Solver: CP Optimizer
Cost type: $COST_TYPE
Timeout: ${TIMEOUT}s
Threads: $THREADS

Build time: ${BUILD_TIME}s
Solve time: ${SOLVE_TIME}s

Files:
  CPO: $CPO_FILE
  OUT: $OUT_FILE
  Log: $CPO_LOG

$VERIFY_OUT
EOF

echo ""
echo "========================================"
echo "Build: ${BUILD_TIME}s | Solve: ${SOLVE_TIME}s"
echo "Summary: $SUMMARY"
echo "========================================"
