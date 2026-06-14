#!/bin/bash
# Batch runner xuất CSV cùng schema với Gurobi mip_ddd:
#   algorithm_time, avg_tracks, conflicting_visit_pairs, conflicts, cost,
#   delay_cost_type, index, internal_cost, intervals, iteration, lb, name,
#   resource_constraints, sol_time, solver_name, solver_time, status,
#   total_time, trains, travel_constraints, ub
#
# Usage:
#   ./batch_gurobi.sh <instances_dir> <bigm|ti> <cost_type> [timeout=120] [threads=4] [filter]
#
# Env vars: TI_INTERVAL, TI_BIG_M
# Output: $OUTPUT_DIR/batch_<solver>_<cost>_<timestamp>.csv

set -e

INSTANCES_DIR="$1"
SOLVER="$2"
COST_TYPE="$3"
TIMEOUT="${4:-120}"
THREADS="${5:-4}"
FILTER="${6:-}"

if [ -z "$INSTANCES_DIR" ] || [ -z "$SOLVER" ] || [ -z "$COST_TYPE" ]; then
    echo "Usage: $0 <instances_dir> <bigm|ti> <cost_type> [timeout=120] [threads=4] [filter]"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-results}"
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUST_BIN="$PROJECT_ROOT/target/release/export_lp"

if ! command -v cplex &> /dev/null; then
    echo "Error: cplex not in PATH. Activate CPLEX env first."
    exit 1
fi
if [ ! -x "$RUST_BIN" ]; then
    echo "Error: Rust binary missing: $RUST_BIN"
    echo "Build with: cd $PROJECT_ROOT && cargo build --release --bin export_lp"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="$OUTPUT_DIR/batch_${SOLVER}_${COST_TYPE}_${TIMESTAMP}.csv"

# CSV header — exactly match Gurobi mip_ddd schema.
echo "algorithm_time,avg_tracks,conflicting_visit_pairs,conflicts,cost,delay_cost_type,index,internal_cost,intervals,iteration,lb,name,resource_constraints,sol_time,solver_name,solver_time,status,total_time,trains,travel_constraints,ub" > "$CSV_FILE"

shopt -s nullglob
if [ -n "$FILTER" ]; then
    INSTANCES=("$INSTANCES_DIR"/*"$FILTER"*.txt)
else
    INSTANCES=("$INSTANCES_DIR"/*.txt)
fi

if [ ${#INSTANCES[@]} -eq 0 ]; then
    echo "No instances found in $INSTANCES_DIR"
    exit 1
fi

echo "Found ${#INSTANCES[@]} instances. Output: $CSV_FILE"
echo ""

if [ "$SOLVER" = "ti" ]; then
    INTERVAL="${TI_INTERVAL:-10}"
    BIG_M="${TI_BIG_M:-900}"
fi

INDEX=0
for INSTANCE in "${INSTANCES[@]}"; do
    INST_NAME=$(basename "$INSTANCE" .txt)
    echo "[$(date +%H:%M:%S)] [$INDEX] $INST_NAME ($SOLVER, $COST_TYPE)"

    # File names tương ứng benchmark.sh.
    if [ "$SOLVER" = "ti" ]; then
        LP_BASE="${INST_NAME}_ti_${COST_TYPE}_i${INTERVAL}_m${BIG_M}"
    else
        LP_BASE="${INST_NAME}_bigm_${COST_TYPE}"
    fi
    LP_FILE="$OUTPUT_DIR/${LP_BASE}.lp"
    SOL_FILE="$OUTPUT_DIR/${LP_BASE}.sol"
    CPLEX_LOG="$OUTPUT_DIR/${LP_BASE}.cplex.log"
    BUILD_LOG="$OUTPUT_DIR/${LP_BASE}.build.log"

    # ── BUILD ──
    START_BUILD=$(date +%s.%N)
    if [ "$SOLVER" = "ti" ]; then
        TI_INTERVAL="$INTERVAL" TI_BIG_M="$BIG_M" \
            "$RUST_BIN" "$INSTANCE" "$SOLVER" "$COST_TYPE" "$OUTPUT_DIR" \
            > "$BUILD_LOG" 2>&1 || true
    else
        "$RUST_BIN" "$INSTANCE" "$SOLVER" "$COST_TYPE" "$OUTPUT_DIR" \
            > "$BUILD_LOG" 2>&1 || true
    fi
    END_BUILD=$(date +%s.%N)
    BUILD_TIME=$(echo "$END_BUILD - $START_BUILD" | bc -l)

    if [ ! -f "$LP_FILE" ]; then
        echo "  build failed"
        python3 "$SCRIPT_DIR/verify_gurobi.py" \
            "$INSTANCE" "$SOLVER" "$COST_TYPE" "NONE" \
            "$CPLEX_LOG" "$BUILD_TIME" "0" "$BUILD_LOG" "$INDEX" \
            >> "$CSV_FILE"
        INDEX=$((INDEX + 1))
        continue
    fi

    # ── SOLVE ──
    rm -f "$SOL_FILE"
    START_SOLVE=$(date +%s.%N)
    cplex -c \
        "set logfile $CPLEX_LOG" \
        "set timelimit $TIMEOUT" \
        "set threads $THREADS" \
        "set mip tolerances mipgap 0.001" \
        "set mip display 2" \
        "read $LP_FILE" \
        "optimize" \
        "write $SOL_FILE" \
        "quit" > /dev/null 2>&1 || true
    END_SOLVE=$(date +%s.%N)
    SOLVE_TIME=$(echo "$END_SOLVE - $START_SOLVE" | bc -l)

    SOL_ARG="$SOL_FILE"
    if [ ! -f "$SOL_FILE" ]; then
        SOL_ARG="NONE"
    fi

    # ── EMIT CSV ROW ──
    python3 "$SCRIPT_DIR/verify_gurobi.py" \
        "$INSTANCE" "$SOLVER" "$COST_TYPE" "$SOL_ARG" \
        "$CPLEX_LOG" "$BUILD_TIME" "$SOLVE_TIME" "$BUILD_LOG" "$INDEX" \
        >> "$CSV_FILE" 2>>"$OUTPUT_DIR/_verify_errors.log"

    INDEX=$((INDEX + 1))
done

echo ""
echo "========================================"
echo "Done: $CSV_FILE"
echo "========================================"
