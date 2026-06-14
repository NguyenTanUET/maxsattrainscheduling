#!/bin/bash
# CP batch runner xuất CSV cùng schema với Gurobi mip_ddd.
#
# Usage:
#   ./batch_gurobi.sh <instances_dir> <cost_type> [timeout=120] [threads=4] [filter]

set -e

INSTANCES_DIR="$1"
COST_TYPE="$2"
TIMEOUT="${3:-120}"
THREADS="${4:-4}"
FILTER="${5:-}"

if [ -z "$INSTANCES_DIR" ] || [ -z "$COST_TYPE" ]; then
    echo "Usage: $0 <instances_dir> <cost_type> [timeout=120] [threads=4] [filter]"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-results}"
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUST_BIN="$PROJECT_ROOT/target/release/export_opl"

# oplrun detection (copy logic from benchmark.sh).
OPLRUN_BIN="oplrun"
if [ -n "$OPL_HOME" ] && [ -x "$OPL_HOME/bin/x86-64_linux/oplrun" ]; then
    OPLRUN_BIN="$OPL_HOME/bin/x86-64_linux/oplrun"
fi
if [ ! -x "$OPLRUN_BIN" ] && ! command -v oplrun &> /dev/null; then
    for c in \
        /opt/ibm/ILOG/CPLEX_Studio222/opl/bin/x86-64_linux/oplrun \
        /opt/ibm/ILOG/CPLEX_Studio2210/opl/bin/x86-64_linux/oplrun \
        /opt/ibm/ILOG/CPLEX_Studio221/opl/bin/x86-64_linux/oplrun; do
        if [ -x "$c" ]; then
            OPLRUN_BIN="$c"
            export OPL_HOME="$(dirname $(dirname $(dirname $c)))"
            break
        fi
    done
fi
if ! command -v "$OPLRUN_BIN" &> /dev/null && [ ! -x "$OPLRUN_BIN" ]; then
    echo "Error: oplrun not found. Set OPL_HOME or add to PATH."
    exit 1
fi
if [ -n "$OPL_HOME" ]; then
    export LD_LIBRARY_PATH="$OPL_HOME/bin/x86-64_linux:$LD_LIBRARY_PATH"
fi
if [ ! -x "$RUST_BIN" ]; then
    echo "Error: Rust binary missing: $RUST_BIN"
    echo "Build with: cd $PROJECT_ROOT && cargo build --release --bin export_opl"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="$OUTPUT_DIR/batch_cp_${COST_TYPE}_${TIMESTAMP}.csv"
echo "algorithm_time,avg_tracks,conflicting_visit_pairs,conflicts,cost,delay_cost_type,index,internal_cost,intervals,iteration,lb,name,resource_constraints,sol_time,solver_name,solver_time,status,total_time,trains,travel_constraints,ub" > "$CSV_FILE"

shopt -s nullglob
if [ -n "$FILTER" ]; then
    INSTANCES=("$INSTANCES_DIR"/*"$FILTER"*.txt)
else
    INSTANCES=("$INSTANCES_DIR"/*.txt)
fi
if [ ${#INSTANCES[@]} -eq 0 ]; then
    echo "No instances in $INSTANCES_DIR"
    exit 1
fi

echo "Found ${#INSTANCES[@]} instances. Output: $CSV_FILE"
echo ""

INDEX=0
for INSTANCE in "${INSTANCES[@]}"; do
    INST_NAME=$(basename "$INSTANCE" .txt)
    echo "[$(date +%H:%M:%S)] [$INDEX] $INST_NAME (CP, $COST_TYPE)"

    OPL_BASE="${INST_NAME}_cp_${COST_TYPE}"
    MOD_FILE="$OUTPUT_DIR/${OPL_BASE}.mod"
    DAT_FILE="$OUTPUT_DIR/${OPL_BASE}.dat"
    OUT_FILE="$OUTPUT_DIR/${OPL_BASE}.out"

    # ── BUILD ──
    START_BUILD=$(date +%s.%N)
    CPO_TIME_LIMIT="$TIMEOUT" CPO_WORKERS="$THREADS" \
        "$RUST_BIN" "$INSTANCE" "$COST_TYPE" "$OUTPUT_DIR" > /dev/null 2>&1 || true
    END_BUILD=$(date +%s.%N)
    BUILD_TIME=$(echo "$END_BUILD - $START_BUILD" | bc -l)

    if [ ! -f "$MOD_FILE" ] || [ ! -f "$DAT_FILE" ]; then
        echo "  build failed"
        python3 "$SCRIPT_DIR/verify_gurobi.py" \
            "$INSTANCE" "$COST_TYPE" "NONE" "$BUILD_TIME" "0" "$INDEX" \
            >> "$CSV_FILE"
        INDEX=$((INDEX + 1))
        continue
    fi

    # ── SOLVE ──
    START_SOLVE=$(date +%s.%N)
    "$OPLRUN_BIN" "$MOD_FILE" "$DAT_FILE" > "$OUT_FILE" 2>&1 || true
    END_SOLVE=$(date +%s.%N)
    SOLVE_TIME=$(echo "$END_SOLVE - $START_SOLVE" | bc -l)

    OUT_ARG="$OUT_FILE"
    [ ! -f "$OUT_FILE" ] && OUT_ARG="NONE"

    # ── EMIT CSV ROW ──
    python3 "$SCRIPT_DIR/verify_gurobi.py" \
        "$INSTANCE" "$COST_TYPE" "$OUT_ARG" "$BUILD_TIME" "$SOLVE_TIME" "$INDEX" \
        >> "$CSV_FILE" 2>>"$OUTPUT_DIR/_verify_errors.log"

    INDEX=$((INDEX + 1))
done

echo ""
echo "========================================"
echo "Done: $CSV_FILE"
echo "========================================"
