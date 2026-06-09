#!/bin/bash
# Batch runner: chạy benchmark.sh trên nhiều instance + ghi CSV summary.
#
# Usage:
#   ./batch.sh <instances_dir> <bigm|ti> <cost_type> [timeout=120] [threads=4] [filter]
#
# Examples:
#   ./batch.sh ../instances/original bigm finsteps123 120 4
#   ./batch.sh ../instances/original ti cont 180 4 "InstanceA"
#
# Output:
#   results/batch_<solver>_<cost>_<timestamp>.csv

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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="$OUTPUT_DIR/batch_${SOLVER}_${COST_TYPE}_${TIMESTAMP}.csv"

# CSV header
echo "instance,solver,cost_type,timeout,build_time,solve_time,cost,gap,status,valid" > "$CSV_FILE"

# Find instances
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

for INSTANCE in "${INSTANCES[@]}"; do
    INST_NAME=$(basename "$INSTANCE" .txt)
    echo "[$(date +%H:%M:%S)] Processing $INST_NAME..."

    # Run benchmark.sh, capture output
    if OUTPUT=$(./benchmark.sh "$INSTANCE" "$SOLVER" "$COST_TYPE" "$TIMEOUT" "$THREADS" 2>&1); then
        # Parse summary
        BUILD_T=$(echo "$OUTPUT" | grep -oP 'Build:\s*\K[0-9.]+' | head -1)
        SOLVE_T=$(echo "$OUTPUT" | grep -oP 'Solve:\s*\K[0-9.]+' | head -1)
        COST=$(echo "$OUTPUT" | grep -oP 'Verified cost:\s*\K[0-9]+' | head -1)
        GAP=$(echo "$OUTPUT" | grep -oP 'MIP gap:\s*\K[0-9.]+' | head -1)
        STATUS=$(echo "$OUTPUT" | grep -oP 'CPLEX status:\s*\K\w+' | head -1)
        VALID=$(echo "$OUTPUT" | grep -oP 'Valid:\s*\K\w+' | head -1)

        echo "$INST_NAME,$SOLVER,$COST_TYPE,$TIMEOUT,${BUILD_T:-},${SOLVE_T:-},${COST:-},${GAP:-},${STATUS:-},${VALID:-}" >> "$CSV_FILE"
        echo "  → cost=${COST:-N/A} valid=${VALID:-N/A} (build=${BUILD_T:-N/A}s, solve=${SOLVE_T:-N/A}s)"
    else
        echo "  ✗ FAILED"
        echo "$INST_NAME,$SOLVER,$COST_TYPE,$TIMEOUT,,,,,error,false" >> "$CSV_FILE"
    fi
done

echo ""
echo "========================================"
echo "Batch complete: $CSV_FILE"
echo "========================================"
