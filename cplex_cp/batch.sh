#!/bin/bash
# Batch runner cho CP: chạy benchmark.sh trên nhiều instance + ghi CSV.
#
# Usage:
#   ./batch.sh <instances_dir> <cost_type> [timeout=120] [threads=4] [filter]
#
# Examples:
#   ./batch.sh ../instances/original finsteps123 120 4
#   ./batch.sh ../instances/original cont 180 4 "InstanceA"

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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="$OUTPUT_DIR/batch_cp_${COST_TYPE}_${TIMESTAMP}.csv"

echo "instance,solver,cost_type,timeout,build_time,solve_time,cpo_obj,verified_cost,status,valid" > "$CSV_FILE"

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

    if OUTPUT=$(./benchmark.sh "$INSTANCE" "$COST_TYPE" "$TIMEOUT" "$THREADS" 2>&1); then
        BUILD_T=$(echo "$OUTPUT" | grep -oP 'Build:\s*\K[0-9.]+' | head -1)
        SOLVE_T=$(echo "$OUTPUT" | grep -oP 'Solve:\s*\K[0-9.]+' | head -1)
        CPO_OBJ=$(echo "$OUTPUT" | grep -oP 'CPO objective:\s*\K[0-9.]+' | head -1)
        COST=$(echo "$OUTPUT" | grep -oP 'Verified cost:\s*\K[0-9]+' | head -1)
        STATUS=$(echo "$OUTPUT" | grep -oP 'CPO status:\s*\K\w+' | head -1)
        VALID=$(echo "$OUTPUT" | grep -oP 'Valid:\s*\K\w+' | head -1)

        echo "$INST_NAME,cp,$COST_TYPE,$TIMEOUT,${BUILD_T:-},${SOLVE_T:-},${CPO_OBJ:-},${COST:-},${STATUS:-},${VALID:-}" >> "$CSV_FILE"
        echo "  -> cost=${COST:-N/A} valid=${VALID:-N/A} (build=${BUILD_T:-N/A}s, solve=${SOLVE_T:-N/A}s)"
    else
        echo "  x FAILED"
        echo "$INST_NAME,cp,$COST_TYPE,$TIMEOUT,,,,,error,false" >> "$CSV_FILE"
    fi
done

echo ""
echo "========================================"
echo "Batch complete: $CSV_FILE"
echo "========================================"
