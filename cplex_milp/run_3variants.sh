#!/bin/bash
# Chạy BigM + TI MILP trên 3 dataset variants với cùng cost type.
# Mỗi (solver × variant) có OUTPUT_DIR riêng để tránh ghi đè file .lp/.sol.
#
# Usage:
#   ./run_3variants.sh [cost_type=finsteps123] [timeout=120] [threads=4]
#
# Environment vars (optional):
#   TI_INTERVAL=30 TI_BIG_M=600  giảm model size cho TI (default 10/900)
#   SOLVERS="bigm ti"            chọn solver (default cả 2)
#
# Examples:
#   ./run_3variants.sh finsteps123 120 4
#   ./run_3variants.sh cont 180 8
#   TI_INTERVAL=30 TI_BIG_M=600 ./run_3variants.sh finsteps123 120 4
#   SOLVERS="bigm" ./run_3variants.sh finsteps123 120 4

set -e

COST_TYPE="${1:-finsteps123}"
TIMEOUT="${2:-120}"
THREADS="${3:-4}"

SOLVERS="${SOLVERS:-bigm ti}"
VARIANTS=(original addstationtime addtracktime)

START_ALL=$(date +%s)
echo "========================================"
echo "MILP Multi-Variant Benchmark"
echo "Cost type: $COST_TYPE"
echo "Timeout:   ${TIMEOUT}s per instance"
echo "Threads:   $THREADS"
echo "Solvers:   $SOLVERS"
echo "Variants:  ${VARIANTS[*]}"
if [ -n "$TI_INTERVAL" ] || [ -n "$TI_BIG_M" ]; then
    echo "TI params: interval=${TI_INTERVAL:-10}, big_m=${TI_BIG_M:-900}"
fi
echo "========================================"

for solver in $SOLVERS; do
    for variant in "${VARIANTS[@]}"; do
        VARIANT_DIR="../instances/$variant"

        if [ ! -d "$VARIANT_DIR" ]; then
            echo "WARNING: skip $variant (directory not found: $VARIANT_DIR)"
            continue
        fi

        N_INSTANCES=$(ls "$VARIANT_DIR"/*.txt 2>/dev/null | wc -l)

        echo ""
        echo "==============================================================="
        echo "[$(date +%H:%M:%S)] $solver × $variant ($N_INSTANCES instances)"
        echo "==============================================================="

        OUTPUT_DIR="results/${solver}/${variant}" \
            ./batch.sh "$VARIANT_DIR" "$solver" "$COST_TYPE" "$TIMEOUT" "$THREADS"

        echo ""
        echo "[$(date +%H:%M:%S)] Done $solver × $variant"
    done
done

END_ALL=$(date +%s)
TOTAL_TIME=$((END_ALL - START_ALL))

echo ""
echo "========================================"
echo "ALL COMPLETED in ${TOTAL_TIME}s"
echo "Cost type: $COST_TYPE"
echo "----------------------------------------"
echo "CSV outputs:"
for solver in $SOLVERS; do
    for variant in "${VARIANTS[@]}"; do
        DIR="results/${solver}/${variant}"
        if [ -d "$DIR" ]; then
            LATEST=$(ls -t $DIR/batch_${solver}_${COST_TYPE}_*.csv 2>/dev/null | head -1)
            if [ -n "$LATEST" ]; then
                echo "  [${solver}/${variant}] $LATEST"
            fi
        fi
    done
done
echo "========================================"
