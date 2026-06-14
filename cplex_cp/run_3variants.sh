#!/bin/bash
# Chạy CP benchmark cho 1 cost type trên 3 dataset variants.
# Mỗi variant có OUTPUT_DIR riêng để tránh ghi đè file .cpo/.mod/.dat.
#
# Usage:
#   ./run_3variants.sh [cost_type=finsteps123] [timeout=120] [threads=4]
#
# Examples:
#   ./run_3variants.sh finsteps123 120 4
#   ./run_3variants.sh infsteps180 120 4
#   ./run_3variants.sh cont 120 4

set -e

COST_TYPE="${1:-finsteps123}"
TIMEOUT="${2:-120}"
THREADS="${3:-4}"

VARIANTS=(original addstationtime addtracktime)

START_ALL=$(date +%s)
echo "========================================"
echo "CP Optimizer Multi-Variant Benchmark"
echo "Cost type: $COST_TYPE"
echo "Timeout:   ${TIMEOUT}s per instance"
echo "Threads:   $THREADS"
echo "Variants:  ${VARIANTS[*]}"
echo "========================================"

for variant in "${VARIANTS[@]}"; do
    VARIANT_DIR="../instances/$variant"

    if [ ! -d "$VARIANT_DIR" ]; then
        echo "WARNING: skipping $variant (directory not found: $VARIANT_DIR)"
        continue
    fi

    N_INSTANCES=$(ls "$VARIANT_DIR"/*.txt 2>/dev/null | wc -l)

    echo ""
    echo "==============================================================="
    echo "[$(date +%H:%M:%S)] Variant: $variant ($N_INSTANCES instances)"
    echo "==============================================================="

    OUTPUT_DIR="results/$variant" \
        ./batch.sh "$VARIANT_DIR" "$COST_TYPE" "$TIMEOUT" "$THREADS"

    echo ""
    echo "[$(date +%H:%M:%S)] Done variant: $variant"
done

END_ALL=$(date +%s)
TOTAL_TIME=$((END_ALL - START_ALL))

echo ""
echo "========================================"
echo "ALL 3 VARIANTS COMPLETED in ${TOTAL_TIME}s"
echo "Cost type: $COST_TYPE"
echo "----------------------------------------"
echo "CSV outputs:"
for variant in "${VARIANTS[@]}"; do
    if [ -d "results/$variant" ]; then
        LATEST=$(ls -t results/$variant/batch_cp_${COST_TYPE}_*.csv 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "  [$variant] $LATEST"
        fi
    fi
done
echo "========================================"
