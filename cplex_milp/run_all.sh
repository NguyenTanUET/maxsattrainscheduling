#!/bin/bash
# Chạy BigM + TI trên 3 dataset variants × 3 cost types.
# Tổng cộng: 3 cost × 2 solver × 3 variant = 18 batches.
#
# Usage:
#   ./run_all.sh [timeout=120] [threads=4]
#
# Environment vars (optional):
#   TI_INTERVAL=30 TI_BIG_M=600  giảm model size cho TI (default 10/900)
#   SOLVERS="bigm ti"            chọn solver (default cả 2)
#   COST_TYPES="finsteps123 infsteps180 cont"   chọn cost types (default cả 3)
#
# Examples:
#   ./run_all.sh                              # default: 120s, 4 threads
#   ./run_all.sh 180 8                        # 180s, 8 threads
#   TI_INTERVAL=30 TI_BIG_M=600 ./run_all.sh  # TI với params nhỏ
#   COST_TYPES="finsteps123" ./run_all.sh     # chỉ 1 cost type

set -e

TIMEOUT="${1:-120}"
THREADS="${2:-4}"

COST_TYPES="${COST_TYPES:-finsteps123 infsteps180 cont}"
SOLVERS="${SOLVERS:-bigm ti}"

# Count for time estimate
N_COSTS=$(echo $COST_TYPES | wc -w)
N_SOLVERS=$(echo $SOLVERS | wc -w)
TOTAL_BATCHES=$((N_COSTS * N_SOLVERS * 3))

START_ALL=$(date +%s)
echo "==================================================="
echo "MILP Multi-Cost Multi-Variant Benchmark"
echo "==================================================="
echo "Cost types: $COST_TYPES"
echo "Solvers:    $SOLVERS"
echo "Variants:   original + addstationtime + addtracktime"
echo "Timeout:    ${TIMEOUT}s per instance"
echo "Threads:    $THREADS"
echo "Total:      $TOTAL_BATCHES batches"
if [ -n "$TI_INTERVAL" ] || [ -n "$TI_BIG_M" ]; then
    echo "TI params:  interval=${TI_INTERVAL:-10}, big_m=${TI_BIG_M:-900}"
fi
echo "==================================================="

BATCH_IDX=0
for cost_type in $COST_TYPES; do
    BATCH_IDX=$((BATCH_IDX + 1))
    echo ""
    echo "###############################################################"
    echo "# [$BATCH_IDX/$N_COSTS] COST TYPE: $cost_type"
    echo "###############################################################"

    SOLVERS="$SOLVERS" \
        ./run_3variants.sh "$cost_type" "$TIMEOUT" "$THREADS"
done

END_ALL=$(date +%s)
TOTAL_TIME=$((END_ALL - START_ALL))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "==================================================="
echo "ALL DONE in ${TOTAL_TIME}s (${MINUTES}m ${SECONDS}s)"
echo "==================================================="
echo "CSV outputs structure:"
echo "  results/<solver>/<variant>/batch_<solver>_<cost>_<timestamp>.csv"
echo ""
echo "Quick check files:"
for solver in $SOLVERS; do
    for variant in original addstationtime addtracktime; do
        DIR="results/${solver}/${variant}"
        if [ -d "$DIR" ]; then
            N_CSV=$(ls $DIR/batch_*.csv 2>/dev/null | wc -l)
            echo "  [${solver}/${variant}] $N_CSV CSV files"
        fi
    done
done
echo "==================================================="
