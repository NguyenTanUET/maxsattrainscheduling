#!/bin/bash
# CP Optimizer: chạy 3 cost × 3 variant → CSV theo schema Gurobi mip_ddd.
#
# Usage:
#   ./run_all_gurobi.sh [timeout=120] [threads=4]
#
# Env var: COST_TYPES="finsteps123 infsteps180 cont"

set -e

TIMEOUT="${1:-120}"
THREADS="${2:-4}"
COST_TYPES="${COST_TYPES:-finsteps123 infsteps180 cont}"
VARIANTS=(original addstationtime addtracktime)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START_ALL=$(date +%s)
echo "================================================="
echo "CP Gurobi-CSV Batch"
echo "Cost types: $COST_TYPES"
echo "Variants:   ${VARIANTS[*]}"
echo "Timeout:    ${TIMEOUT}s | Threads: $THREADS"
echo "================================================="

for cost in $COST_TYPES; do
    for variant in "${VARIANTS[@]}"; do
        VARIANT_DIR="../instances/$variant"
        if [ ! -d "$VARIANT_DIR" ]; then
            echo "skip: $VARIANT_DIR not found"
            continue
        fi
        echo ""
        echo "── [$cost / cp / $variant] ──"
        OUTPUT_DIR="results/${variant}" \
            "$SCRIPT_DIR/batch_gurobi.sh" "$VARIANT_DIR" "$cost" "$TIMEOUT" "$THREADS"
    done
done

END_ALL=$(date +%s)
echo ""
echo "================================================="
echo "ALL DONE in $((END_ALL - START_ALL))s"
echo "CSV outputs: results/<variant>/batch_cp_<cost>_*.csv"
echo "================================================="
