#!/bin/bash
# MILP: chạy BigM + TI × 3 cost × 3 variant → CSV theo schema Gurobi mip_ddd.
#
# Usage:
#   ./run_all_gurobi.sh [timeout=120] [threads=4]
#
# Env vars:
#   TI_INTERVAL, TI_BIG_M       params cho TI (default 10/900, khớp Gurobi)
#   SOLVERS="bigm ti"           default cả 2
#   COST_TYPES="finsteps123 infsteps180 cont"  default cả 3

set -e

TIMEOUT="${1:-120}"
THREADS="${2:-4}"
COST_TYPES="${COST_TYPES:-finsteps123 infsteps180 cont}"
SOLVERS="${SOLVERS:-bigm ti}"
VARIANTS=(original addstationtime addtracktime)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START_ALL=$(date +%s)
echo "================================================="
echo "MILP Gurobi-CSV Batch"
echo "Cost types: $COST_TYPES"
echo "Solvers:    $SOLVERS"
echo "Variants:   ${VARIANTS[*]}"
echo "Timeout:    ${TIMEOUT}s | Threads: $THREADS"
echo "================================================="

for cost in $COST_TYPES; do
    for solver in $SOLVERS; do
        for variant in "${VARIANTS[@]}"; do
            VARIANT_DIR="../instances/$variant"
            if [ ! -d "$VARIANT_DIR" ]; then
                echo "skip: $VARIANT_DIR not found"
                continue
            fi
            echo ""
            echo "── [$cost / $solver / $variant] ──"
            OUTPUT_DIR="results/${solver}/${variant}" \
                "$SCRIPT_DIR/batch_gurobi.sh" "$VARIANT_DIR" "$solver" "$cost" "$TIMEOUT" "$THREADS"
        done
    done
done

END_ALL=$(date +%s)
echo ""
echo "================================================="
echo "ALL DONE in $((END_ALL - START_ALL))s"
echo "CSV outputs: results/<solver>/<variant>/batch_<solver>_<cost>_*.csv"
echo "================================================="
