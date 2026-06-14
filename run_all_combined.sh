#!/bin/bash
# Run TOÀN BỘ benchmark CPLEX: 3 methods × 3 cost types × 3 dataset variants.
# Output: 9 CSV files (mỗi (method × cost) gộp 3 variant), giống cách Gurobi
# tổ chức (e.g. results_mip_ddd_cont_120s.csv chứa tất cả 72 instance).
#
# Methods: bigm (lazy), ti, cp
# Cost types: finsteps123, infsteps180, cont
# Variants: original, addstationtime, addtracktime
#
# Usage:
#   ./run_all_combined.sh [timeout=120] [threads=4]
#
# Env vars (optional):
#   METHODS="bigm ti cp"                       chọn methods
#   COSTS="finsteps123 infsteps180 cont"       chọn cost types
#   VARIANTS="original addstationtime addtracktime"   chọn dataset
#   OUT_DIR=results_combined                   thư mục output (root)
#   TI_INTERVAL=10 TI_BIG_M=900                params cho TI
#
# Examples:
#   ./run_all_combined.sh 120 4                          # full
#   METHODS="bigm" ./run_all_combined.sh 120 4           # chỉ bigm
#   COSTS="finsteps123" ./run_all_combined.sh 60 4       # chỉ 1 cost, 60s

set -e

TIMEOUT="${1:-120}"
THREADS="${2:-4}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/results_combined/$TIMESTAMP}"
WORK_DIR="$OUT_DIR/_work"
mkdir -p "$OUT_DIR" "$WORK_DIR"

CSV_HEADER="algorithm_time,avg_tracks,conflicting_visit_pairs,conflicts,cost,delay_cost_type,index,internal_cost,intervals,iteration,lb,name,resource_constraints,sol_time,solver_name,solver_time,status,total_time,trains,travel_constraints,ub"

METHODS="${METHODS:-bigm ti cp}"
COSTS="${COSTS:-finsteps123 infsteps180 cont}"
VARIANTS="${VARIANTS:-original addstationtime addtracktime}"

# Check binaries needed.
if ! command -v cplex &> /dev/null; then
    echo "Error: cplex not in PATH. Activate CPLEX env first."
    exit 1
fi
if [ ! -x "$PROJECT_ROOT/target/release/export_lp" ]; then
    echo "Error: export_lp binary missing. Build with:"
    echo "  cd $PROJECT_ROOT && cargo build --release --bin export_lp"
    exit 1
fi
if [[ " $METHODS " == *" cp "* ]]; then
    if [ ! -x "$PROJECT_ROOT/target/release/export_opl" ]; then
        echo "Error: export_opl binary missing. Build with:"
        echo "  cd $PROJECT_ROOT && cargo build --release --bin export_opl"
        exit 1
    fi
fi

# Count combos.
N_METHODS=$(echo $METHODS | wc -w)
N_COSTS=$(echo $COSTS | wc -w)
N_VARIANTS=$(echo $VARIANTS | wc -w)
TOTAL_COMBOS=$((N_METHODS * N_COSTS))

echo "═══════════════════════════════════════════════════════════"
echo "CPLEX COMBINED BENCHMARK (Gurobi-format CSV)"
echo "═══════════════════════════════════════════════════════════"
echo "Methods:      $METHODS"
echo "Cost types:   $COSTS"
echo "Variants:     $VARIANTS"
echo "Timeout:      ${TIMEOUT}s | Threads: $THREADS"
echo "Output root:  $OUT_DIR"
echo "Combos:       $TOTAL_COMBOS files (mỗi file gộp $N_VARIANTS variant)"
echo "═══════════════════════════════════════════════════════════"

START_ALL=$(date +%s)
COMBO_IDX=0

# Helper: run 1 (method, cost), loop variants, gộp vào 1 CSV với INDEX liên tục.
run_combo() {
    local method="$1"
    local cost="$2"
    local final_csv="$OUT_DIR/results_${method}_${cost}_${TIMEOUT}s.csv"

    echo "$CSV_HEADER" > "$final_csv"
    local global_idx=0

    for variant in $VARIANTS; do
        local var_dir="$PROJECT_ROOT/instances/$variant"
        if [ ! -d "$var_dir" ]; then
            echo "  skip $variant (not found)"
            continue
        fi

        local sub_work="$WORK_DIR/${method}_${cost}_${variant}"
        mkdir -p "$sub_work"

        echo ""
        echo "── [$method / $cost / $variant] ──"
        if [ "$method" = "cp" ]; then
            OUTPUT_DIR="$sub_work" \
                "$PROJECT_ROOT/cplex_cp/batch_gurobi.sh" \
                "$var_dir" "$cost" "$TIMEOUT" "$THREADS"
        else
            OUTPUT_DIR="$sub_work" \
                "$PROJECT_ROOT/cplex_milp/batch_gurobi.sh" \
                "$var_dir" "$method" "$cost" "$TIMEOUT" "$THREADS"
        fi

        # Tìm CSV mới nhất do batch_gurobi.sh ghi.
        local sub_csv=$(ls -t "$sub_work"/batch_*.csv 2>/dev/null | head -1)
        if [ -z "$sub_csv" ] || [ ! -f "$sub_csv" ]; then
            echo "  WARNING: không có CSV từ $variant"
            continue
        fi

        # Append rows (skip header), renumber column index (7) thành global_idx tiếp.
        # CSV: 21 columns. Column 7 (1-based) = index.
        awk -F',' -v OFS=',' -v offset="$global_idx" '
            NR > 1 {
                $7 = (NR - 2) + offset
                print
            }
        ' "$sub_csv" >> "$final_csv"

        local n_rows=$(($(wc -l < "$sub_csv") - 1))
        global_idx=$((global_idx + n_rows))
    done

    local total_rows=$(($(wc -l < "$final_csv") - 1))
    echo ""
    echo "  ✓ $(basename "$final_csv") — $total_rows rows total"
}

for method in $METHODS; do
    for cost in $COSTS; do
        COMBO_IDX=$((COMBO_IDX + 1))
        echo ""
        echo "###############################################################"
        echo "# [$COMBO_IDX/$TOTAL_COMBOS] $method × $cost"
        echo "###############################################################"
        run_combo "$method" "$cost"
    done
done

END_ALL=$(date +%s)
TOTAL_TIME=$((END_ALL - START_ALL))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "ALL DONE in ${TOTAL_TIME}s (${MINUTES}m ${SECONDS}s)"
echo "═══════════════════════════════════════════════════════════"
echo "Output CSVs:"
ls -lh "$OUT_DIR"/results_*.csv 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
echo ""
echo "Intermediate files in $WORK_DIR (có thể xoá nếu không cần debug)"
echo "═══════════════════════════════════════════════════════════"
