#!/usr/bin/env bash
# Re-run all 3 SC-affected method/config combinations with the new
# PAIRWISE_AMO_MAX_SIZE = 10 (changed from 5). Only the Default configs
# need re-running, since Nothing configs have SC AMO disabled
# (use_sc_amo = false) and are unaffected by the threshold change.
#
# This script:
#   1. Rebuilds the binary (must succeed before benches run).
#   2. Runs 3 Default benches × 3 objectives = 9 runs total.
#   3. Saves outputs to *_n10/ dirs so old n=5 results are preserved.
#
# Configs (1 per method, Default only):
#   MaxsatScDefault → maxsat_verify_n10/
#   IncTotDefault    → inc_totalizer_verify_n10/
#   PureDefault      → puresat_verify_n10/
#
# Usage:
#   bash quick_scripts/ablation/sc_threshold10_all_methods.sh
#
# Background:
#   nohup bash quick_scripts/ablation/sc_threshold10_all_methods.sh \
#       > sc_n10_bench.log 2>&1 &
#   tail -f sc_n10_bench.log
#
# Tunables:
#   INSTANCE_TIMEOUT_SECS  per-instance wall cap (default 150s)
#   RAM_LIMIT_KB           per-instance virtual-mem cap (default 15GB)

set -euo pipefail

# ─── Step 1: rebuild binary ──────────────────────────────────────────────
echo "=========================================="
echo "[1/4] Rebuilding binary with new threshold"
echo "=========================================="
cargo build --release
BIN="${BIN:-./target/release/ddd}"
[[ -x "$BIN" ]] || { echo "ERROR: $BIN not found after build"; exit 1; }
echo "Binary OK: $BIN"
echo

BASE_OUT="${BASE_OUT:-2026-05-16-Verified-Result-For-Graduation-Thesis}"
CSV_BATCH="quick_scripts/analyze/json_to_csv_batch.py"
OBJECTIVES=(finsteps123 infsteps180 cont)

# ─── Helper: one (method, objective) combination ─────────────────────────
run_one() {
    local out_dir="$1"
    local tag="$2"
    local solver="$3"
    local obj="$4"

    mkdir -p "$out_dir"
    local json_out="${out_dir}/${tag}_${obj}.json"

    echo "------------------------------------------------------------"
    echo "[${tag} | ${obj}]   (PAIRWISE_AMO_MAX_SIZE = 10)"
    echo "  JSON:   $json_out"
    echo "  Solver: $solver"
    echo "------------------------------------------------------------"

    BIN="$BIN" \
    SOLVER="$solver" \
    OBJECTIVE="$obj" \
    JSON_OUT="$json_out" \
    INSTANCE_TIMEOUT_SECS="${INSTANCE_TIMEOUT_SECS:-150}" \
    RAM_LIMIT_KB="${RAM_LIMIT_KB:-15000000}" \
    bash quick_scripts/run_all_txt_instances_limited.sh

    python3 "$CSV_BATCH" "$json_out" --format compact --overwrite \
        && echo "  CSV ok" || echo "  (CSV failed)"
    echo
}

START_TIME=$(date +%s)
echo "=========================================="
echo "[2/4] MaxSAT-SC Default × 3 objectives"
echo "Output: $BASE_OUT/maxsat_verify_n10/"
echo "=========================================="
for obj in "${OBJECTIVES[@]}"; do
    run_one "$BASE_OUT/maxsat_verify_n10" MaxsatScDefault maxsat_ddd_ladder_sc "$obj"
done

echo "=========================================="
echo "[3/4] IncTot Default × 3 objectives"
echo "Output: $BASE_OUT/inc_totalizer_verify_n10/"
echo "=========================================="
for obj in "${OBJECTIVES[@]}"; do
    run_one "$BASE_OUT/inc_totalizer_verify_n10" IncTotDefault sat_ddd_sc_totalizer "$obj"
done

echo "=========================================="
echo "[4/4] PureSat Default × 3 objectives"
echo "Output: $BASE_OUT/puresat_verify_n10/"
echo "=========================================="
for obj in "${OBJECTIVES[@]}"; do
    run_one "$BASE_OUT/puresat_verify_n10" PureDefault sat_ddd_sc_fresh_addclauses "$obj"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "============================================================"
echo "=== All 9 runs DONE in ${ELAPSED}s ($(( ELAPSED / 60 )) min) ==="
echo "End: $(date)"
echo "Outputs:"
ls -la "$BASE_OUT/maxsat_verify_n10/" 2>/dev/null
ls -la "$BASE_OUT/inc_totalizer_verify_n10/" 2>/dev/null
ls -la "$BASE_OUT/puresat_verify_n10/" 2>/dev/null
