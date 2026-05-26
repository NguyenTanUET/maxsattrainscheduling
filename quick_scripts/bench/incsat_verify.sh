#!/usr/bin/env bash
# Verify the post-fix behaviour of `incremental_sat.rs` after gating its 3
# init-time pre-allocation patterns behind `SatDddSettings` flags:
#   - prealloc_cost_thresholds      (was const SEED_COST_THRESHOLDS = true)
#   - seed_precedence_from_earliest (was const SEED_PRECEDENCE_FROM_EARLIEST = true)
#   - seed_resource_conflicts       (was const SEED_RESOURCE_CONFLICTS = true)
#
# Compares:
#   IncDefault  new default — all 3 seeds OFF (lazy, matches maxsatddd_ladder)
#   IncLegacy   replicate previous behaviour — all 3 seeds ON
#
# Across 3 objectives (finsteps123, infsteps180, cont). Total: 6 runs.
#
# Usage:
#   bash quick_scripts/bench/incsat_verify.sh

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OUT_DIR="${OUT_DIR:-2026-05-16-Verified-Result-For-Graduation-Thesis/incremental_sat_verify}"
CSV_BATCH="quick_scripts/analyze/json_to_csv_batch.py"

mkdir -p "$OUT_DIR"

OBJECTIVES=(finsteps123 infsteps180 cont)

run() {
    local tag="$1"; shift
    local obj="$1"; shift
    local out="${OUT_DIR}/${tag}_${obj}.json"
    echo "--- [${tag} | ${obj}] ---"
    "$BIN" "$@" \
      --txt-instances \
      --objective "$obj" \
      --json-output "$out"
    python3 "$CSV_BATCH" "$out" --format compact --overwrite \
        && echo "  CSV ok" || echo "  (CSV failed)"
    echo
}

echo "=== incremental_sat verify: 2 configs × 3 objectives ==="
echo "Output dir: $OUT_DIR"
echo "Binary:     $BIN"
echo

for obj in "${OBJECTIVES[@]}"; do
    # New default: all 3 seeds OFF — lazy, formula starts small.
    run IncDefault "$obj" \
      -s sat_ddd_inc

    # Legacy: replicate pre-fix behaviour with all 3 seeds ON.
    run IncLegacy "$obj" \
      -s sat_ddd_inc \
      --satddd-prealloc-cost-thresholds true \
      --satddd-seed-precedence-from-earliest true \
      --satddd-seed-resource-conflicts true
done

echo "=== Done. Outputs in $OUT_DIR ==="
