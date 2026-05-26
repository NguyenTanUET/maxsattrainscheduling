#!/usr/bin/env bash
# Verify the post-fix behaviour of `maxsatddd_ladder_sc` after:
#   - gating cost-threshold pre-alloc behind --prealloc-cost-thresholds (default OFF)
#   - reverting n_assumps init from `soft_constraints.len()` back to `20`
#     (matches `maxsatddd_ladder` incremental WPMaxSAT strategy)
#
# Only re-runs ScDefault and ScNothing — the `Ldr_*` JSONs/CSVs from the
# previous run are reused as the baseline. Output overwrites the existing
# Sc* files in the same directory so the analysis script still finds
# Ldr_*, ScDefault_*, ScNothing_* side by side.
#
# Configs:
#   ScDefault   Option B (new default after fixes)
#   ScNothing   all SC feature flags OFF — should now match Ldr
#
# Across all 3 objectives (finsteps123, infsteps180, cont). Total: 6 runs.
#
# Usage:
#   bash quick_scripts/bench/post_fix_verify.sh

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OUT_DIR="${OUT_DIR:-2026-05-16-Verified-Result-For-Graduation-Thesis/post_fix_verify}"
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

echo "=== post-fix verify: 2 configs × 3 objectives (Ldr baseline reused) ==="
echo "Output dir: $OUT_DIR"
echo "Binary:     $BIN"
echo

for obj in "${OBJECTIVES[@]}"; do
    # New default Option B — all explicit flags omitted, so defaults from
    # MaxSatDddLadderScSettings::default() are used. After the patch:
    #   prealloc_cost_thresholds=false, seed_sc_from_earliest=false,
    #   n_assumps starts at 20 (lazy/incremental MaxSAT)
    run ScDefault "$obj" \
      -s maxsat_ddd_ladder_sc

    # All SC feature flags OFF — should now match Ldr after the patches.
    run ScNothing "$obj" \
      -s maxsat_ddd_ladder_sc \
      --maxsat-ladder-sc-use-precedence-graph false \
      --maxsat-ladder-sc-use-eager-chain-expansion false \
      --maxsat-ladder-sc-use-interval-graph false \
      --maxsat-ladder-sc-use-touched-clique-amo false \
      --maxsat-ladder-sc-use-sc-amo false \
      --maxsat-ladder-sc-use-scamo false \
      --maxsat-ladder-sc-seed-from-earliest false \
      --maxsat-ladder-sc-prealloc-cost-thresholds false
done

echo "=== Done. Outputs in $OUT_DIR (Ldr_*.{json,csv} reused) ==="
