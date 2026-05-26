#!/usr/bin/env bash
# Isolate which SC feature causes the 2x slowdown on continuous objective.
# Run 4 configs of maxsat_ddd_ladder_sc + 1 baseline ladder, all on `cont`.
#
#   Ldr            baseline (maxsat_ddd_ladder)
#   ScDefault     sc current default = Option B (prec + touched-amo + sc-amo)
#   ScNoTouched   sc - touched-clique-amo only
#   ScNoPrec      sc - precedence-graph only
#   ScNothing     sc with all SC flags OFF (= sc_as_ladder)
#
# Comparing these reveals which flag(s) cause the cont regression.
#
# Usage:
#   bash quick_scripts/ablation/cont_isolate.sh

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OBJECTIVE=cont
OUT_DIR="${OUT_DIR:-2026-05-16-Verified-Result-For-Graduation-Thesis/cont_isolate}"
CSV_BATCH="quick_scripts/analyze/json_to_csv_batch.py"

mkdir -p "$OUT_DIR"

run() {
    local tag="$1"; shift
    local out="${OUT_DIR}/${tag}_${OBJECTIVE}.json"
    echo "--- [${tag}] ---"
    "$BIN" "$@" \
      --txt-instances \
      --objective "$OBJECTIVE" \
      --json-output "$out"
    python3 "$CSV_BATCH" "$out" --format compact --overwrite \
        && echo "  CSV ok" || echo "  (CSV failed)"
    echo
}

echo "=== cont-isolation: 5 configs ==="
echo

run Ldr \
  -s maxsat_ddd_ladder

run ScDefault \
  -s maxsat_ddd_ladder_sc

run ScNoTouched \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-touched-clique-amo false

run ScNoPrec \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph false

run ScNothing \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph false \
  --maxsat-ladder-sc-use-eager-chain-expansion false \
  --maxsat-ladder-sc-use-interval-graph false \
  --maxsat-ladder-sc-use-touched-clique-amo false

echo "=== Done. Outputs in $OUT_DIR ==="
