#!/usr/bin/env bash
# Isolate which SC feature (use_sc chain encoding vs use_interval_graph
# conflict encoding) causes the slowdown.
#
# Starting from B (the best config from previous comparison: prec_graph ON,
# sc OFF, interval-graph OFF), turn each SC feature back on individually:
#
#   B: prec=on, sc=off, intgr=off   (baseline, already in results_B_*.json)
#   D: prec=on, sc=ON,  intgr=off
#   E: prec=on, sc=off, intgr=ON
#   C: prec=on, sc=ON,  intgr=ON    (= default full SC, already in results_C_*.json)
#
# By comparing D/B and E/B we see which feature alone hurts (or helps).
#
# Usage:
#   bash quick_scripts/ablation/sc_isolate.sh
#   OBJECTIVE=finsteps123 bash quick_scripts/ablation/sc_isolate.sh

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OBJECTIVE="${OBJECTIVE:-infsteps180}"
OUT_PREFIX="${OUT_PREFIX:-results}"

d_json="${OUT_PREFIX}_D_${OBJECTIVE}.json"
e_json="${OUT_PREFIX}_E_${OBJECTIVE}.json"

echo "=== SC feature isolation on objective=${OBJECTIVE} ==="
echo "Existing baseline: results_B_${OBJECTIVE}.json (prec ON, sc OFF, intgr OFF)"
echo "Existing default:  results_C_${OBJECTIVE}.json (prec ON, sc ON,  intgr ON)"
echo

echo "--- [D] prec ON + sc ON  + intgr OFF ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph true \
  --maxsat-ladder-sc-use-eager-chain-expansion true \
  --maxsat-ladder-sc-use-interval-graph false \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$d_json"

echo
echo "--- [E] prec ON + sc OFF + intgr ON ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph true \
  --maxsat-ladder-sc-use-eager-chain-expansion false \
  --maxsat-ladder-sc-use-interval-graph true \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$e_json"

echo
echo "=== Done. Outputs: ==="
echo "  D: $d_json"
echo "  E: $e_json"
echo
echo "Compare with existing:"
echo "  B: results_B_${OBJECTIVE}.json  (prec only)"
echo "  C: results_C_${OBJECTIVE}.json  (full SC)"
