#!/usr/bin/env bash
# Compare 4 configurations on a chosen objective (default: infsteps180).
#
# Usage:
#   bash quick_scripts/ablation/sc_config_compare.sh
#       → defaults: OBJECTIVE=infsteps180, output prefix = results
#
#   OBJECTIVE=finsteps123 bash quick_scripts/ablation/sc_config_compare.sh
#   OUT_PREFIX=results_v2 bash quick_scripts/ablation/sc_config_compare.sh
#
# Outputs (in cwd):
#   ${OUT_PREFIX}_Ldr_${OBJECTIVE}.json    — maxsat_ddd_ladder (no preprocessing)
#   ${OUT_PREFIX}_A_${OBJECTIVE}.json      — sc_as_ladder, NO prec_graph
#   ${OUT_PREFIX}_B_${OBJECTIVE}.json      — sc_as_ladder, WITH prec_graph
#   ${OUT_PREFIX}_C_${OBJECTIVE}.json      — full SC (default config)

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OBJECTIVE="${OBJECTIVE:-infsteps180}"
OUT_PREFIX="${OUT_PREFIX:-results}"

ldr_json="${OUT_PREFIX}_Ldr_${OBJECTIVE}.json"
a_json="${OUT_PREFIX}_A_${OBJECTIVE}.json"
b_json="${OUT_PREFIX}_B_${OBJECTIVE}.json"
c_json="${OUT_PREFIX}_C_${OBJECTIVE}.json"

echo "=== Config compare on objective=${OBJECTIVE} ==="
echo

echo "--- [Ldr] maxsat_ddd_ladder (no preprocessing, original codebase) ---"
"$BIN" \
  -s maxsat_ddd_ladder \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$ldr_json"

echo
echo "--- [A] sc_as_ladder (all SC flags OFF, no prec_graph) ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph false \
  --maxsat-ladder-sc-use-eager-chain-expansion false \
  --maxsat-ladder-sc-use-interval-graph false \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$a_json"

echo
echo "--- [B] sc_as_ladder + prec_graph (isolate prec_graph effect) ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph true \
  --maxsat-ladder-sc-use-eager-chain-expansion false \
  --maxsat-ladder-sc-use-interval-graph false \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$b_json"

echo
echo "--- [C] full SC features (default config) ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$c_json"

echo
echo "=== Done. Outputs: ==="
echo "  Ldr:  $ldr_json"
echo "  A:    $a_json"
echo "  B:    $b_json"
echo "  C:    $c_json"
