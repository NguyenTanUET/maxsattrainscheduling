#!/usr/bin/env bash
# 3-way comparison:
#
#   Ldr      = `maxsat_ddd_ladder` baseline (no SC features, original codebase).
#   ScOnly  = `maxsat_ddd_ladder_sc` with ONLY the SC AMO encoding active
#              (`use_interval_graph_conflicts = true`, which routes large
#              conflict cliques through `add_sc_amo` — the Truong/Kieu/To
#              SC AMO encoding). Other SC knobs OFF.
#   Nothing  = `maxsat_ddd_ladder_sc` with ALL SC knobs OFF (= sc_as_ladder
#              mode). This isolates the codebase difference vs `Ldr`.
#
# Usage:
#   bash quick_scripts/ablation/3way_sc_compare.sh
#   OBJECTIVE=finsteps123 bash quick_scripts/ablation/3way_sc_compare.sh
#
# Output prefix overridable via OUT_PREFIX.

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OBJECTIVE="${OBJECTIVE:-infsteps180}"
OUT_PREFIX="${OUT_PREFIX:-results_3way}"

ldr_json="${OUT_PREFIX}_Ldr_${OBJECTIVE}.json"
sconly_json="${OUT_PREFIX}_ScOnly_${OBJECTIVE}.json"
nothing_json="${OUT_PREFIX}_Nothing_${OBJECTIVE}.json"

echo "=== 3-way SC compare on objective=${OBJECTIVE} ==="
echo

echo "--- [Ldr] maxsat_ddd_ladder (original codebase, no preprocessing) ---"
"$BIN" \
  -s maxsat_ddd_ladder \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$ldr_json"

echo
echo "--- [ScOnly] maxsat_ddd_ladder_sc, only SC AMO encoding active ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph false \
  --maxsat-ladder-sc-use-eager-chain-expansion false \
  --maxsat-ladder-sc-use-interval-graph true \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$sconly_json"

echo
echo "--- [Nothing] maxsat_ddd_ladder_sc, all SC knobs OFF (sc_as_ladder) ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph false \
  --maxsat-ladder-sc-use-eager-chain-expansion false \
  --maxsat-ladder-sc-use-interval-graph false \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$nothing_json"

echo
echo "=== Done. Outputs: ==="
echo "  Ldr:      $ldr_json"
echo "  ScOnly:  $sconly_json"
echo "  Nothing:  $nothing_json"
