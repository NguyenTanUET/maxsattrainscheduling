#!/usr/bin/env bash
# Compare 3 configs to isolate the effect of touched-clique SC AMO:
#
#   Ldr:           `maxsat_ddd_ladder` baseline (original codebase, no SC).
#   Nothing:       `maxsat_ddd_ladder_sc` with ALL SC knobs OFF
#                  (= sc_as_ladder mode, pure pair-based, no AMO).
#   TouchedAmo:    `maxsat_ddd_ladder_sc` with ONLY touched-clique AMO on
#                  (interval-graph still OFF, but pair scan accumulates
#                  cliques and encodes SC AMO over each size ≥ 3 clique).
#
# All `maxsat_ddd_ladder_sc` runs share these flags so the only diff
# between Nothing and TouchedAmo is the touched-clique-AMO behaviour:
#   --use-precedence-graph        false
#   --use-eager-chain-expansion   false
#   --use-interval-graph          false
#   --use-sc-amo                 true   (only matters when AMO fires)
#
# Usage:
#   bash quick_scripts/ablation/touched_amo_compare.sh
#   OBJECTIVE=finsteps123 bash quick_scripts/ablation/touched_amo_compare.sh

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OBJECTIVE="${OBJECTIVE:-infsteps180}"
OUT_PREFIX="${OUT_PREFIX:-results_tamo}"

ldr_json="${OUT_PREFIX}_Ldr_${OBJECTIVE}.json"
nothing_json="${OUT_PREFIX}_Nothing_${OBJECTIVE}.json"
touched_json="${OUT_PREFIX}_TouchedAmo_${OBJECTIVE}.json"

echo "=== Touched-AMO compare on objective=${OBJECTIVE} ==="
echo

echo "--- [Ldr] maxsat_ddd_ladder (baseline) ---"
"$BIN" \
  -s maxsat_ddd_ladder \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$ldr_json"

echo
echo "--- [Nothing] sc, all SC knobs OFF (= sc_as_ladder, pure pair-based) ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph false \
  --maxsat-ladder-sc-use-eager-chain-expansion false \
  --maxsat-ladder-sc-use-interval-graph false \
  --maxsat-ladder-sc-use-touched-clique-amo false \
  --maxsat-ladder-sc-use-sc-amo true \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$nothing_json"

echo
echo "--- [TouchedAmo] sc, pair-based + lite clique AMO with SC encoding ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph false \
  --maxsat-ladder-sc-use-eager-chain-expansion false \
  --maxsat-ladder-sc-use-interval-graph false \
  --maxsat-ladder-sc-use-touched-clique-amo true \
  --maxsat-ladder-sc-use-sc-amo true \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$touched_json"

echo
echo "=== Done. Outputs: ==="
echo "  Ldr:         $ldr_json"
echo "  Nothing:     $nothing_json"
echo "  TouchedAmo:  $touched_json"
