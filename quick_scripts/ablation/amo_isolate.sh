#!/usr/bin/env bash
# Isolate the effect of SC AMO encoding alone.
#
# All 3 configs share these flags (so that the only difference is
# whether `add_sc_amo` fires for large cliques):
#   --use-precedence-graph        false
#   --use-eager-chain-expansion   false
#   --use-interval-graph          true  (clique-cover conflict encoding)
#
# Then `--use-sc-amo` is toggled to isolate the SC vs pairwise AMO
# choice for cliques larger than PAIRWISE_AMO_MAX_SIZE.
#
#   Ldr:        `maxsat_ddd_ladder` (original codebase, no SC).
#   ScAmoOn:   maxsat_ddd_ladder_sc, --use-sc-amo=true  (SC for big cliques).
#   ScAmoOff:  maxsat_ddd_ladder_sc, --use-sc-amo=false (always pairwise).
#
# Usage:
#   bash quick_scripts/ablation/amo_isolate.sh
#   OBJECTIVE=finsteps123 bash quick_scripts/ablation/amo_isolate.sh

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OBJECTIVE="${OBJECTIVE:-infsteps180}"
OUT_PREFIX="${OUT_PREFIX:-results_amoiso}"

ldr_json="${OUT_PREFIX}_Ldr_${OBJECTIVE}.json"
on_json="${OUT_PREFIX}_ScAmoOn_${OBJECTIVE}.json"
off_json="${OUT_PREFIX}_ScAmoOff_${OBJECTIVE}.json"

echo "=== AMO-isolation compare on objective=${OBJECTIVE} ==="
echo

echo "--- [Ldr] maxsat_ddd_ladder (baseline) ---"
"$BIN" \
  -s maxsat_ddd_ladder \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$ldr_json"

echo
echo "--- [ScAmoOn] sc, interval-graph + SC AMO for large cliques ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph false \
  --maxsat-ladder-sc-use-eager-chain-expansion false \
  --maxsat-ladder-sc-use-interval-graph true \
  --maxsat-ladder-sc-use-sc-amo true \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$on_json"

echo
echo "--- [ScAmoOff] sc, interval-graph + pairwise AMO (NO SC) ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph false \
  --maxsat-ladder-sc-use-eager-chain-expansion false \
  --maxsat-ladder-sc-use-interval-graph true \
  --maxsat-ladder-sc-use-sc-amo false \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$off_json"

echo
echo "=== Done. Outputs: ==="
echo "  Ldr:        $ldr_json"
echo "  ScAmoOn:   $on_json"
echo "  ScAmoOff:  $off_json"
