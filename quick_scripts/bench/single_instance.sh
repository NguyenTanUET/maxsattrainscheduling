#!/usr/bin/env bash
# Run a SINGLE instance through both `maxsat_ddd_ladder` (baseline) and
# `maxsat_ddd_ladder_sc` (current defaults = Option B: prec+touched-clique-
# AMO+sc-amo, eager-chain-expansion + interval-graph OFF). Use this to
# sanity-check a specific instance's behaviour (cost, ub/lb on timeout, etc).
#
# Default instance: trackA11 (one that timed out previously on infsteps180).
#
# Usage:
#   bash quick_scripts/bench/single_instance.sh
#   INSTANCE=stationA11   OBJECTIVE=infsteps180 bash quick_scripts/bench/single_instance.sh
#   INSTANCE=origA1      OBJECTIVE=finsteps123 bash quick_scripts/bench/single_instance.sh

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OBJECTIVE="${OBJECTIVE:-infsteps180}"
INSTANCE="${INSTANCE:-trackA11}"

ldr_json="single_${INSTANCE}_Ldr_${OBJECTIVE}.json"
sc_json="single_${INSTANCE}_Sc_${OBJECTIVE}.json"

echo "=== Single-instance test: ${INSTANCE} @ ${OBJECTIVE} ==="
echo

echo "--- [Ldr] maxsat_ddd_ladder ---"
"$BIN" \
  -s maxsat_ddd_ladder \
  --txt-instances \
  --instance-name-filter "$INSTANCE" \
  --instance-name-exact \
  --objective "$OBJECTIVE" \
  --json-output "$ldr_json"

echo
echo "--- [Sc] maxsat_ddd_ladder_sc (default = Option B) ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --txt-instances \
  --instance-name-filter "$INSTANCE" \
  --instance-name-exact \
  --objective "$OBJECTIVE" \
  --json-output "$sc_json"

echo
echo "=== Outputs ==="
echo "  Ldr: $ldr_json"
echo "  Sc: $sc_json"
echo

echo "--- Quick summary ---"
python3 - <<EOF
import json
for tag, path in [("Ldr", "$ldr_json"), ("Sc", "$sc_json")]:
    with open(path) as f:
        data = json.load(f)
    if not data:
        print(f"{tag}: NO RESULT (filter matched no instance?)")
        continue
    s = data[0]["solves"][0]
    print(f"{tag}: status={s.get('status')} cost={s.get('cost')} "
          f"ub={s.get('ub')} lb={s.get('lb')} "
          f"time={s.get('sol_time'):.1f}ms iters={s.get('iterations')}")
EOF
