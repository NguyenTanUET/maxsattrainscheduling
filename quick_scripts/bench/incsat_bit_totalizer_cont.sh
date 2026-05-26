#!/usr/bin/env bash
# OOM-safe runner for `sat_ddd_sc` on cont with BitTotalizer encoding.
#
# Same SatDddSettings defaults as `IncTotDefault` — only the cost-objective
# encoding differs. Uses `sat_ddd_sc` (NOT `sat_ddd_sc_totalizer`, which
# hardcodes the IncrementalTotalizer encoding and can't be overridden).
#
# Settings (all defaults from `SatDddSettings::default()`):
#   use_precedence_graph: true
#   prealloc_cost_thresholds:      false
#   seed_precedence_from_earliest: false
#   seed_resource_conflicts:       false
#   use_sc_amo:                   true
# Encoding override:
#   --satddd-objective-encoding bit_totalizer
#
# Output: `2026-05-16-Verified-Result-For-Graduation-Thesis/
#          inc_totalizer_cont_verify/IncBitTot_cont.json` + matching CSV,
# next to `IncTotDefault_cont.{json,csv}` for direct comparison.
#
# Usage:
#   bash quick_scripts/bench/incsat_bit_totalizer_cont.sh
#
# Tunables (env vars, all optional):
#   INSTANCE_TIMEOUT_SECS  per-instance wall-clock cap (default 150s)
#   RAM_LIMIT_KB           per-instance virtual-memory cap (default 15GB)

set -euo pipefail

OUT_DIR="${OUT_DIR:-2026-05-16-Verified-Result-For-Graduation-Thesis/inc_totalizer_cont_verify}"
mkdir -p "$OUT_DIR"

JSON_OUT="${OUT_DIR}/IncBitTot_cont.json"
CSV_BATCH="quick_scripts/analyze/json_to_csv_batch.py"

echo "=== inc bit_totalizer cont (OOM-safe per-instance isolation) ==="
echo "Output JSON: $JSON_OUT"
echo

# Note: sat_ddd_sc + SATDDD_OBJECTIVE_ENCODING=bit_totalizer.
# The wrapper auto-switches `sat_ddd_sc_totalizer` → `sat_ddd_sc` when
# encoding is set, but we pass `sat_ddd_sc` explicitly for clarity.
BIN="${BIN:-./target/release/ddd}" \
SOLVER=sat_ddd_sc \
OBJECTIVE=cont \
SATDDD_OBJECTIVE_ENCODING=bit_totalizer \
JSON_OUT="$JSON_OUT" \
INSTANCE_TIMEOUT_SECS="${INSTANCE_TIMEOUT_SECS:-150}" \
RAM_LIMIT_KB="${RAM_LIMIT_KB:-15000000}" \
bash quick_scripts/run_all_txt_instances_limited.sh

echo
python3 "$CSV_BATCH" "$JSON_OUT" --format compact --overwrite \
    && echo "CSV ok: ${JSON_OUT%.json}.csv" \
    || echo "(CSV failed)"

echo
echo "=== Done. Output: $JSON_OUT ==="
