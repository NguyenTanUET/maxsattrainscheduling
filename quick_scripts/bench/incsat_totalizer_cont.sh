#!/usr/bin/env bash
# OOM-safe runner for `sat_ddd_sc_totalizer` on cont with current defaults.
#
# Uses `run_all_txt_instances_limited.sh` under the hood so each instance
# runs in its OWN child process with `timeout` + `ulimit -Sv`. If one
# instance OOMs / times out the rest still complete.
#
# Defaults align with `SatDddSettings::default()` post-fix:
#   use_precedence_graph: true
#   prealloc_cost_thresholds:      false
#   seed_precedence_from_earliest: false
#   seed_resource_conflicts:       false
#   use_sc_amo:                   true
#
# Output: `2026-05-16-Verified-Result-For-Graduation-Thesis/
#          inc_totalizer_cont_verify/IncTotDefault_cont.json` + matching CSV.
#
# Usage:
#   bash quick_scripts/bench/incsat_totalizer_cont.sh
#
# Tunables (env vars, all optional):
#   INSTANCE_TIMEOUT_SECS  per-instance wall-clock cap (default 150s)
#   RAM_LIMIT_KB           per-instance virtual-memory cap (default 15GB)

set -euo pipefail

OUT_DIR="${OUT_DIR:-2026-05-16-Verified-Result-For-Graduation-Thesis/inc_totalizer_cont_verify}"
mkdir -p "$OUT_DIR"

JSON_OUT="${OUT_DIR}/IncTotDefault_cont.json"
CSV_BATCH="quick_scripts/analyze/json_to_csv_batch.py"

echo "=== inc_totalizer cont (default, OOM-safe per-instance isolation) ==="
echo "Output JSON: $JSON_OUT"
echo

# Let the binary's defaults stand for the new SatDddSettings fields —
# don't set any SATDDD_* env vars so `unwrap_or(default)` in main.rs fires.
BIN="${BIN:-./target/release/ddd}" \
SOLVER=sat_ddd_sc_totalizer \
OBJECTIVE=cont \
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
