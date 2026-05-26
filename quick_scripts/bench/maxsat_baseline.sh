#!/usr/bin/env bash
# Run maxsat_ddd_ladder (baseline — Croella2024 implementation, no SC)
# across all 3 objectives. Mirrors the layout of
# run_maxsat_default_all_objectives.sh so the baseline lives in a parallel
# verify dir next to the SC run.
#
# Config (1):
#   MaxsatBaseline — maxsat_ddd_ladder with CLI defaults
#                    (pure Croella2024 ladder encoding; 4-literal pairwise
#                     AMO for resource cliques; no precedence preprocessing).
#
# Objectives (3): finsteps123, infsteps180, cont.
# Total: 3 runs. Each instance isolated via run_all_txt_instances_limited.sh
# so OOM/timeout on one instance doesn't kill the batch.
#
# Output: 2026-05-16-Verified-Result-For-Graduation-Thesis/maxsat_baseline/
#         MaxsatBaseline_{finsteps123,infsteps180,cont}.{json,csv}
#
# Usage:
#   bash quick_scripts/bench/maxsat_baseline.sh
#
# Background usage (recommended while editing thesis):
#   nohup bash quick_scripts/bench/maxsat_baseline.sh \
#       > maxsat_baseline_bench.log 2>&1 &
#   tail -f maxsat_baseline_bench.log    # check progress
#
# Tunables (env vars):
#   INSTANCE_TIMEOUT_SECS  per-instance wall cap (default 150s)
#   RAM_LIMIT_KB           per-instance virtual-mem cap (default 15GB)

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OUT_DIR="${OUT_DIR:-2026-05-16-Verified-Result-For-Graduation-Thesis/maxsat_baseline}"
CSV_BATCH="quick_scripts/analyze/json_to_csv_batch.py"

mkdir -p "$OUT_DIR"

OBJECTIVES=(finsteps123 infsteps180 cont)

run_config() {
    local tag="$1"; shift
    local obj="$1"; shift

    local json_out="${OUT_DIR}/${tag}_${obj}.json"

    echo "============================================================"
    echo "[${tag} | ${obj}]"
    echo "  JSON: $json_out"
    echo "  Solver: maxsat_ddd_ladder (Croella2024 baseline, CLI defaults)"
    echo "============================================================"

    BIN="$BIN" \
    SOLVER=maxsat_ddd_ladder \
    OBJECTIVE="$obj" \
    JSON_OUT="$json_out" \
    INSTANCE_TIMEOUT_SECS="${INSTANCE_TIMEOUT_SECS:-150}" \
    RAM_LIMIT_KB="${RAM_LIMIT_KB:-15000000}" \
    bash quick_scripts/run_all_txt_instances_limited.sh

    python3 "$CSV_BATCH" "$json_out" --format compact --overwrite \
        && echo "  CSV ok" || echo "  (CSV failed)"
    echo
}

START_TIME=$(date +%s)
echo "=== MaxSAT-Ladder baseline bench: 1 config × 3 objectives (3 runs) ==="
echo "Start: $(date)"
echo "Output dir: $OUT_DIR"
echo

for obj in "${OBJECTIVES[@]}"; do
    # MaxsatBaseline — no overrides; baseline ladder solver (Croella2024).
    run_config MaxsatBaseline "$obj"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "============================================================"
echo "=== All 3 runs DONE in ${ELAPSED}s ($(( ELAPSED / 60 )) min) ==="
echo "End: $(date)"
echo "Outputs: $OUT_DIR/"
ls -la "$OUT_DIR/"
