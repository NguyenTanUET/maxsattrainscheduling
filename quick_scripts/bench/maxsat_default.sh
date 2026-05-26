#!/usr/bin/env bash
# Run maxsat_ddd_ladder_sc with default settings across all 3 objectives —
# mirrors the puresat / sat-incremental verify runs so we have a comparable
# maxsat baseline saved alongside them.
#
# Config (1):
#   MaxsatScDefault — all MaxSatDddLadderScSettings at default:
#                        use_precedence_graph:           true
#                        use_eager_chain_expansion:      false
#                        use_interval_graph_conflicts:   false
#                        use_sc_amo:                    true
#                        use_touched_clique_amo:         true
#                        seed_sc_from_earliest:         false
#                        use_scamo_encoding:             false (experimental)
#                        prealloc_cost_thresholds:       false
#
# Objectives (3): finsteps123, infsteps180, cont.
# Total: 3 runs. Each instance isolated via run_all_txt_instances_limited.sh
# so OOM/timeout on one instance doesn't kill the batch.
#
# Output: 2026-05-16-Verified-Result-For-Graduation-Thesis/maxsatddd_sc_verify/
#         MaxsatScDefault_{finsteps123,infsteps180,cont}.{json,csv}
#
# Usage:
#   bash quick_scripts/bench/maxsat_default.sh
#
# Background usage (recommended while editing thesis):
#   nohup bash quick_scripts/bench/maxsat_default.sh \
#       > maxsat_bench.log 2>&1 &
#   tail -f maxsat_bench.log    # check progress
#
# Tunables (env vars):
#   INSTANCE_TIMEOUT_SECS  per-instance wall cap (default 150s)
#   RAM_LIMIT_KB           per-instance virtual-mem cap (default 15GB)

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OUT_DIR="${OUT_DIR:-2026-05-16-Verified-Result-For-Graduation-Thesis/maxsatddd_sc_verify}"
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
    echo "  Solver: maxsat_ddd_ladder_sc (CLI defaults)"
    echo "============================================================"

    BIN="$BIN" \
    SOLVER=maxsat_ddd_ladder_sc \
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
echo "=== MaxSAT-SC bench: 1 config × 3 objectives (3 runs) ==="
echo "Start: $(date)"
echo "Output dir: $OUT_DIR"
echo

for obj in "${OBJECTIVES[@]}"; do
    # MaxsatScDefault — no overrides, binary uses MaxSatDddLadderScSettings::default().
    run_config MaxsatScDefault "$obj"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "============================================================"
echo "=== All 3 runs DONE in ${ELAPSED}s ($(( ELAPSED / 60 )) min) ==="
echo "End: $(date)"
echo "Outputs: $OUT_DIR/"
ls -la "$OUT_DIR/"
