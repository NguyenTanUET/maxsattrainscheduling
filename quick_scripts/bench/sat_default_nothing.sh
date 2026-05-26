#!/usr/bin/env bash
# Run incremental-SAT (sat_ddd_sc_totalizer) with two configurations across
# all 3 objectives — designed to be left running in the background while
# the thesis document is being edited.
#
# Configs (2):
#   SatDefault   — all SatDddSettings at default (post-fix):
#                    use_precedence_graph: true
#                    use_sc_amo:                   true
#                    prealloc_cost_thresholds:      false
#                    seed_precedence_from_earliest: false
#                    seed_resource_conflicts:       false
#   SatNothing   — strip everything to closest analog of ScNothing:
#                    use_precedence_graph: false
#                    use_sc_amo:                   false
#                    (3 seed flags already default false)
#
# Objectives (3): finsteps123, infsteps180, cont.
# Total: 6 runs. Each instance isolated via run_all_txt_instances_limited.sh
# so OOM/timeout on one instance doesn't kill the batch.
#
# Output: 2026-05-16-Verified-Result-For-Graduation-Thesis/sat_verify/
#         {SatDefault,SatNothing}_{finsteps123,infsteps180,cont}.{json,csv}
#
# Usage:
#   bash quick_scripts/bench/sat_default_nothing.sh
#
# Background usage (recommended while editing thesis):
#   nohup bash quick_scripts/bench/sat_default_nothing.sh \
#       > sat_bench.log 2>&1 &
#   tail -f sat_bench.log    # check progress
#
# Tunables (env vars):
#   INSTANCE_TIMEOUT_SECS  per-instance wall cap (default 150s)
#   RAM_LIMIT_KB           per-instance virtual-mem cap (default 15GB)

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OUT_DIR="${OUT_DIR:-2026-05-16-Verified-Result-For-Graduation-Thesis/sat_verify}"
CSV_BATCH="quick_scripts/analyze/json_to_csv_batch.py"

mkdir -p "$OUT_DIR"

OBJECTIVES=(finsteps123 infsteps180 cont)

run_config() {
    local tag="$1"; shift
    local obj="$1"; shift
    local extra_env=("$@")

    local json_out="${OUT_DIR}/${tag}_${obj}.json"

    echo "============================================================"
    echo "[${tag} | ${obj}]"
    echo "  JSON: $json_out"
    echo "  Env:  ${extra_env[*]:-(default)}"
    echo "============================================================"

    BIN="$BIN" \
    SOLVER=sat_ddd_sc_totalizer \
    OBJECTIVE="$obj" \
    JSON_OUT="$json_out" \
    INSTANCE_TIMEOUT_SECS="${INSTANCE_TIMEOUT_SECS:-150}" \
    RAM_LIMIT_KB="${RAM_LIMIT_KB:-15000000}" \
    env "${extra_env[@]}" \
    bash quick_scripts/run_all_txt_instances_limited.sh

    python3 "$CSV_BATCH" "$json_out" --format compact --overwrite \
        && echo "  CSV ok" || echo "  (CSV failed)"
    echo
}

START_TIME=$(date +%s)
echo "=== SAT bench: 2 configs × 3 objectives (6 runs) ==="
echo "Start: $(date)"
echo "Output dir: $OUT_DIR"
echo

for obj in "${OBJECTIVES[@]}"; do
    # SatDefault — no env vars set, binary's CLI defaults take over.
    run_config SatDefault "$obj"

    # SatNothing — disable extended precedence graph + SC AMO.
    # (3 seed flags are already off by default.)
    run_config SatNothing "$obj" \
        SATDDD_USE_PRECEDENCE_GRAPH=false \
        SATDDD_USE_SC_AMO=false
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "============================================================"
echo "=== All 6 runs DONE in ${ELAPSED}s ($(( ELAPSED / 60 )) min) ==="
echo "End: $(date)"
echo "Outputs: $OUT_DIR/"
ls -la "$OUT_DIR/"
