#!/usr/bin/env bash
# Run all 4 MaxSAT configurations across all 3 objectives in one go,
# with the new CountingSolver wrapper recording total variables and
# clauses into the JSON/CSV output. All results go into ONE folder
# `maxsat_clause_var_check/` for direct comparison.
#
# 4 configurations:
#   MaxsatBaseline — maxsat_ddd_ladder (Croella2024 baseline, no SC/Prec)
#   MaxsatDefault  — maxsat_ddd_ladder_sc with both cải tiến on
#                       use_precedence_graph:    true
#                       use_sc_amo:             true
#                       use_touched_clique_amo:  true
#   MaxsatSC      — maxsat_ddd_ladder_sc with ONLY SC AMO
#                       use_precedence_graph:    false
#                       use_sc_amo:             true
#                       use_touched_clique_amo:  true
#   MaxsatPrec     — maxsat_ddd_ladder_sc with ONLY preprocessing
#                       use_precedence_graph:    true
#                       use_sc_amo:             false
#                       use_touched_clique_amo:  false
#
# Objectives (3): finsteps123, infsteps180, cont.
# Total: 12 runs. Each instance isolated via run_all_txt_instances_limited.sh
# so OOM/timeout on one instance doesn't kill the batch.
#
# Output: 2026-05-16-Verified-Result-For-Graduation-Thesis/maxsat_clause_var_check/
#         {MaxsatBaseline,MaxsatDefault,MaxsatSC,MaxsatPrec}_{finsteps123,infsteps180,cont}.{json,csv}
#
# Each CSV will contain the new columns:
#   num_vars_total    -- total SAT variables in the final CNF
#   num_clauses_total -- total CNF clauses (initial + DDD-generated)
#   num_conflicts     -- fixed: now correctly counts AMO resource clauses
#                        (was bugged as num_traveltime before this commit)
#
# Usage:
#   bash quick_scripts/bench/maxsat_4config_clause_var.sh
#
# Background:
#   nohup bash quick_scripts/bench/maxsat_4config_clause_var.sh \
#       > maxsat_4config_clause_var.log 2>&1 &
#   tail -f maxsat_4config_clause_var.log
#
# Tunables (env vars):
#   INSTANCE_TIMEOUT_SECS  per-instance wall cap (default 150s)
#   RAM_LIMIT_KB           per-instance virtual-mem cap (default 15GB)

set -euo pipefail

echo "[0/2] Rebuilding binary (with CountingSolver instrumentation)..."
cargo build --release
BIN="${BIN:-./target/release/ddd}"
[[ -x "$BIN" ]] || { echo "ERROR: $BIN not found after build"; exit 1; }
echo "Binary OK: $BIN"
echo

OUT_DIR="${OUT_DIR:-2026-05-16-Verified-Result-For-Graduation-Thesis/maxsat_clause_var_check}"
CSV_BATCH="quick_scripts/analyze/json_to_csv_batch.py"

mkdir -p "$OUT_DIR"

OBJECTIVES=(finsteps123 infsteps180 cont)

run_config() {
    local tag="$1"; shift
    local solver="$1"; shift
    local obj="$1"; shift
    local extra_env=("$@")

    local json_out="${OUT_DIR}/${tag}_${obj}.json"

    echo "============================================================"
    echo "[${tag} | ${obj}]"
    echo "  Solver: $solver"
    echo "  JSON:   $json_out"
    echo "  Env:    ${extra_env[*]:-(default)}"
    echo "============================================================"

    BIN="$BIN" \
    SOLVER="$solver" \
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
echo "=== MaxSAT 4 configs × 3 objectives = 12 runs ==="
echo "Start:      $(date)"
echo "Output dir: $OUT_DIR"
echo

for obj in "${OBJECTIVES[@]}"; do
    # MaxsatBaseline — Croella ladder, no SC/Prec settings to override.
    run_config MaxsatBaseline maxsat_ddd_ladder "$obj"

    # MaxsatDefault — both improvements ON (CLI defaults).
    run_config MaxsatDefault maxsat_ddd_ladder_sc "$obj"

    # MaxsatSC — only SC AMO improvement ON, precedence preprocessing OFF.
    run_config MaxsatSC maxsat_ddd_ladder_sc "$obj" \
        MAXSATDDD_USE_PRECEDENCE_GRAPH=false \
        MAXSATDDD_USE_SC_AMO=true \
        MAXSATDDD_USE_TOUCHED_CLIQUE_AMO=true

    # MaxsatPrec — only preprocessing ON, SC AMO OFF (pairwise as Croella).
    run_config MaxsatPrec maxsat_ddd_ladder_sc "$obj" \
        MAXSATDDD_USE_PRECEDENCE_GRAPH=true \
        MAXSATDDD_USE_SC_AMO=false \
        MAXSATDDD_USE_TOUCHED_CLIQUE_AMO=false
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "============================================================"
echo "=== All 12 runs DONE in ${ELAPSED}s ($(( ELAPSED / 60 )) min) ==="
echo "End: $(date)"
echo "Outputs in $OUT_DIR/:"
ls -la "$OUT_DIR/"
