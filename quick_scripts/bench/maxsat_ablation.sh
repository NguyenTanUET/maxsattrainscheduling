#!/usr/bin/env bash
# Ablation study for the two MaxSAT+SC improvements (Chapter 3 contributions):
#   Contribution 1: SC AMO encoding for resource cliques (use_sc_amo + use_touched_clique_amo)
#   Contribution 2: Precedence graph preprocessing      (use_precedence_graph)
#
# Two ablation configurations:
#   MaxsatOnlyPrec — only Contribution 2 (precedence preprocessing):
#                      use_precedence_graph:    true
#                      use_sc_amo:             false
#                      use_touched_clique_amo:  false
#   MaxsatOnlySc  — only Contribution 1 (SC AMO encoding):
#                      use_precedence_graph:    false
#                      use_sc_amo:             true
#                      use_touched_clique_amo:  true
#
# Compare with the existing reference points:
#   MaxsatScDefault (maxsat_verify/)    — both contributions on
#   MaxsatBaseline   (maxsat_baseline/)  — neither (Croella2024 ladder)
#
# Objectives (3): finsteps123, infsteps180, cont.
# Total: 6 runs. Each instance isolated via run_all_txt_instances_limited.sh.
#
# Output: 2026-05-16-Verified-Result-For-Graduation-Thesis/maxsat_ablation/
#         {MaxsatOnlyPrec,MaxsatOnlySc}_{finsteps123,infsteps180,cont}.{json,csv}
#
# Usage:
#   bash quick_scripts/bench/maxsat_ablation.sh
#
# Background usage (recommended while editing thesis):
#   nohup bash quick_scripts/bench/maxsat_ablation.sh \
#       > maxsat_ablation_bench.log 2>&1 &
#   tail -f maxsat_ablation_bench.log
#
# Tunables (env vars):
#   INSTANCE_TIMEOUT_SECS  per-instance wall cap (default 150s)
#   RAM_LIMIT_KB           per-instance virtual-mem cap (default 15GB)

set -euo pipefail

echo "[0/2] Rebuilding binary..."
cargo build --release
BIN="${BIN:-./target/release/ddd}"
[[ -x "$BIN" ]] || { echo "ERROR: $BIN not found after build"; exit 1; }
echo "Binary OK: $BIN"
echo

OUT_DIR="${OUT_DIR:-2026-05-16-Verified-Result-For-Graduation-Thesis/maxsat_ablation}"
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
    SOLVER=maxsat_ddd_ladder_sc \
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
echo "=== MaxSAT ablation: 2 configs × 3 objectives (6 runs) ==="
echo "Start: $(date)"
echo "Output dir: $OUT_DIR"
echo

for obj in "${OBJECTIVES[@]}"; do
    # MaxsatOnlyPrec — only precedence preprocessing on, SC AMO off.
    run_config MaxsatOnlyPrec "$obj" \
        MAXSATDDD_USE_PRECEDENCE_GRAPH=true \
        MAXSATDDD_USE_SC_AMO=false \
        MAXSATDDD_USE_TOUCHED_CLIQUE_AMO=false

    # MaxsatOnlySc — only SC AMO encoding on, precedence preprocessing off.
    run_config MaxsatOnlySc "$obj" \
        MAXSATDDD_USE_PRECEDENCE_GRAPH=false \
        MAXSATDDD_USE_SC_AMO=true \
        MAXSATDDD_USE_TOUCHED_CLIQUE_AMO=true
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "============================================================"
echo "=== All 6 runs DONE in ${ELAPSED}s ($(( ELAPSED / 60 )) min) ==="
echo "End: $(date)"
echo "Outputs: $OUT_DIR/"
ls -la "$OUT_DIR/"
