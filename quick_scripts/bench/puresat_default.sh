#!/usr/bin/env bash
# OOM-safe puresat runner across one or more objectives, with default settings.
#
# `puresat` = `sat_ddd_sc_fresh_addclauses` — fresh-per-iteration rebuild.
# Has O(N²) clause replay overhead, so cont/infsteps180 use per-instance
# isolation via `run_all_txt_instances_limited.sh` (timeout + RAM cap so
# OOM/hang on one instance doesn't kill the batch).
#
# Settings (puresat::SatDddSettings::default()):
#   use_precedence_graph:           true    (chain prop only, no extended)
#   prealloc_cost_thresholds:       false   (lazy)
#   seed_precedence_from_earliest:  false   (lazy)
#   seed_resource_conflicts:        false   (lazy)
#   use_sc_amo:                    true    (SC AMO for cliques ≥ 6)
#
# Usage:
#   bash quick_scripts/bench/puresat_default.sh
#       → all 3 objectives (finsteps123, infsteps180, cont)
#
#   OBJECTIVES="cont" bash quick_scripts/bench/puresat_default.sh
#       → cont only
#
#   OBJECTIVES="finsteps123 infsteps180" bash quick_scripts/bench/puresat_default.sh
#       → two objectives
#
# Tunables:
#   OUT_DIR_BASE       base output dir (default: thesis verify root)
#   INSTANCE_TIMEOUT_SECS   per-instance wall cap (default 150s)
#   RAM_LIMIT_KB       per-instance virtual-mem cap (default 15 GB)

set -euo pipefail

OBJECTIVES_DEFAULT=(finsteps123 infsteps180 cont)
read -r -a OBJECTIVES <<<"${OBJECTIVES:-${OBJECTIVES_DEFAULT[*]}}"

OUT_DIR_BASE="${OUT_DIR_BASE:-2026-05-16-Verified-Result-For-Graduation-Thesis}"
CSV_BATCH="quick_scripts/analyze/json_to_csv_batch.py"

for OBJECTIVE in "${OBJECTIVES[@]}"; do
    case "$OBJECTIVE" in
        finsteps123) SUFFIX=finsteps ;;
        infsteps180) SUFFIX=infsteps ;;
        cont)        SUFFIX=cont ;;
        *) echo "Unknown objective: $OBJECTIVE" >&2; exit 1 ;;
    esac

    OUT_DIR="${OUT_DIR_BASE}/puresat_${SUFFIX}_verify"
    mkdir -p "$OUT_DIR"
    JSON_OUT="${OUT_DIR}/PureDefault_${OBJECTIVE}.json"

    echo "=== puresat ${OBJECTIVE} (default, OOM-safe per-instance) ==="
    echo "Output JSON: $JSON_OUT"
    echo

    BIN="${BIN:-./target/release/ddd}" \
    SOLVER=sat_ddd_sc_fresh_addclauses \
    OBJECTIVE="$OBJECTIVE" \
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
    echo
done
