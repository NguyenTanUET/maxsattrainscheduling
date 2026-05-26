#!/usr/bin/env bash
# Verification runs for `sat_ddd_sc_totalizer` (incremental SAT with
# hardcoded IncrementalTotalizer encoding) across one or more objectives
# and one or more configurations.
#
# Configurations:
#   IncTotDefault — current SatDddSettings::default(): chain prop ON,
#                   SC AMO ON, no seeds, no prealloc.
#   IncTotNothing — all SatDddSettings stripped (pairwise AMO, no chain,
#                   no seeds, no prealloc) — analog of ScNothing.
#
# Usage:
#   bash quick_scripts/bench/incsat_totalizer_verify.sh
#       → all 3 objectives × both configs
#
#   OBJECTIVES="finsteps123" CONFIGS="IncTotDefault" \
#       bash quick_scripts/bench/incsat_totalizer_verify.sh
#       → finsteps123 × Default only
#
# Tunables:
#   BIN              path to release binary (default ./target/release/ddd)
#   OUT_DIR_BASE     base output dir (default thesis verify root)

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OUT_DIR_BASE="${OUT_DIR_BASE:-2026-05-16-Verified-Result-For-Graduation-Thesis}"
CSV_BATCH="quick_scripts/analyze/json_to_csv_batch.py"

OBJECTIVES_DEFAULT=(finsteps123 infsteps180 cont)
CONFIGS_DEFAULT=(IncTotDefault IncTotNothing)

read -r -a OBJECTIVES <<<"${OBJECTIVES:-${OBJECTIVES_DEFAULT[*]}}"
read -r -a CONFIGS    <<<"${CONFIGS:-${CONFIGS_DEFAULT[*]}}"

# Per-config extra CLI flags.
flags_for_config() {
    case "$1" in
        IncTotDefault) echo "" ;;
        IncTotNothing)
            echo "--satddd-use-precedence-graph false \
                  --satddd-prealloc-cost-thresholds false \
                  --satddd-seed-precedence-from-earliest false \
                  --satddd-seed-resource-conflicts false \
                  --satddd-use-sc-amo false" ;;
        *) echo "Unknown config: $1" >&2; return 1 ;;
    esac
}

for OBJECTIVE in "${OBJECTIVES[@]}"; do
    case "$OBJECTIVE" in
        finsteps123) SUFFIX=finsteps ;;
        infsteps180) SUFFIX=infsteps ;;
        cont)        SUFFIX=cont ;;
        *) echo "Unknown objective: $OBJECTIVE" >&2; exit 1 ;;
    esac

    OUT_DIR="${OUT_DIR_BASE}/inc_totalizer_${SUFFIX}_verify"
    mkdir -p "$OUT_DIR"

    echo "=== inc_totalizer ${OBJECTIVE} ==="
    echo "Output dir: $OUT_DIR"
    echo "Binary:     $BIN"
    echo

    for TAG in "${CONFIGS[@]}"; do
        OUT="${OUT_DIR}/${TAG}_${OBJECTIVE}.json"
        echo "--- [${TAG} | ${OBJECTIVE}] ---"

        # shellcheck disable=SC2046
        "$BIN" \
          -s sat_ddd_sc_totalizer \
          --txt-instances \
          --objective "$OBJECTIVE" \
          $(flags_for_config "$TAG") \
          --json-output "$OUT"

        python3 "$CSV_BATCH" "$OUT" --format compact --overwrite \
            && echo "  CSV ok" || echo "  (CSV failed)"
        echo
    done

    echo "=== Done ${OBJECTIVE}. Outputs in $OUT_DIR ==="
    echo
done
