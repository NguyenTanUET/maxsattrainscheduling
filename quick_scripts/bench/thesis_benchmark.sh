#!/usr/bin/env bash
# Thesis verification benchmark:
#   - Solvers:    maxsat_ddd_ladder, maxsat_ddd_ladder_sc (default = Option B)
#   - Objectives: finsteps123, infsteps180, cont
#   - Total 6 runs (2 solvers × 3 objectives).
#   - After each run, convert JSON → CSV.
#   - All outputs land in `2026-05-16-Verified-Result-For-Graduation-Thesis/`.
#
# Usage:
#   bash quick_scripts/bench/thesis_benchmark.sh

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OUT_DIR="${OUT_DIR:-2026-05-16-Verified-Result-For-Graduation-Thesis}"
CSV_BATCH="quick_scripts/analyze/json_to_csv_batch.py"

mkdir -p "$OUT_DIR"

SOLVERS=(maxsat_ddd_ladder maxsat_ddd_ladder_sc)
OBJECTIVES=(finsteps123 infsteps180 cont)

echo "=== Thesis verification benchmark ==="
echo "Output dir: $OUT_DIR"
echo "Binary:     $BIN"
echo "Solvers:    ${SOLVERS[*]}"
echo "Objectives: ${OBJECTIVES[*]}"
echo

for solver in "${SOLVERS[@]}"; do
    for obj in "${OBJECTIVES[@]}"; do
        json_out="${OUT_DIR}/${solver}_${obj}.json"
        echo "--- [${solver} | ${obj}] ---"

        "$BIN" \
          -s "$solver" \
          --txt-instances \
          --objective "$obj" \
          --json-output "$json_out"

        echo "  → JSON: $json_out"

        # Convert this run's JSON → CSV immediately (compact format).
        python3 "$CSV_BATCH" "$json_out" --format compact --overwrite \
            && echo "  → CSV:  ${json_out%.json}.csv" \
            || echo "  (CSV conversion failed for $json_out)"
        echo
    done
done

echo "=== All 6 runs complete. ==="
echo "JSON + CSV files saved in: $OUT_DIR"
