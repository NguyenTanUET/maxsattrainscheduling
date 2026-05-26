#!/usr/bin/env bash
# Helper to benchmark different SC-AMO tunings.
#
# Each tuning requires editing two `const` values at the top of
# `src/solvers/maxsatddd_ladder_sc.rs` (PAIRWISE_AMO_MAX_SIZE,
# LAZY_AMO_THRESHOLD), rebuilding, and running the benchmark.
#
# This script automates the build+run for a SINGLE chosen tuning,
# tagging the output file accordingly. Run it after manually editing
# the two consts to the values you want to test.
#
# Usage:
#   PAIRWISE_TAG=10 LAZY_TAG=3 bash quick_scripts/ablation/sc_amo_tune.sh
#
# Output: results_amo_pw${PAIRWISE_TAG}_lazy${LAZY_TAG}_${OBJECTIVE}.json
#
# Recommended sweep:
#   pw=5  lazy=0   (= current eager SC baseline)
#   pw=10 lazy=0   (pairwise up to 10, then SC)
#   pw=20 lazy=0   (pairwise up to 20, then SC)
#   pw=5  lazy=3   (3-touch lazy warm-up, then SC)
#   pw=10 lazy=3   (both tweaks combined)
#
# Default config tested: full SC features ON
# (`--use-interval-graph=true` so SC-AMO actually fires).

set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OBJECTIVE="${OBJECTIVE:-infsteps180}"
PAIRWISE_TAG="${PAIRWISE_TAG:-5}"
LAZY_TAG="${LAZY_TAG:-0}"
OUT_PREFIX="${OUT_PREFIX:-results_amo}"

out="${OUT_PREFIX}_pw${PAIRWISE_TAG}_lazy${LAZY_TAG}_${OBJECTIVE}.json"

echo "=== SC-AMO tuning: PAIRWISE_AMO_MAX_SIZE=${PAIRWISE_TAG} LAZY_AMO_THRESHOLD=${LAZY_TAG} ==="
echo "Objective: ${OBJECTIVE}"
echo "Output:    ${out}"
echo
echo "VERIFY before running:"
echo "  grep 'PAIRWISE_AMO_MAX_SIZE: usize' src/solvers/maxsatddd_ladder_sc.rs"
echo "  grep 'LAZY_AMO_THRESHOLD: usize' src/solvers/maxsatddd_ladder_sc.rs"
echo

echo "--- cargo build --release ---"
cargo build --release

echo
echo "--- Running benchmark ---"
"$BIN" \
  -s maxsat_ddd_ladder_sc \
  --maxsat-ladder-sc-use-precedence-graph true \
  --maxsat-ladder-sc-use-eager-chain-expansion false \
  --maxsat-ladder-sc-use-interval-graph true \
  --txt-instances \
  --objective "$OBJECTIVE" \
  --json-output "$out"

echo
echo "=== Done. Output: ${out} ==="
