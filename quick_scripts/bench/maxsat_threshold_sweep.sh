#!/usr/bin/env bash
# MaxSAT SC AMO threshold sweep — runs MaxSAT-Default × 3 objectives at the
# specified PAIRWISE_AMO_MAX_SIZE value.
#
# The threshold lives as a `const` in
# src/solvers/maxsatddd_ladder_sc.rs and is compile-time, so this script
# expects the user to have set the desired value already, then takes the
# threshold tag as input (used only for the output dir name).
#
# Usage:
#   N=3  bash quick_scripts/bench/maxsat_threshold_sweep.sh   # → maxsat_verify_n3/
#   N=5  bash quick_scripts/bench/maxsat_threshold_sweep.sh   # → maxsat_verify_n5/
#   N=10 bash quick_scripts/bench/maxsat_threshold_sweep.sh   # → maxsat_verify_n10/
#   N=5  RUN_TAG=run2 bash quick_scripts/bench/maxsat_threshold_sweep.sh
#                                                             # → maxsat_verify_n5_run2/
#
# Background:
#   nohup N=5 bash quick_scripts/bench/maxsat_threshold_sweep.sh \
#       > maxsat_n5_bench.log 2>&1 &
#   tail -f maxsat_n5_bench.log
#
# Background:
#   - n=3:  aggressive (SC AMO active for cliques ≥ 4); risks gaining aux
#           vars without enough clause savings.
#   - n=5:  empirically best on the Croella2024 benchmark; current default.
#   - n=10: conservative (cliques 6-10 fall back to pairwise); often slower
#           than n=5 by 15-46% on infsteps180/cont.

set -euo pipefail

: "${N:?Set N (e.g. N=5) — the PAIRWISE_AMO_MAX_SIZE the binary was built with.}"
RUN_TAG="${RUN_TAG:-}"
SUFFIX="n${N}${RUN_TAG:+_${RUN_TAG}}"

echo "[1/2] Rebuilding binary (assumes PAIRWISE_AMO_MAX_SIZE = $N in source)..."
cargo build --release

echo "[2/2] Running MaxSAT-Default × 3 objectives → maxsat_verify_${SUFFIX}/"
OUT_DIR="2026-05-16-Verified-Result-For-Graduation-Thesis/maxsat_verify_${SUFFIX}" \
    bash quick_scripts/bench/maxsat_default.sh
