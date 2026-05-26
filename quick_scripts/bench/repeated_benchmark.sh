#!/usr/bin/env bash
# Repeated benchmark runner for two solvers on the same objective.
#
# Usage examples:
#   bash quick_scripts/bench/repeated_benchmark.sh
#       → defaults: SC=10 reps, LADDER=10 reps, OBJECTIVE=infsteps180
#
#   SC_REPS=10 LADDER_REPS=10 bash quick_scripts/bench/repeated_benchmark.sh
#
#   OBJECTIVE=finsteps123 SC_REPS=5 LADDER_REPS=5 \
#       bash quick_scripts/bench/repeated_benchmark.sh
#
# Outputs:
#   results_repeated/<timestamp>/
#     sc_run_1.json,   sc_run_1.log,   ... sc_run_N.json
#     ladder_run_1.json, ladder_run_1.log, ... ladder_run_M.json
#     summary.txt   (aggregate per solver)
#
# To compare specific sc variant, change SC_SOLVER and SC_EXTRA_ARGS.

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────
BIN="${BIN:-./target/release/ddd}"
OBJECTIVE="${OBJECTIVE:-infsteps180}"

SC_SOLVER="${SC_SOLVER:-maxsat_ddd_ladder_sc}"
# Current default: ISOLATE the eager-chain-expansion feature.
#   use_precedence_graph         = false
#   use_eager_chain_expansion    = TRUE   ← only this one ON
#   use_interval_graph_conflicts = false
#
# This activates:
#   - Eager chain expansion inside add_fixed_precedence_row (long chains)
#   - Eager precedence propagation in pair-based resource-conflict fallback
#   - Seed-from-earliest precedence rows
# Does NOT activate the SC AMO encoding — that lives in
# add_sc_amo and is only reached via interval_graph_conflicts.
#
# To run "sc_as_ladder" (all 3 OFF, closest to ladder semantics):
#   SC_EXTRA_ARGS="--maxsat-ladder-sc-use-precedence-graph false \
#                   --maxsat-ladder-sc-use-eager-chain-expansion false \
#                   --maxsat-ladder-sc-use-interval-graph false" \
#     bash quick_scripts/bench/repeated_benchmark.sh
#
# To use full default sc features (all 3 ON):
#   SC_EXTRA_ARGS="" bash quick_scripts/bench/repeated_benchmark.sh
SC_EXTRA_ARGS=(${SC_EXTRA_ARGS:-"--maxsat-ladder-sc-use-precedence-graph" "false" "--maxsat-ladder-sc-use-eager-chain-expansion" "true" "--maxsat-ladder-sc-use-interval-graph" "false"})

LADDER_SOLVER="${LADDER_SOLVER:-maxsat_ddd_ladder}"
LADDER_EXTRA_ARGS=(${LADDER_EXTRA_ARGS:-})

SC_REPS="${SC_REPS:-10}"
LADDER_REPS="${LADDER_REPS:-10}"

# Output directory: one subdirectory per run-batch, timestamped.
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-results_repeated/${TS}_${OBJECTIVE}}"
mkdir -p "$OUT_DIR"

# ─── Echo configuration ──────────────────────────────────────────
{
  echo "=== Repeated Benchmark ==="
  echo "Timestamp:        $TS"
  echo "Output dir:       $OUT_DIR"
  echo "Binary:           $BIN"
  echo "Objective:        $OBJECTIVE"
  echo
  echo "SC solver:       $SC_SOLVER"
  echo "SC extra args:   ${SC_EXTRA_ARGS[*]}"
  echo "SC reps:         $SC_REPS"
  echo
  echo "LADDER solver:    $LADDER_SOLVER"
  echo "LADDER extra:     ${LADDER_EXTRA_ARGS[*]}"
  echo "LADDER reps:      $LADDER_REPS"
} | tee "$OUT_DIR/config.txt"

# ─── Run helper ──────────────────────────────────────────────────
run_one() {
  local solver="$1"; shift
  local label="$1"; shift
  local rep="$1"; shift
  local extra_args=("$@")

  local json_out="$OUT_DIR/${label}_run_${rep}.json"
  local stdout_log="$OUT_DIR/${label}_run_${rep}.log"

  echo "─── Run $label #$rep (solver=$solver) ───"
  local start_ts="$(date +%s.%N)"

  "$BIN" \
    -s "$solver" \
    --txt-instances \
    --objective "$OBJECTIVE" \
    --json-output "$json_out" \
    "${extra_args[@]}" \
    > "$stdout_log" 2>&1 \
    || echo "  (run exited non-zero, see log)"

  local end_ts="$(date +%s.%N)"
  local elapsed
  elapsed=$(awk "BEGIN {printf \"%.2f\", $end_ts - $start_ts}")
  echo "  → wall-clock: ${elapsed}s   json: $(basename "$json_out")"
}

# ─── Run SC reps ────────────────────────────────────────────────
echo
echo "=== Running $SC_SOLVER × $SC_REPS ==="
for i in $(seq 1 "$SC_REPS"); do
  run_one "$SC_SOLVER" sc "$i" "${SC_EXTRA_ARGS[@]}"
done

# ─── Run LADDER reps ─────────────────────────────────────────────
echo
echo "=== Running $LADDER_SOLVER × $LADDER_REPS ==="
for i in $(seq 1 "$LADDER_REPS"); do
  run_one "$LADDER_SOLVER" ladder "$i" "${LADDER_EXTRA_ARGS[@]}"
done

# ─── Convert JSON → CSV (if helper script exists) ────────────────
if [[ -f quick_scripts/analyze/json_to_csv_batch.py ]]; then
  echo
  echo "=== Converting JSON → CSV ==="
  python3 quick_scripts/analyze/json_to_csv_batch.py \
    "$OUT_DIR"/sc_run_*.json "$OUT_DIR"/ladder_run_*.json \
    --format compact --overwrite \
    || echo "  (conversion failed — run manually if needed)"
fi

echo
echo "=== Done. Results in: $OUT_DIR ==="
