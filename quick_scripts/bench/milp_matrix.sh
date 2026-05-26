#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OUT_DIR="${OUT_DIR:-.}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_LOGS="${RUN_LOGS:-1}"
RAM_LIMIT_KB="${RAM_LIMIT_KB:-0}"
ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-}"
INSTANCE_NAME_FILTER="${INSTANCE_NAME_FILTER:-}"

objectives=(cont infsteps180 finsteps123)

mkdir -p "$OUT_DIR"
if [[ "$RUN_LOGS" = "1" ]]; then
  mkdir -p "$LOG_DIR"
fi

run_one() {
  local solver="$1"
  local objective="$2"
  local json_name="$3"
  shift 3
  local -a extra_args=("$@")
  local json_out="$OUT_DIR/$json_name"
  local log_out="$LOG_DIR/${json_name%.json}.log"
  local -a cmd=(
    "$BIN"
    -s "$solver"
    --txt-instances
    --objective "$objective"
  )

  if [[ -n "$INSTANCE_NAME_FILTER" ]]; then
    cmd+=(--instance-name-filter "$INSTANCE_NAME_FILTER")
  fi

  cmd+=("${extra_args[@]}")
  cmd+=(--json-output "$json_out")

  echo "Running solver=$solver objective=$objective"
  echo "  json: $json_out"

  if [[ "$RUN_LOGS" = "1" ]]; then
    (
      if [[ -n "$ACTIVATE_SCRIPT" ]]; then
        # shellcheck disable=SC1090
        source "$ACTIVATE_SCRIPT"
      fi

      if [[ "$RAM_LIMIT_KB" -gt 0 ]]; then
        ulimit -Sv "$RAM_LIMIT_KB"
      fi

      exec "${cmd[@]}"
    ) >"$log_out" 2>&1
    echo "  log: $log_out"
  else
    (
      if [[ -n "$ACTIVATE_SCRIPT" ]]; then
        # shellcheck disable=SC1090
        source "$ACTIVATE_SCRIPT"
      fi

      if [[ "$RAM_LIMIT_KB" -gt 0 ]]; then
        ulimit -Sv "$RAM_LIMIT_KB"
      fi

      exec "${cmd[@]}"
    )
  fi
}

for objective in "${objectives[@]}"; do
  run_one \
    "bigm_lazy" \
    "$objective" \
    "results_bigm_lazy_${objective}_120s.json"

  run_one \
    "mip_ddd" \
    "$objective" \
    "results_mip_ddd_${objective}_120s.json"

  run_one \
    "sat_ddd_sc" \
    "$objective" \
    "results_sat_sc_${objective}_bit_totalizer_120s.json" \
    --satddd-objective-encoding bit_totalizer \
    --satddd-use-precedence-graph true
done

echo "Done."
