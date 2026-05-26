#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
OUT_DIR="${OUT_DIR:-.}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_LOGS="${RUN_LOGS:-1}"
RAM_LIMIT_KB="${RAM_LIMIT_KB:-0}"

objectives=(cont infsteps180 finsteps123)
solvers=(bigm_lazy mip_ddd)

mkdir -p "$OUT_DIR"
if [[ "$RUN_LOGS" = "1" ]]; then
  mkdir -p "$LOG_DIR"
fi

run_one() {
  local solver="$1"
  local objective="$2"
  local json_out="$OUT_DIR/results_${solver}_${objective}_120s.json"
  local log_out="$LOG_DIR/${solver}_${objective}_120s.log"

  echo "Running solver=$solver objective=$objective"
  echo "  json: $json_out"

  if [[ "$RUN_LOGS" = "1" ]]; then
    (
      if [[ "$RAM_LIMIT_KB" -gt 0 ]]; then
        ulimit -Sv "$RAM_LIMIT_KB"
      fi

      exec "$BIN" \
        -s "$solver" \
        --txt-instances \
        --objective "$objective" \
        --json-output "$json_out"
    ) >"$log_out" 2>&1
    echo "  log: $log_out"
  else
    (
      if [[ "$RAM_LIMIT_KB" -gt 0 ]]; then
        ulimit -Sv "$RAM_LIMIT_KB"
      fi

      exec "$BIN" \
        -s "$solver" \
        --txt-instances \
        --objective "$objective" \
        --json-output "$json_out"
    )
  fi
}

for objective in "${objectives[@]}"; do
  for solver in "${solvers[@]}"; do
    run_one "$solver" "$objective"
  done
done

echo "Done."
