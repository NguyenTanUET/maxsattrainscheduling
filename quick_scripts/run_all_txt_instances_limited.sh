#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./target/release/ddd}"
SOLVER="${SOLVER:-sat_ddd_sc_totalizer}"
OBJECTIVE="${OBJECTIVE:-cont}"
JSON_OUT="${JSON_OUT:-results_sat_sc_cont_totalizer.json}"
INSTANCE_TIMEOUT_SECS="${INSTANCE_TIMEOUT_SECS:-130}"
RAM_LIMIT_KB="${RAM_LIMIT_KB:-12000000}"
KEEP_TMPDIR="${KEEP_TMPDIR:-0}"
SATDDD_OBJECTIVE_ENCODING="${SATDDD_OBJECTIVE_ENCODING:-}"

# `SatDddSettings` knobs forwarded to `--satddd-*` flags when set.
# Empty value means "leave the binary's default in place".
# `SATDDD_USE_PRECEDENCE_GRAPH` is kept as an alias for the renamed
# `--satddd-use-precedence-graph` so older invocations still work.
SATDDD_USE_PRECEDENCE_GRAPH="${SATDDD_USE_PRECEDENCE_GRAPH:-${SATDDD_USE_PRECEDENCE_GRAPH:-}}"
SATDDD_PREALLOC_COST_THRESHOLDS="${SATDDD_PREALLOC_COST_THRESHOLDS:-}"
SATDDD_SEED_PRECEDENCE_FROM_EARLIEST="${SATDDD_SEED_PRECEDENCE_FROM_EARLIEST:-}"
SATDDD_SEED_RESOURCE_CONFLICTS="${SATDDD_SEED_RESOURCE_CONFLICTS:-}"
SATDDD_USE_SC_AMO="${SATDDD_USE_SC_AMO:-}"

# `MaxSatDddLadderScSettings` knobs forwarded to `--maxsat-ladder-sc-*`
# flags when set. Effective only when SOLVER=maxsat_ddd_ladder_sc. Empty
# value means "leave the binary's default in place".
MAXSATDDD_USE_PRECEDENCE_GRAPH="${MAXSATDDD_USE_PRECEDENCE_GRAPH:-}"
MAXSATDDD_USE_SC_AMO="${MAXSATDDD_USE_SC_AMO:-}"
MAXSATDDD_USE_TOUCHED_CLIQUE_AMO="${MAXSATDDD_USE_TOUCHED_CLIQUE_AMO:-}"

tmpdir="$(mktemp -d /tmp/ddd-run-all.XXXXXX)"
order_file="$tmpdir/order.txt"
touch "$order_file"
cleanup() {
  if [[ "$KEEP_TMPDIR" = "1" ]]; then
    echo "Preserved run artifacts in $tmpdir"
  else
    rm -rf "$tmpdir"
  fi
}
trap cleanup EXIT

run_one() {
  local instance_name="$1"
  local out_json="$tmpdir/$instance_name.json"
  local stdout_log="$tmpdir/$instance_name.stdout.log"
  local stderr_log="$tmpdir/$instance_name.stderr.log"
  local -a extra_args=()
  local effective_solver="$SOLVER"
  local status=0
  local start_ms=0
  local end_ms=0
  local elapsed_ms=0
  local elapsed_seconds="0.0"

  if [[ -n "$SATDDD_OBJECTIVE_ENCODING" ]]; then
    if [[ "$effective_solver" == "sat_ddd_sc_totalizer" ]]; then
      effective_solver="sat_ddd_sc"
    fi
    extra_args+=(--satddd-objective-encoding "$SATDDD_OBJECTIVE_ENCODING")
  fi

  if [[ -n "$SATDDD_USE_PRECEDENCE_GRAPH" ]]; then
    extra_args+=(--satddd-use-precedence-graph "$SATDDD_USE_PRECEDENCE_GRAPH")
  fi
  if [[ -n "$SATDDD_PREALLOC_COST_THRESHOLDS" ]]; then
    extra_args+=(--satddd-prealloc-cost-thresholds "$SATDDD_PREALLOC_COST_THRESHOLDS")
  fi
  if [[ -n "$SATDDD_SEED_PRECEDENCE_FROM_EARLIEST" ]]; then
    extra_args+=(--satddd-seed-precedence-from-earliest "$SATDDD_SEED_PRECEDENCE_FROM_EARLIEST")
  fi
  if [[ -n "$SATDDD_SEED_RESOURCE_CONFLICTS" ]]; then
    extra_args+=(--satddd-seed-resource-conflicts "$SATDDD_SEED_RESOURCE_CONFLICTS")
  fi
  if [[ -n "$SATDDD_USE_SC_AMO" ]]; then
    extra_args+=(--satddd-use-sc-amo "$SATDDD_USE_SC_AMO")
  fi

  if [[ -n "$MAXSATDDD_USE_PRECEDENCE_GRAPH" ]]; then
    extra_args+=(--maxsat-ladder-sc-use-precedence-graph "$MAXSATDDD_USE_PRECEDENCE_GRAPH")
  fi
  if [[ -n "$MAXSATDDD_USE_SC_AMO" ]]; then
    extra_args+=(--maxsat-ladder-sc-use-sc-amo "$MAXSATDDD_USE_SC_AMO")
  fi
  if [[ -n "$MAXSATDDD_USE_TOUCHED_CLIQUE_AMO" ]]; then
    extra_args+=(--maxsat-ladder-sc-use-touched-clique-amo "$MAXSATDDD_USE_TOUCHED_CLIQUE_AMO")
  fi

  echo "Running $instance_name"
  printf '%s\n' "$instance_name" >> "$order_file"
  start_ms="$(date +%s%3N)"

  (
    if [[ "$RAM_LIMIT_KB" -gt 0 ]]; then
      ulimit -Sv "$RAM_LIMIT_KB"
    fi

    exec timeout -k 20 "${INSTANCE_TIMEOUT_SECS}s" \
      "$BIN" \
      -s "$effective_solver" \
      --txt-instances \
      --instance-name-filter "$instance_name" \
      --instance-name-exact \
      --objective "$OBJECTIVE" \
      "${extra_args[@]}" \
      --json-output "$out_json" \
      >"$stdout_log" \
      2>"$stderr_log"
  ) || status=$?
  end_ms="$(date +%s%3N)"
  elapsed_ms=$((end_ms - start_ms))
  elapsed_seconds="$(awk "BEGIN { printf \"%.3f\", $elapsed_ms / 1000 }")"

  if [[ -s "$out_json" ]]; then
    return 0
  fi

  local external_status="failed"
  if [[ "$status" -eq 124 ]]; then
    external_status="external_timeout"
  elif [[ "$status" -eq 137 ]]; then
    external_status="killed"
  elif [[ "$status" -eq 134 ]]; then
    external_status="aborted"
  elif [[ "$status" -eq 139 ]]; then
    external_status="segfault"
  elif [[ "$status" -eq 101 ]]; then
    external_status="panic"
  fi

  cat >"$out_json" <<EOF
[
  {
    "index": null,
    "name": "$instance_name",
    "trains": null,
    "conflicts": null,
    "avg_tracks": null,
    "conflicting_visit_pairs": null,
    "delay_cost_type": "$OBJECTIVE",
    "solves": [
      {
        "solver_name": "$effective_solver",
        "status": "$external_status",
        "external_status": "$external_status",
        "exit_code": $status,
        "sol_time": $elapsed_ms,
        "total_time": $elapsed_seconds
      }
    ]
  }
]
EOF
}

for prefix in orig track station; do
  for infra in A B; do
    for number in $(seq 1 12); do
      run_one "${prefix}${infra}${number}"
    done
  done
done

python3 - "$tmpdir" "$order_file" "$JSON_OUT" <<'PY'
import json
import sys
from pathlib import Path

tmpdir = Path(sys.argv[1])
order_file = Path(sys.argv[2])
json_out = Path(sys.argv[3])

items = []
for line in order_file.read_text().splitlines():
    if not line:
        continue
    data = json.loads((tmpdir / f"{line}.json").read_text())
    if isinstance(data, list):
        items.extend(data)
    else:
        items.append(data)

for idx, item in enumerate(items):
    item["index"] = idx

json_out.write_text(json.dumps(items, indent=2))
print(f"Wrote {json_out}")
PY
