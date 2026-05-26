#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  quick_scripts/debug/bisect_origA1_regression.sh <predicate>

Predicates:
  sat_sc
      Good if:
        sat_ddd_sc + scpb + no precedence + origA1 + finsteps123
      returns status=ok and cost=11.

  maxsat_sc
      Good if:
        maxsat_ddd_ladder_sc + no precedence + origA1 + finsteps123
      returns status=ok and cost=11.

Environment variables:
  EXPECTED_COST=11
      Expected objective value for the selected predicate.

  OBJECTIVE=finsteps123
      Objective passed to ddd.

  INSTANCE_NAME=origA1
      TXT instance short name filter.

  ACTIVATE_SCRIPT=
      Optional shell script sourced before build/run.
      Example: ACTIVATE_SCRIPT=quick_scripts/activation.txt

  CARGO_TARGET_DIR=target-bisect
      Separate target dir for bisect builds.

  SATCODER_DIR=../salvers/satcoder
      Path dependency directory to sanity-check before each run.

  SATCODER_EXPECT_REV=
      If set, require satcoder to be at this exact commit.

  SATCODER_AUTO_CHECKOUT=0
      If SATCODER_EXPECT_REV is set and this is 1, the script will try
      to checkout satcoder to that detached revision automatically.
      Otherwise the script exits 125 (skip) on mismatch.

  REQUIRE_CONSISTENT_BOUNDS=0
      If set to 1 for maxsat_sc, also require lb == ub == cost.

Examples:
  git bisect start HEAD 21a732d
  ACTIVATE_SCRIPT=quick_scripts/activation.txt \
    SATCODER_EXPECT_REV="$(git -C ../salvers/satcoder rev-parse HEAD)" \
    git bisect run quick_scripts/debug/bisect_origA1_regression.sh sat_sc

  ACTIVATE_SCRIPT=quick_scripts/activation.txt \
    SATCODER_EXPECT_REV="$(git -C ../salvers/satcoder rev-parse HEAD)" \
    REQUIRE_CONSISTENT_BOUNDS=1 \
    git bisect run quick_scripts/debug/bisect_origA1_regression.sh maxsat_sc

Exit codes for git bisect:
  0   good
  1   bad
  125 skip (build/run/environment not comparable)
EOF
}

log() {
  printf '[bisect-origA1] %s\n' "$*" >&2
}

skip() {
  log "skip: $*"
  exit 125
}

predicate="${1:-}"
case "${predicate}" in
  sat_sc|maxsat_sc)
    ;;
  -h|--help|"")
    usage
    exit 0
    ;;
  *)
    log "unknown predicate: ${predicate}"
    usage >&2
    exit 125
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"

expected_cost="${EXPECTED_COST:-11}"
objective="${OBJECTIVE:-finsteps123}"
instance_name="${INSTANCE_NAME:-origA1}"
activate_script="${ACTIVATE_SCRIPT:-}"
cargo_target_dir="${CARGO_TARGET_DIR:-target-bisect}"
satcoder_dir="${SATCODER_DIR:-${root_dir}/../salvers/satcoder}"
satcoder_expect_rev="${SATCODER_EXPECT_REV:-}"
satcoder_auto_checkout="${SATCODER_AUTO_CHECKOUT:-0}"
require_consistent_bounds="${REQUIRE_CONSISTENT_BOUNDS:-0}"

tmpdir="$(mktemp -d /tmp/ddd-bisect-origA1.XXXXXX)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT

build_log="${tmpdir}/build.log"
run_log="${tmpdir}/run.log"
json_out="${tmpdir}/result.json"

source_activate() {
  if [[ -n "${activate_script}" ]]; then
    if [[ ! -f "${activate_script}" ]]; then
      skip "activate script not found: ${activate_script}"
    fi
    # shellcheck disable=SC1090
    source "${activate_script}"
  fi
}

ensure_satcoder_revision() {
  if [[ -z "${satcoder_expect_rev}" ]]; then
    return
  fi

  if [[ ! -d "${satcoder_dir}/.git" ]]; then
    skip "satcoder dir is not a git repo: ${satcoder_dir}"
  fi

  local current_rev
  current_rev="$(git -C "${satcoder_dir}" rev-parse HEAD 2>/dev/null)" \
    || skip "cannot read satcoder revision"

  if [[ "${current_rev}" == "${satcoder_expect_rev}" ]]; then
    return
  fi

  if [[ "${satcoder_auto_checkout}" == "1" ]]; then
    git -C "${satcoder_dir}" checkout --detach "${satcoder_expect_rev}" \
      >/dev/null 2>&1 \
      || skip "failed to checkout satcoder revision ${satcoder_expect_rev}"
    current_rev="$(git -C "${satcoder_dir}" rev-parse HEAD 2>/dev/null)" \
      || skip "cannot verify satcoder revision after checkout"
    if [[ "${current_rev}" != "${satcoder_expect_rev}" ]]; then
      skip "satcoder revision mismatch after checkout: got ${current_rev}, want ${satcoder_expect_rev}"
    fi
  else
    skip "satcoder revision mismatch: got ${current_rev}, want ${satcoder_expect_rev}"
  fi
}

build_repo() {
  log "building predicate=${predicate} commit=$(git -C "${root_dir}" rev-parse --short HEAD)"
  (
    cd "${root_dir}"
    source_activate
    CARGO_TARGET_DIR="${cargo_target_dir}" cargo build --release --bin ddd
  ) >"${build_log}" 2>&1 || {
    tail -n 40 "${build_log}" >&2 || true
    skip "cargo build failed"
  }
}

bin_path() {
  if [[ "${cargo_target_dir}" = /* ]]; then
    printf '%s\n' "${cargo_target_dir}/release/ddd"
  else
    printf '%s\n' "${root_dir}/${cargo_target_dir}/release/ddd"
  fi
}

run_predicate() {
  local bin
  bin="$(bin_path)"

  local -a cmd=(
    "${bin}"
    --txt-instances
    --instance-name-filter "${instance_name}"
    --instance-name-exact
    --objective "${objective}"
    --json-output "${json_out}"
  )

  case "${predicate}" in
    sat_sc)
      cmd=(
        "${bin}"
        -s sat_ddd_sc
        --satddd-objective-encoding scpb
        --satddd-use-precedence-graph false
        --txt-instances
        --instance-name-filter "${instance_name}"
        --instance-name-exact
        --objective "${objective}"
        --json-output "${json_out}"
      )
      ;;
    maxsat_sc)
      cmd=(
        "${bin}"
        -s maxsat_ddd_ladder_sc
        --maxsat-ladder-sc-use-precedence-graph false
        --txt-instances
        --instance-name-filter "${instance_name}"
        --instance-name-exact
        --objective "${objective}"
        --json-output "${json_out}"
      )
      ;;
  esac

  (
    cd "${root_dir}"
    source_activate
    "${cmd[@]}"
  ) >"${run_log}" 2>&1 || {
    if rg -n "Could not resolve host: token\.gurobi\.com|WLSAccessID|LicenseID|libgurobi|Unknown .*flag|unknown option|unrecognized option|panic|thread 'main'.*panicked" "${run_log}" >/dev/null 2>&1; then
      tail -n 60 "${run_log}" >&2 || true
      skip "run failed for environment/compatibility reasons"
    fi
    tail -n 60 "${run_log}" >&2 || true
    skip "run failed without comparable JSON"
  }

  [[ -s "${json_out}" ]] || skip "solver finished without JSON output"
}

evaluate_json() {
  python3 - "${json_out}" "${instance_name}" "${expected_cost}" "${predicate}" "${require_consistent_bounds}" <<'PY'
import json
import sys

json_path, instance_name, expected_cost, predicate, require_consistent_bounds = sys.argv[1:]
expected_cost = int(expected_cost)
require_consistent_bounds = require_consistent_bounds == "1"

try:
    data = json.load(open(json_path))
except Exception as exc:
    print(f"skip: failed to read json: {exc}", file=sys.stderr)
    raise SystemExit(125)

if not isinstance(data, list):
    print("skip: json root is not a list", file=sys.stderr)
    raise SystemExit(125)

target = None
for item in data:
    if item.get("name") == instance_name:
        target = item
        break

if target is None:
    print(f"skip: instance {instance_name} not found in json", file=sys.stderr)
    raise SystemExit(125)

solves = target.get("solves") or []
if not solves:
    print(f"skip: instance {instance_name} has no solve rows", file=sys.stderr)
    raise SystemExit(125)

solve = solves[0]
status = solve.get("status")
cost = solve.get("cost")
lb = solve.get("lb")
ub = solve.get("ub")

if status != "ok":
    print(f"skip: solver status is {status!r}", file=sys.stderr)
    raise SystemExit(125)

if cost is None:
    print("skip: missing cost in solve row", file=sys.stderr)
    raise SystemExit(125)

if require_consistent_bounds and predicate == "maxsat_sc":
    if lb != cost or ub != cost:
        print(
            f"bad: cost={cost} lb={lb} ub={ub} (expected all equal to {expected_cost})",
            file=sys.stderr,
        )
        raise SystemExit(1)

if cost == expected_cost:
    if predicate == "maxsat_sc":
        print(f"good: maxsat_sc cost={cost} lb={lb} ub={ub}", file=sys.stderr)
    else:
        print(f"good: sat_sc cost={cost} lb={lb} ub={ub}", file=sys.stderr)
    raise SystemExit(0)

print(
    f"bad: predicate={predicate} cost={cost} lb={lb} ub={ub} expected_cost={expected_cost}",
    file=sys.stderr,
)
raise SystemExit(1)
PY
}

ensure_satcoder_revision
build_repo
run_predicate
evaluate_json
