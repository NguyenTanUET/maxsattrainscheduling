#!/usr/bin/env python3
"""Generate chapter 4 tables (4.4-4.10) from verified raw CSVs.

Reads CSVs from 2026-05-16-Verified-Result-For-Graduation-Thesis/
Outputs 7 CSV files in chapter4_tables/

Tables produced:
  4.4  tab_4_4_result_aggregate.csv          # Tổng quan: #Solved + avg time
  4.5  tab_4_5_result_hard_step.csv          # Hard 10 instances, mục tiêu Step
  4.6  tab_4_6_result_hard_round.csv         # Hard 10, mục tiêu Round
  4.7  tab_4_7_result_hard_cont.csv          # Hard 10, mục tiêu Cont
  4.8  tab_4_8_result_gap.csv                # GAP% timeout instances
  4.9  tab_4_9_ablation_headtohead.csv       # 4 MaxSAT variants × track/station × 3 obj
  4.10 tab_4_10_ablation_cnf_structure.csv   # V, C, I × 4 variants × 3 obj
"""
import csv
import os
from collections import OrderedDict

ROOT = "d:/github/maxsattrainscheduling/2026-05-16-Verified-Result-For-Graduation-Thesis"
OUT = "d:/github/maxsattrainscheduling/chapter4_tables"

os.makedirs(OUT, exist_ok=True)

# ─── Source CSV mapping ──────────────────────────────────────────────────
SOLVERS = OrderedDict([
    ("BigM",        {"step": f"{ROOT}/BigM_MILP/results_bigm_lazy_finsteps123_120s.csv",
                     "round": f"{ROOT}/BigM_MILP/results_bigm_lazy_infsteps180_120s.csv",
                     "cont":  f"{ROOT}/BigM_MILP/results_bigm_lazy_cont_120s.csv"}),
    ("TI",          {"step": f"{ROOT}/BigM_MILP/results_mip_ddd_finsteps123_120s.csv",
                     "round": f"{ROOT}/BigM_MILP/results_mip_ddd_infsteps180_120s.csv",
                     "cont":  f"{ROOT}/BigM_MILP/results_mip_ddd_cont_120s.csv"}),
    ("MS-Base",     {"step": f"{ROOT}/maxsat_baseline/maxsat_ddd_ladder_finsteps123.csv",
                     "round": f"{ROOT}/maxsat_baseline/maxsat_ddd_ladder_infsteps180.csv",
                     "cont":  f"{ROOT}/maxsat_baseline/maxsat_ddd_ladder_cont.csv"}),
    ("MS-Def",      {"step": f"{ROOT}/maxsat_verify/MaxsatSclDefault_finsteps123.csv",
                     "round": f"{ROOT}/maxsat_verify/MaxsatSclDefault_infsteps180.csv",
                     "cont":  f"{ROOT}/maxsat_verify/MaxsatSclDefault_cont.csv"}),
    ("IS-Def",      {"step": f"{ROOT}/inc_totalizer_verify/IncTotDefault_finsteps123.csv",
                     "round": f"{ROOT}/inc_totalizer_verify/IncTotDefault_infsteps180.csv",
                     "cont":  f"{ROOT}/inc_totalizer_verify/IncTotDefault_cont.csv"}),
    ("IS-Not",      {"step": f"{ROOT}/inc_totalizer_verify/IncTotNothing_finsteps123.csv",
                     "round": f"{ROOT}/inc_totalizer_verify/IncTotNothing_infsteps180.csv",
                     "cont":  f"{ROOT}/inc_totalizer_verify/IncTotNothing_cont.csv"}),
    ("PS-Def",      {"step": f"{ROOT}/puresat_verify/PureDefault_finsteps123.csv",
                     "round": f"{ROOT}/puresat_verify/PureDefault_infsteps180.csv",
                     "cont":  f"{ROOT}/puresat_verify/PureDefault_cont.csv"}),
    ("PS-Not",      {"step": f"{ROOT}/puresat_verify/PureNothing_finsteps123.csv",
                     "round": f"{ROOT}/puresat_verify/PureNothing_infsteps180.csv",
                     "cont":  f"{ROOT}/puresat_verify/PureNothing_cont.csv"}),
])

OBJECTIVES = ["step", "round", "cont"]

HARD_INSTANCES = [
    "trackA1", "trackA2", "trackA8", "trackA11", "trackA12",
    "stationA1", "stationA2", "stationA8", "stationA11", "stationA12",
]

# Ablation: 4 MaxSAT variants for tables 4.9 + 4.10
MAXSAT_VARIANTS = OrderedDict([
    ("MS-Base",    {"step": f"{ROOT}/maxsat_baseline/maxsat_ddd_ladder_finsteps123.csv",
                    "round": f"{ROOT}/maxsat_baseline/maxsat_ddd_ladder_infsteps180.csv",
                    "cont":  f"{ROOT}/maxsat_baseline/maxsat_ddd_ladder_cont.csv"}),
    ("MS-Def",     {"step": f"{ROOT}/maxsat_verify/MaxsatSclDefault_finsteps123.csv",
                    "round": f"{ROOT}/maxsat_verify/MaxsatSclDefault_infsteps180.csv",
                    "cont":  f"{ROOT}/maxsat_verify/MaxsatSclDefault_cont.csv"}),
    ("MS-SC",      {"step": f"{ROOT}/maxsat_ablation/MaxsatOnlyScl_finsteps123.csv",
                    "round": f"{ROOT}/maxsat_ablation/MaxsatOnlyScl_infsteps180.csv",
                    "cont":  f"{ROOT}/maxsat_ablation/MaxsatOnlyScl_cont.csv"}),
    ("MS-Prec",    {"step": f"{ROOT}/maxsat_ablation/MaxsatOnlyPrec_finsteps123.csv",
                    "round": f"{ROOT}/maxsat_ablation/MaxsatOnlyPrec_infsteps180.csv",
                    "cont":  f"{ROOT}/maxsat_ablation/MaxsatOnlyPrec_cont.csv"}),
])

# CNF check files for table 4.10 — these have num_vars_total / num_clauses_total columns
CNF_CHECK = OrderedDict([
    ("MS-Base",    {"step": f"{ROOT}/maxsat_clause_var_check/MaxsatBaseline_finsteps123.csv",
                    "round": f"{ROOT}/maxsat_clause_var_check/MaxsatBaseline_infsteps180.csv",
                    "cont":  f"{ROOT}/maxsat_clause_var_check/MaxsatBaseline_cont.csv"}),
    ("MS-Def",     {"step": f"{ROOT}/maxsat_clause_var_check/MaxsatDefault_finsteps123.csv",
                    "round": f"{ROOT}/maxsat_clause_var_check/MaxsatDefault_infsteps180.csv",
                    "cont":  f"{ROOT}/maxsat_clause_var_check/MaxsatDefault_cont.csv"}),
    ("MS-SC",      {"step": f"{ROOT}/maxsat_clause_var_check/MaxsatSCL_finsteps123.csv",
                    "round": f"{ROOT}/maxsat_clause_var_check/MaxsatSCL_infsteps180.csv",
                    "cont":  f"{ROOT}/maxsat_clause_var_check/MaxsatSCL_cont.csv"}),
    ("MS-Prec",    {"step": f"{ROOT}/maxsat_clause_var_check/MaxsatPrec_finsteps123.csv",
                    "round": f"{ROOT}/maxsat_clause_var_check/MaxsatPrec_infsteps180.csv",
                    "cont":  f"{ROOT}/maxsat_clause_var_check/MaxsatPrec_cont.csv"}),
])


def load_csv(path):
    """Load CSV → dict keyed by instance name."""
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            out[row["name"]] = row
    return out


def is_solved(row):
    """Status check — only 'ok' counts as solved."""
    return row.get("status", "") == "ok"


def is_timeout(row):
    """Timeout includes timeout + external_timeout + aborted."""
    return row.get("status", "") in ("timeout", "external_timeout", "aborted")


def is_oom(row):
    return row.get("status", "") == "oom"


def total_time_ms(row):
    """total_time field in CSV is seconds → convert to ms."""
    return float(row["total_time"]) * 1000.0


def fmt_ms(ms):
    """Format ms with thousands separator (vd 1234.5 → '1,234.5')."""
    if ms < 1000:
        return f"{ms:.1f}"
    return f"{ms:,.1f}"


# ─── Table 4.4: Aggregate ───────────────────────────────────────────────
def build_table_4_4():
    rows = []
    for solver, paths in SOLVERS.items():
        row = {"solver": solver}
        for obj in OBJECTIVES:
            data = load_csv(paths[obj])
            solved = [r for r in data.values() if is_solved(r)]
            n_solved = len(solved)
            n_total = len(data)
            avg_time = sum(total_time_ms(r) for r in solved) / n_solved if solved else 0
            row[f"{obj}_solved"] = f"{n_solved}/{n_total}"
            row[f"{obj}_avg_time_ms"] = fmt_ms(avg_time) if solved else "—"
        rows.append(row)

    out_path = f"{OUT}/tab_4_4_result_aggregate.csv"
    with open(out_path, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["solver",
            "step_solved", "step_avg_time_ms",
            "round_solved", "round_avg_time_ms",
            "cont_solved", "cont_avg_time_ms"])
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {out_path}")


# ─── Table 4.5/4.6/4.7: Hard 10 instances per objective ─────────────────
def build_hard_table(obj_key, label):
    """obj_key: step/round/cont, label: '4.5'/'4.6'/'4.7'."""
    # Load each solver's data for this objective
    solver_data = {s: load_csv(SOLVERS[s][obj_key]) for s in SOLVERS}

    rows = []
    for inst in HARD_INSTANCES:
        row = {"instance": inst}
        for solver in SOLVERS:
            r = solver_data[solver].get(inst)
            if r is None:
                row[solver] = "—"
            elif is_solved(r):
                row[solver] = fmt_ms(total_time_ms(r))
            elif is_oom(r):
                row[solver] = "OOM"
            elif is_timeout(r):
                row[solver] = "T/O"
            else:
                row[solver] = "?"
        rows.append(row)

    out_path = f"{OUT}/tab_{label.replace('.', '_')}_result_hard_{obj_key}.csv"
    with open(out_path, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["instance"] + list(SOLVERS.keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {out_path}")


# ─── Table 4.8: GAP% on timeout instances ───────────────────────────────
def gap_percent(row):
    """Compute GAP% from lb + ub. Return None if invalid."""
    try:
        lb = float(row["lb"])
        ub = float(row["ub"])
        if ub <= 0 or ub == lb:
            return 0.0
        # If status is OOM or no valid ub → undefined
        if is_oom(row):
            return None
        # Treat very large ub (like i32::MAX) as no valid ub
        if ub > 1e9:
            return None
        return (ub - lb) / ub * 100.0
    except (ValueError, KeyError):
        return None


def build_table_4_8():
    # 4 SAT/MaxSAT solvers
    solvers_4 = ["MS-Base", "MS-Def", "IS-Def", "PS-Def"]

    rows = []

    # Round objective: instances that ALL 4 timeout/oom
    round_data = {s: load_csv(SOLVERS[s]["round"]) for s in solvers_4}
    for inst in sorted(round_data["MS-Base"].keys()):
        all_failed = True
        for s in solvers_4:
            r = round_data[s].get(inst)
            if r is None or is_solved(r):
                all_failed = False
                break
        if not all_failed:
            continue
        row = {"objective": "Round", "instance": inst}
        for s in solvers_4:
            r = round_data[s][inst]
            if is_oom(r):
                row[s] = "—"
            else:
                gp = gap_percent(r)
                row[s] = f"{gp:.1f}%" if gp is not None else "—"
        rows.append(row)

    # Cont objective: instances that ALL 4 timeout/oom
    cont_data = {s: load_csv(SOLVERS[s]["cont"]) for s in solvers_4}
    for inst in sorted(cont_data["MS-Base"].keys()):
        all_failed = True
        for s in solvers_4:
            r = cont_data[s].get(inst)
            if r is None or is_solved(r):
                all_failed = False
                break
        if not all_failed:
            continue
        row = {"objective": "Cont", "instance": inst}
        for s in solvers_4:
            r = cont_data[s][inst]
            if is_oom(r):
                row[s] = "—"
            else:
                gp = gap_percent(r)
                row[s] = f"{gp:.1f}%" if gp is not None else "—"
        rows.append(row)

    out_path = f"{OUT}/tab_4_8_result_gap.csv"
    with open(out_path, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["objective", "instance"] + solvers_4)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {out_path}")


# ─── Table 4.9: 4 MaxSAT variants × track/station × 3 objectives ────────
def build_table_4_9():
    # Load all data
    all_data = {}
    for variant, paths in MAXSAT_VARIANTS.items():
        all_data[variant] = {obj: load_csv(paths[obj]) for obj in OBJECTIVES}

    # Filter: instances where ALL 4 variants solved
    # Compute avg time per (variant, obj, group) where group = track / station
    def is_track(name):
        return name.startswith("track")

    def is_station(name):
        return name.startswith("station")

    rows = []
    for variant in MAXSAT_VARIANTS:
        row = {"variant": variant}
        for obj in OBJECTIVES:
            for group_name, group_fn in [("track", is_track), ("station", is_station)]:
                # Find instances all 4 variants solved in this obj
                common_solved = None
                for v in MAXSAT_VARIANTS:
                    solved_set = {n for n, r in all_data[v][obj].items()
                                  if is_solved(r) and group_fn(n)}
                    common_solved = solved_set if common_solved is None else (common_solved & solved_set)

                # Avg time for this variant on common solved instances
                times = [total_time_ms(all_data[variant][obj][n]) for n in common_solved]
                avg = sum(times) / len(times) if times else 0
                row[f"{obj}_{group_name}"] = fmt_ms(avg) if times else "—"
        rows.append(row)

    out_path = f"{OUT}/tab_4_9_ablation_headtohead.csv"
    with open(out_path, "w", newline='', encoding='utf-8') as f:
        fieldnames = ["variant"] + [f"{obj}_{grp}" for obj in OBJECTIVES for grp in ("track", "station")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {out_path}")


# ─── Table 4.10: CNF structure (V, C, I) × 4 variants × 3 obj ───────────
def build_table_4_10():
    all_data = {}
    for variant, paths in CNF_CHECK.items():
        all_data[variant] = {obj: load_csv(paths[obj]) for obj in OBJECTIVES}

    # Hard instances = track + station (filter by all 4 variants solved)
    def is_hard(name):
        return name.startswith("track") or name.startswith("station")

    rows = []
    for variant in CNF_CHECK:
        row = {"variant": variant}
        for obj in OBJECTIVES:
            common_solved = None
            for v in CNF_CHECK:
                solved_set = {n for n, r in all_data[v][obj].items()
                              if is_solved(r) and is_hard(n)}
                common_solved = solved_set if common_solved is None else (common_solved & solved_set)

            if not common_solved:
                row[f"{obj}_V"] = "—"
                row[f"{obj}_C"] = "—"
                row[f"{obj}_I"] = "—"
                continue

            vs, cs, is_ = [], [], []
            for n in common_solved:
                r = all_data[variant][obj][n]
                try:
                    vs.append(int(r["num_vars_total"]))
                    cs.append(int(r["num_clauses_total"]))
                    is_.append(int(r["iterations"]))
                except (KeyError, ValueError):
                    pass

            row[f"{obj}_V"] = f"{int(sum(vs) / len(vs)):,}" if vs else "—"
            row[f"{obj}_C"] = f"{int(sum(cs) / len(cs)):,}" if cs else "—"
            row[f"{obj}_I"] = f"{int(sum(is_) / len(is_))}" if is_ else "—"
        rows.append(row)

    # Add ratio rows
    base_row = {r["variant"]: r for r in rows}["MS-Base"]
    for variant in ["MS-Def", "MS-SC", "MS-Prec"]:
        cur = {r["variant"]: r for r in rows}[variant]
        ratio_row = {"variant": f"{variant} (%)"}
        for obj in OBJECTIVES:
            for col in ("V", "C", "I"):
                try:
                    cv = int(cur[f"{obj}_{col}"].replace(",", ""))
                    bv = int(base_row[f"{obj}_{col}"].replace(",", ""))
                    pct = (cv - bv) / bv * 100.0
                    ratio_row[f"{obj}_{col}"] = f"{pct:+.1f}"
                except (ValueError, ZeroDivisionError):
                    ratio_row[f"{obj}_{col}"] = "—"
        rows.append(ratio_row)

    out_path = f"{OUT}/tab_4_10_ablation_cnf_structure.csv"
    with open(out_path, "w", newline='', encoding='utf-8') as f:
        fieldnames = ["variant"] + [f"{obj}_{col}" for obj in OBJECTIVES for col in ("V", "C", "I")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    print("Building Chapter 4 tables from raw CSVs...")
    print()
    build_table_4_4()
    build_hard_table("step", "4.5")
    build_hard_table("round", "4.6")
    build_hard_table("cont", "4.7")
    build_table_4_8()
    build_table_4_9()
    build_table_4_10()
    print()
    print(f"All 7 tables written to: {OUT}/")
