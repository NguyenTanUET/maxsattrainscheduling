"""Generate summary table giống format thesis Excel (Objective × Set × Method).

Output:
  results_combined/<timestamp>/summary_table.csv  (Excel-friendly)
  results_combined/<timestamp>/summary_table.md   (Markdown for thesis)

Cấu trúc:
  Cột: Average time (ms) | #Solved, mỗi cụm 5 methods
  Hàng: 3 Objective (Linear Continuous / Rounded Linear / Stepwise)
        × 4 Set (Original / Track time / Station time / All)
"""
from __future__ import annotations
import csv, os, sys

LATEST = os.popen("ls -td results_combined/*/ | head -1").read().strip()
G_DIR = "2026-05-16-Verified-Result-For-Graduation-Thesis/BigM_MILP"

COST_DISPLAY = {
    'cont': 'Linear Continuous',
    'infsteps180': 'Rounded Linear',
    'finsteps123': 'Stepwise',
}
SET_PREFIX = {
    'orig': 'Original',
    'track': 'Track time',
    'addtrack': 'Track time',
    'station': 'Station time',
    'addstation': 'Station time',
}

METHODS = [
    ('BigM (Gurobi)',  f"{G_DIR}/results_bigm_lazy_{{cost}}_120s.csv"),
    ('TI (Gurobi)',    f"{G_DIR}/results_mip_ddd_{{cost}}_120s.csv"),
    ('BigM (CPLEX)',   f"{LATEST}/results_bigm_{{cost}}_120s.csv"),
    ('TI (CPLEX)',     f"{LATEST}/results_ti_{{cost}}_120s.csv"),
    ('CP (CPLEX)',     f"{LATEST}/results_cp_{{cost}}_120s.csv"),
]
SETS = ['Original', 'Track time', 'Station time', 'All']
COSTS = ['cont', 'infsteps180', 'finsteps123']


def normalize(name):
    return name.replace("addstation", "station").replace("addtrack", "track")


def load(path):
    if not os.path.exists(path):
        return []
    rows = []
    for r in csv.DictReader(open(path)):
        nname = normalize(r['name'])
        prefix = nname.rstrip('0123456789').rstrip('ABCDEFGHIJ')
        r['_set'] = SET_PREFIX.get(prefix, 'Unknown')
        rows.append(r)
    return rows


def aggregate(rows, set_filter=None):
    """Tính avg_time(ms) over PROVED instances + #solved."""
    if set_filter is not None:
        rows = [r for r in rows if r['_set'] == set_filter]
    n_solved = 0
    time_sum = 0.0
    for r in rows:
        try:
            lb, ub = float(r['lb']), float(r['ub'])
            if r['status'] == 'ok' and abs(lb - ub) < 0.01:
                n_solved += 1
                t = float(r.get('solver_time', '') or 0)
                time_sum += t * 1000
        except (ValueError, KeyError):
            continue
    avg_ms = time_sum / n_solved if n_solved > 0 else 0
    return avg_ms, n_solved


# Build all cell values
table = {}  # (cost, set, method_idx) -> (avg_ms, n_solved)
for cost in COSTS:
    for i, (label, tmpl) in enumerate(METHODS):
        rows = load(tmpl.format(cost=cost))
        for s in SETS:
            sf = s if s != 'All' else None
            table[(cost, s, i)] = aggregate(rows, sf)

# Write CSV (Excel-friendly with merged-style cells)
csv_path = f"{LATEST}/summary_table.csv"
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    # Title row
    w.writerow([''] * 2 + ['120 seconds timeout'])
    # Top header: blank + Average time + #Solved
    w.writerow([''] * 2 + ['Average time (ms)'] + [''] * (len(METHODS) - 1)
               + ['#Solved'] + [''] * (len(METHODS) - 1))
    # Method names (2 groups)
    method_names = [m[0] for m in METHODS]
    w.writerow(['Objective function', 'Set'] + method_names + method_names)
    # Data rows
    for cost in COSTS:
        obj = COST_DISPLAY[cost]
        for j, s in enumerate(SETS):
            row = [obj if j == 0 else '', s]
            for i in range(len(METHODS)):
                ms, _ = table[(cost, s, i)]
                row.append(f"{ms:.0f}")
            for i in range(len(METHODS)):
                _, ns = table[(cost, s, i)]
                row.append(str(ns))
            w.writerow(row)

# Write Markdown
md_path = f"{LATEST}/summary_table.md"
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# Summary Table (120s timeout)\n\n")
    f.write("**Solved = proved optimal (lb = ub).** Avg time tính trên instances solved.\n\n")
    # Avg time table
    f.write("## Average time (ms) — proved-optimal instances\n\n")
    hdrs = ['Objective', 'Set'] + [m[0] for m in METHODS]
    f.write("| " + " | ".join(hdrs) + " |\n")
    f.write("|" + "|".join(['---'] * len(hdrs)) + "|\n")
    for cost in COSTS:
        obj = COST_DISPLAY[cost]
        for j, s in enumerate(SETS):
            cells = [obj if j == 0 else '', s]
            for i in range(len(METHODS)):
                ms, _ = table[(cost, s, i)]
                cells.append(f"{ms:,.0f}")
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("|" + "|".join([''] * len(hdrs)) + "|\n")  # blank separator
    # #Solved table
    f.write("\n## #Solved (proved optimal)\n\n")
    f.write("| " + " | ".join(hdrs) + " |\n")
    f.write("|" + "|".join(['---'] * len(hdrs)) + "|\n")
    for cost in COSTS:
        obj = COST_DISPLAY[cost]
        for j, s in enumerate(SETS):
            cells = [obj if j == 0 else '', s]
            for i in range(len(METHODS)):
                _, ns = table[(cost, s, i)]
                cells.append(str(ns))
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("|" + "|".join([''] * len(hdrs)) + "|\n")

print(f"Written:")
print(f"  {csv_path}")
print(f"  {md_path}")
print(f"\nMở .csv bằng Excel → table giống image. Mở .md → preview.")
