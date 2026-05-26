#!/usr/bin/env python3
import argparse
import csv
import html
import math
from pathlib import Path


OBJECTIVES = ["cont", "infsteps180", "finsteps123"]
METRICS = [
    ("time_to_best_value_ms", "Time to Best Value (ms)"),
    ("time_to_prove_best_value_ms", "Time to Prove Best Value (ms)"),
]

SERIES = [
    ("bigm_lazy", "BigMLazy", "#1f77b4", "circle"),
    ("mip_ddd", "MipDdd", "#2ca02c", "square"),
]

SAT_KEY = "sat_sc"
SAT_LABEL = "SatDddSc"


def parse_optional_float(value):
    if value in (None, ""):
        return None
    return float(value)


def parse_optional_int(value):
    if value in (None, ""):
        return None
    return int(float(value))


def load_csv(path: Path):
    rows = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["name"]] = row
    return rows


def metric_ticks(min_value, max_value):
    low_exp = math.floor(math.log10(min_value))
    high_exp = math.ceil(math.log10(max_value))
    ticks = []
    for exp in range(low_exp, high_exp + 1):
        ticks.append(10 ** exp)
    return ticks


def build_points(rows_by_solver, metric):
    sat_rows = rows_by_solver[SAT_KEY]
    names = sorted(set().union(*(set(rows.keys()) for rows in rows_by_solver.values())))
    points = []
    for name in names:
        sat_row = sat_rows.get(name)
        if sat_row is None:
            continue
        sat_metric = parse_optional_float(sat_row.get(metric))
        if sat_metric is None or sat_metric <= 0:
            continue

        cost_values = []
        for solver_rows in rows_by_solver.values():
            row = solver_rows.get(name)
            if row is None:
                continue
            cost = parse_optional_int(row.get("cost"))
            if cost is not None:
                cost_values.append(cost)
        consistent_cost = len(set(cost_values)) == 1 if cost_values else False

        for solver_key, solver_label, color, shape in SERIES:
            row = rows_by_solver[solver_key].get(name)
            if row is None:
                continue
            solver_metric = parse_optional_float(row.get(metric))
            if solver_metric is None or solver_metric <= 0:
                continue
            points.append(
                {
                    "name": name,
                    "solver_key": solver_key,
                    "solver_label": solver_label,
                    "color": color,
                    "shape": shape,
                    "x": sat_metric,
                    "y": solver_metric,
                    "sat_cost": parse_optional_int(sat_row.get("cost")),
                    "solver_cost": parse_optional_int(row.get("cost")),
                    "sat_status": sat_row.get("status"),
                    "solver_status": row.get("status"),
                    "consistent_cost": consistent_cost,
                }
            )
    return points


def value_to_px(value, min_value, max_value, length):
    log_min = math.log10(min_value)
    log_max = math.log10(max_value)
    return (math.log10(value) - log_min) / (log_max - log_min) * length


def draw_marker(point, cx, cy):
    title = (
        f"{point['name']} | {point['solver_label']} vs {SAT_LABEL}\n"
        f"SAT metric={point['x']:.3f} ms, solver metric={point['y']:.3f} ms\n"
        f"SAT cost={point['sat_cost']} ({point['sat_status']}), "
        f"solver cost={point['solver_cost']} ({point['solver_status']})\n"
        f"consistent_cost={point['consistent_cost']}"
    )
    title = html.escape(title)

    fill = point["color"] if point["consistent_cost"] else "white"
    stroke = point["color"] if point["consistent_cost"] else "#d62728"
    stroke_width = 1.5

    if point["shape"] == "square":
        size = 5
        shape = (
            f'<rect x="{cx - size}" y="{cy - size}" width="{2 * size}" height="{2 * size}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}">'
            f"<title>{title}</title></rect>"
        )
    else:
        shape = (
            f'<circle cx="{cx}" cy="{cy}" r="5" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}">'
            f"<title>{title}</title></circle>"
        )
    return shape


def make_svg(points, metric_key, metric_label, objective, out_path: Path):
    width = 900
    height = 720
    margin_left = 95
    margin_right = 35
    margin_top = 60
    margin_bottom = 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    values = [point["x"] for point in points] + [point["y"] for point in points]
    min_value = min(values)
    max_value = max(values)
    min_value = 10 ** math.floor(math.log10(min_value))
    max_value = 10 ** math.ceil(math.log10(max_value))
    ticks = metric_ticks(min_value, max_value)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="22" font-family="Arial">Scatter by Instance: {objective}</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-size="14" fill="#444" font-family="Arial">{metric_label} | x = {SAT_LABEL}, y = comparator</text>',
    ]

    x0 = margin_left
    y0 = height - margin_bottom

    parts.append(f'<rect x="{x0}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222"/>')

    for tick in ticks:
        offset = value_to_px(tick, min_value, max_value, plot_w)
        x = x0 + offset
        y = y0 - value_to_px(tick, min_value, max_value, plot_h)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{y0}" stroke="#eee"/>')
        parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_w}" y2="{y:.2f}" stroke="#eee"/>')
        label = f"{int(tick):,}"
        parts.append(f'<text x="{x:.2f}" y="{y0 + 22}" text-anchor="middle" font-size="12" fill="#444" font-family="Arial">{label}</text>')
        parts.append(f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#444" font-family="Arial">{label}</text>')

    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0 + plot_w}" y2="{margin_top}" stroke="#888" stroke-dasharray="6 6" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{x0 + plot_w - 6}" y="{margin_top + 16}" text-anchor="end" font-size="12" fill="#666" font-family="Arial">y = x</text>'
    )

    for point in points:
        cx = x0 + value_to_px(point["x"], min_value, max_value, plot_w)
        cy = y0 - value_to_px(point["y"], min_value, max_value, plot_h)
        parts.append(draw_marker(point, cx, cy))

    parts.append(
        f'<text x="{x0 + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-size="15" font-family="Arial">{SAT_LABEL} {metric_label}</text>'
    )
    parts.append(
        f'<text x="24" y="{margin_top + plot_h / 2}" transform="rotate(-90 24 {margin_top + plot_h / 2})" text-anchor="middle" font-size="15" font-family="Arial">Comparator {metric_label}</text>'
    )

    legend_x = x0 + 12
    legend_y = margin_top + 14
    for idx, (_, label, color, shape) in enumerate(SERIES):
        y = legend_y + idx * 22
        sample = {
            "solver_label": label,
            "name": label,
            "x": 1.0,
            "y": 1.0,
            "sat_cost": None,
            "solver_cost": None,
            "sat_status": "ok",
            "solver_status": "ok",
            "consistent_cost": True,
            "color": color,
            "shape": shape,
        }
        parts.append(draw_marker(sample, legend_x + 8, y - 4))
        parts.append(f'<text x="{legend_x + 22}" y="{y}" font-size="13" font-family="Arial">{label}</text>')
    parts.append(
        f'<circle cx="{legend_x + 8}" cy="{legend_y + 44 - 4}" r="5" fill="white" stroke="#d62728" stroke-width="1.5"/>'
        f'<text x="{legend_x + 22}" y="{legend_y + 44}" font-size="13" font-family="Arial">Inconsistent cost</text>'
    )

    consistent_count = sum(1 for point in points if point["consistent_cost"])
    point_count = len(points)
    parts.append(
        f'<text x="{x0 + plot_w}" y="{height - 56}" text-anchor="end" font-size="12" fill="#444" font-family="Arial">points={point_count}, consistent={consistent_count}</text>'
    )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(
        description="Generate SVG scatter plots comparing BigM/MIP vs SAT by instance."
    )
    parser.add_argument(
        "--out-dir",
        default="plots/scatter_instance",
        help="Output directory for SVG plots",
    )
    parser.add_argument(
        "--objective",
        action="append",
        dest="objectives",
        choices=OBJECTIVES,
        help="Objective(s) to plot; default is all",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    objectives = args.objectives or OBJECTIVES

    for objective in objectives:
        rows_by_solver = {
            "bigm_lazy": load_csv(Path(f"results_bigm_lazy_{objective}_120s.csv")),
            "mip_ddd": load_csv(Path(f"results_mip_ddd_{objective}_120s.csv")),
            SAT_KEY: load_csv(Path(f"results_sat_sc_{objective}_bit_totalizer_120s.csv")),
        }

        for metric_key, metric_label in METRICS:
            points = build_points(rows_by_solver, metric_key)
            if not points:
                print(f"Skipping {objective} {metric_key}: no comparable points")
                continue
            out_path = out_dir / f"scatter_{objective}_{metric_key}.svg"
            make_svg(points, metric_key, metric_label, objective, out_path)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
