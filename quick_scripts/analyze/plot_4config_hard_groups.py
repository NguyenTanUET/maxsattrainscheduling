"""Generate three separate PDFs (one per objective) showing 4 MaxSAT configs
(Base, Default, SC-only, Prec-only) head-to-head on the track/station hard
groups. Mean sol_time on common-solved subset, log scale, no titles
(captions provided by LaTeX).

Outputs:
  img/maxsat_compare_step.pdf    -- Bậc thang
  img/maxsat_compare_round.pdf   -- Tuyến tính làm tròn
  img/maxsat_compare_cont.pdf    -- Tuyến tính liên tục
"""
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / '2026-05-16-Verified-Result-For-Graduation-Thesis'

CONFIGS = [
    ('MaxSAT-Base',    'maxsat_baseline/maxsat_ddd_ladder_{obj}.csv',  '#888888'),
    ('MaxSAT-Default', 'maxsat_verify/MaxsatScDefault_{obj}.csv',     '#1f77b4'),
    ('MaxSAT-SC',      'maxsat_ablation/MaxsatOnlySc_{obj}.csv',      '#2ca02c'),
    ('MaxSAT-Prec',    'maxsat_ablation/MaxsatOnlyPrec_{obj}.csv',     '#ff7f0e'),
]

OBJECTIVES = [
    ('finsteps123', 'step'),
    ('infsteps180', 'round'),
    ('cont',        'cont'),
]

GROUPS = [
    ('track',   lambda n: n.startswith('track')),
    ('station', lambda n: n.startswith('station')),
]


def load_csv(path):
    with open(path) as f:
        return {r['name']: r for r in csv.DictReader(f)}


def compute_means_for_obj(obj_key):
    """For (group, config): mean sol_time on common-solved subset."""
    data = {cfg: load_csv(ROOT / path.format(obj=obj_key))
            for cfg, path, _ in CONFIGS}
    out = {}
    for group_name, group_filter in GROUPS:
        common = None
        for d in data.values():
            ok = {n for n, r in d.items()
                  if group_filter(n) and r['status'] == 'ok'}
            common = ok if common is None else common & ok
        for cfg, _, _ in CONFIGS:
            times = [float(data[cfg][n]['sol_time']) for n in common]
            out[(group_name, cfg)] = (
                statistics.mean(times) if times else 0, len(common))
    return out


def plot_objective(obj_key, slug):
    means = compute_means_for_obj(obj_key)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bar_width = 0.18
    x = np.arange(len(GROUPS))

    for cfg_i, (cfg, _, color) in enumerate(CONFIGS):
        heights = [means[(g, cfg)][0] for g, _ in GROUPS]
        offsets = (cfg_i - 1.5) * bar_width
        bars = ax.bar(x + offsets, heights, bar_width,
                      label=cfg, color=color, edgecolor='black', linewidth=0.4)
        for b, h in zip(bars, heights):
            if h > 0:
                ax.annotate(f'{h:.0f}', xy=(b.get_x() + b.get_width()/2, h),
                            xytext=(0, 2), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g}\n({means[(g, "MaxSAT-Base")][1]} bài)'
                        for g, _ in GROUPS], fontsize=10)
    ax.set_yscale('log')
    ax.set_ylabel('Thời gian giải trung bình (ms)', fontsize=10)
    ax.grid(axis='y', which='both', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=9, frameon=True, framealpha=0.9)

    fig.tight_layout()
    out_pdf = Path(f'D:/github/KLTN_TRP_Final/img/maxsat_compare_{slug}.pdf')
    out_png = out_pdf.with_suffix('.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_pdf.name}')


def main():
    for obj_key, slug in OBJECTIVES:
        print(f'Plotting {obj_key}...')
        plot_objective(obj_key, slug)


if __name__ == '__main__':
    main()
