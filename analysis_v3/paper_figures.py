"""Figures of the article, drawn from the results of record (never from typed numbers).

Inputs (same files as paper_fullcorpus.py):
  results/v2/DIPSER_grid_v1_metrics.json    slice grid (495 fits)
  results/v9/results/v3b/gate3.json         full-release grid, person-bootstrap CIs
  results/v9/results/logs/label_prior.txt   label-only references
  results/v9/results/v4/split_leakage.json  random vs person-grouped split
  results/v10/summary.json                  person-grouped vs leave-one-class-out
  results/v11/phase10a.json                 sparse-probe tracking

Run from the repository root:  python analysis_v3/paper_figures.py
Output: paper/figures/fig_{slice_grid,full_release,protocol,probe}.pdf
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
K = R / "v9" / "results"
OUT = ROOT / "paper" / "figures"

BLUE = "#257de0"      # series 1
ORANGE = "#f47230"    # series 2
GREY = "#8a8a8a"      # references and controls (not a data series)
INK = "#191919"
INK2 = "#666666"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

COL = 3.45  # IEEE single-column width in inches


def load(path):
    return json.loads(Path(path).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / (name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("wrote", OUT / (name + ".pdf"))


def fig_slice_grid():
    rows = load(R / "v2" / "DIPSER_grid_v1_metrics.json")["summary"]["rows"]
    groups = [("Single streams", [], BLUE), ("Gate fusion", [], ORANGE),
              ("Core fusion", [], ORANGE), ("Permuted labels", [], GREY)]
    for row in rows:
        if row.get("shuffle"):
            groups[3][1].append(row)
        elif row["fusion"] == "unimodal":
            groups[0][1].append(row)
        elif row["fusion"] == "gate":
            groups[1][1].append(row)
        else:
            groups[2][1].append(row)
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for y, (name, items, colour) in enumerate(groups):
        items = sorted(items, key=lambda r: r["qwk_mean"])
        n = len(items)
        for i, row in enumerate(items):
            offset = 0.0 if n == 1 else (i / (n - 1) - 0.5) * 0.56
            ax.errorbar(row["qwk_mean"], y + offset, xerr=row["qwk_std"], fmt="o", ms=3.2,
                        color=colour, ecolor=colour, elinewidth=0.5, alpha=0.9,
                        markeredgecolor="white", markeredgewidth=0.4)
    best_single = max(groups[0][1], key=lambda r: r["qwk_mean"])
    ax.axvline(best_single["qwk_mean"], color=BLUE, lw=0.6, ls=(0, (3, 2)))
    ax.annotate("best single stream\n(face mesh, soft)", (best_single["qwk_mean"], 3.0),
                xytext=(4, 0), textcoords="offset points", fontsize=7, va="center", color=INK)
    ax.axvline(0.0, color=INK2, lw=0.5)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(["%s (%d)" % (g[0], len(g[1])) for g in groups])
    ax.set_ylim(3.5, -0.5)
    ax.set_xlim(-0.33, 0.52)
    ax.set_xlabel("QWK (mean over 3 seeds, bars: s.d.)")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_slice_grid")


def label_refs():
    text = (K / "logs" / "label_prior.txt").read_text()
    exp = float(re.search(r"experiment-prior \(leave-person-out\) QWK ([0-9.]+)", text).group(1))
    mom = float(re.search(r"class-moment prior .* QWK ([0-9.]+)", text).group(1))
    return exp, mom


def fig_full_release():
    gate = load(K / "v3b" / "gate3.json")
    point, ci = gate["point"], gate["ci"]
    names = {"facemesh": "Face mesh", "head_pose": "Head pose", "physio": "Physiology",
             "face": "Whole frame"}
    bars = []
    for stream in ("facemesh", "head_pose", "physio", "face"):
        key = gate["best_stream"][stream]
        bars.append((names[stream], point[key]["qwk"], ci[key]["qwk"], BLUE))
    key = gate["best_fusion"]
    bars.append(("Best fusion", point[key]["qwk"], ci[key]["qwk"], ORANGE))
    exp, mom = label_refs()
    bars.append(("Label-only: same lesson", exp, None, GREY))
    bars.append(("Label-only: same moment", mom, None, GREY))
    control = max(point[k]["qwk"] for k in point if k.startswith("classmate_unimodal"))
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for y, (name, value, interval, colour) in enumerate(bars):
        ax.barh(y, value, height=0.55, color=colour)
        if interval:
            ax.plot(interval, [y, y], color=INK, lw=0.8)
        right = interval[1] if interval else value
        ax.text(right + 0.008, y, "%.3f" % value, va="center", fontsize=7, color=INK)
    ax.axvline(control, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.text(control + 0.004, -0.72, "classmate-label control", fontsize=6.5, color=INK2, va="center")
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars])
    ax.set_ylim(len(bars) - 0.4, -1.0)
    ax.set_xlim(0, 0.45)
    ax.set_xlabel("QWK, held-out students (lines: 95% CI)")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_full_release")


def fig_protocol():
    leak = load(K / "v4" / "split_leakage.json")
    arms = load(R / "v10" / "summary.json")["arms"]
    majority = leak["random stratified (windows)"]["majority_acc"] * 100
    rows = [
        ("A, random windows", leak["random stratified (windows)"], GREY),
        ("A, held-out students", leak["person-grouped stratified"], BLUE),
        ("B, held-out students", arms["pg5_own"], BLUE),
        ("B, held-out class", arms["logo_own"], ORANGE),
        ("B$'$, held-out students", arms["pg5_own_class"], BLUE),
        ("B$'$, held-out class", arms["logo_own_class"], ORANGE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for y, (name, arm, colour) in enumerate(rows):
        qwk, acc = arm["qwk"][0], arm["acc"][0] * 100
        axes[0].barh(y, qwk, height=0.55, color=colour)
        axes[0].text(qwk + 0.008, y, "%.3f" % qwk, va="center", fontsize=7)
        axes[1].plot([acc], [y], "o", ms=4.5, color=colour)   # dots: the axis does not start at 0
        axes[1].text(acc + 0.9, y, "%.1f" % acc, va="center", fontsize=7)
    axes[1].axvline(majority, color=INK2, lw=0.6, ls=(0, (3, 2)))
    axes[1].text(majority, -0.85, "majority", fontsize=6.5, color=INK2, ha="center", va="center")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[0].set_ylim(len(rows) - 0.4, -1.1)
    axes[0].set_xlim(0, 0.45)
    axes[1].set_xlim(40, 72)
    axes[0].set_xlabel("QWK")
    axes[1].set_xlabel("Accuracy (%)")
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    save(fig, "fig_protocol")


def fig_probe():
    probe = load(R / "v11" / "phase10a.json")
    keys = [("P8", "32 s"), ("P15", "1 min"), ("P38", "2.5 min")]
    xs = range(len(keys))
    persist = [probe[k]["full_vs_persist"]["a_acc"] * 100 for k, _ in keys]
    full = [probe[k]["full_vs_persist"]["b_acc"] * 100 for k, _ in keys]
    sensors = [probe[k]["sensors_only"]["acc"] * 100 for k, _ in keys]
    majority = probe["majority_acc"] * 100
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    ax.plot(xs, full, "-o", color=BLUE, lw=1.4, ms=4, label="Probe + sensors")
    ax.plot(xs, persist, "-s", color=ORANGE, lw=1.4, ms=4, label="Persistence (last probe)")
    ax.plot(xs, sensors, "-^", color=GREY, lw=1.2, ms=4, label="Sensors only")
    ax.axhline(majority, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.text(0, majority - 1.8, "majority", fontsize=6.5, color=INK2, ha="center")
    ax.text(0.06, full[0] + 0.4, "%.1f" % full[0], fontsize=7, va="bottom")
    ax.text(0.0, persist[0] - 1.6, "%.1f" % persist[0], fontsize=7, va="top", ha="center")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for _, label in keys])
    end = len(keys) - 1
    ax.text(end + 0.08, full[-1] + 0.3, "Probe + sensors", fontsize=7, va="bottom")
    ax.text(end + 0.08, persist[-1] + 0.6, "Persistence\n(last probe)", fontsize=7, va="top")
    ax.text(end + 0.08, sensors[-1], "Sensors only", fontsize=7, va="center")
    ax.set_xlim(-0.25, len(keys) - 0.15)
    ax.set_ylim(46, 78)
    ax.set_xlabel("Interval between label probes")
    ax.set_ylabel("Accuracy (%)")
    save(fig, "fig_probe")


if __name__ == "__main__":
    fig_slice_grid()
    fig_full_release()
    fig_protocol()
    fig_probe()
