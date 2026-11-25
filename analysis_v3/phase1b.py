"""Phase 1b (exploratory, post hoc): who-vs-when decomposition across runs and datasets.

    cd analysis_v3 && python phase1b.py [--corpus <corpus_full.npz>]

For every run: share of label variance between persons, and for the best config of each
single stream and the best fusion: QWK, between-person r (per-person mean prediction vs
mean label: does the model rank STUDENTS?) and within-person r (after removing each
person's means: does it follow changes WITHIN a student?).
Writes ../results/phase1b/whowhen.json and ../paper/figures/fig_whowhen.pdf (+ .png).
With --corpus it also adds the label-only decomposition of the full DIPSER corpus.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from whowhen import label_between_share, load_runs, summarise

ROOT = Path(__file__).resolve().parent.parent
RUNS = [
    ("DIPSER v2 (paper slice, session folds)", ROOT / "results/v2/raw_dipser/gtao_grid_v1"),
    ("DIPSER v3 fold (person folds, z-score)", ROOT / "results/v3/gtao_grid_v3_fold_person_strat"),
    ("DIPSER v3 subject (within-session z-score)", ROOT / "results/v3/gtao_grid_v3_subject_person_strat"),
    ("DAiSEE (group k-fold)", ROOT / "results/v2/raw_daisee/daisee_gt_grid/P2_groupkfold"),
    ("EngageNet (group k-fold)", ROOT / "results/v2/raw_engagenet/engagenet_gt_grid/P2_groupkfold"),
]
STREAMS = ("face", "facemesh", "head_pose", "physio")
LABELS = {"face": "Face", "facemesh": "Face mesh", "head_pose": "Head pose",
          "physio": "Physiology", "fusion": "Best fusion"}
BLUE, ORANGE = "#2a78d6", "#eb6834"          # dataviz palette slots 1 and 2 (light)
SHORT = {"paper slice, session folds": "slice, session folds",
         "person folds, z-score": "person folds, z-score",
         "within-session z-score": "session z-score",
         "group k-fold": "group k-fold"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e4e1"


def analyse(path):
    runs = load_runs(path)
    summary = {c: summarise(s) for c, s in runs.items() if not c.startswith("shuffle")}
    first = next(iter(runs.values()))[0]
    rows = {}
    for stream in STREAMS:
        cands = [c for c in summary if c.startswith(f"unimodal_{stream}")
                 and not c.startswith("unimodal_facemesh" if stream == "face" else "~")]
        if cands:
            best = max(cands, key=lambda c: summary[c]["qwk"])
            rows[stream] = {"config": best, **summary[best]}
    fusions = [c for c in summary if c.startswith(("gate_", "core_"))]
    if fusions:
        best = max(fusions, key=lambda c: summary[c]["qwk"])
        rows["fusion"] = {"config": best, **summary[best]}
    return {"n_windows": int(len(first["y"])), "n_persons": int(len(np.unique(first["person"]))),
            "label_between_person": label_between_share(first["y"], first["person"]),
            "label_between_session": label_between_share(first["y"], first["subject"]),
            "rows": rows}


def corpus_labels(path):
    d = np.load(path, allow_pickle=False)
    y = d["y"].astype(float)
    sessions = d["subjects"].astype(str)
    persons = np.array(["/".join([s.split("/")[0], s.split("/")[2]]) for s in sessions])
    experiments = np.array(["/".join(s.split("/")[:2]) for s in sessions])
    return {"n_windows": int(len(y)), "n_persons": int(len(np.unique(persons))),
            "label_between_person": label_between_share(y, persons),
            "label_between_experiment": label_between_share(y, experiments),
            "label_between_session": label_between_share(y, sessions)}


def figure(results, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK, "font.family": "sans-serif"})
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(7.2, 2.5), sharex=True, sharey=True)
    order = ["face", "facemesh", "head_pose", "physio", "fusion"]
    for ax, name in zip(axes, names):
        res = results[name]
        for i, key in enumerate(order):
            row = res["rows"].get(key)
            ax.axhline(i, color=GRID, lw=0.6, zorder=0)
            if row is None:
                ax.text(0.5, i, "no sensor", ha="center", va="center", color=MUTED, fontsize=6.5)
                continue
            b, w = row["between"], row["within"]
            ax.plot([w, b], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
            ax.scatter([b], [i], s=34, color=BLUE, edgecolor="white", linewidth=1, zorder=3)
            ax.scatter([w], [i], s=34, color=ORANGE, edgecolor="white", linewidth=1, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=0)
        title = name.split(" (")[0]
        sub = name.split(" (")[1].rstrip(")") if " (" in name else ""
        ax.set_title(f"{title}\n{SHORT.get(sub, sub)}\nlabels {res['label_between_person']:.0%} between",
                     fontsize=6.8, color=INK)
        ax.set_xlim(-0.75, 1.0)
        ax.set_xticks([-0.5, 0, 0.5, 1.0])
        ax.set_xticklabels(["-.5", "0", ".5", "1"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([LABELS[k] for k in order])
    axes[0].invert_yaxis()
    fig.supxlabel("Pearson r between prediction and label", fontsize=8, color=INK, y=0.03)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, markersize=6,
                          label="Between persons (who)"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, markersize=6,
                          label="Within persons (when)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=None)
    args = ap.parse_args()
    results = {name: analyse(path) for name, path in RUNS if path.is_dir()}
    out = {"runs": results}
    if args.corpus:
        out["dipser_full_corpus_labels"] = corpus_labels(args.corpus)
    for name, res in results.items():
        print(f"\n{name}: {res['n_windows']} windows, {res['n_persons']} persons, label "
              f"between-person {res['label_between_person']:.0%}, between-session "
              f"{res['label_between_session']:.0%}")
        for key, row in res["rows"].items():
            print(f"  {LABELS[key]:12s} {row['config']:28s} QWK {row['qwk']:+.3f}  "
                  f"between {row['between']:+.2f}  within {row['within']:+.2f}")
    if "dipser_full_corpus_labels" in out:
        c = out["dipser_full_corpus_labels"]
        print(f"\nDIPSER full corpus (labels only): {c['n_windows']} windows, {c['n_persons']} "
              f"persons; between person {c['label_between_person']:.0%}, experiment "
              f"{c['label_between_experiment']:.0%}, session {c['label_between_session']:.0%}")
    dest = ROOT / "results/phase1b"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "whowhen.json").write_text(json.dumps(out, indent=1, default=float))
    figure(results, ROOT / "paper/figures/fig_whowhen.pdf")


if __name__ == "__main__":
    main()
