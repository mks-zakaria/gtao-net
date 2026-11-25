"""Phase 7 confirmatory analysis (rules pre-declared in context/PLAN_accuracy.md "Phase 7").

1. family: crop_only:<f> vs crop_only:tcheb, paired person bootstrap of pooled out-of-fold
   QWK over the 57 persons (2000 draws), two-sided p, Holm over 4 families, per run;
   a family "matters" only if Holm p < 0.05 and CI excludes 0 in BOTH runs (acc, qwk).
2. crop: crop_only:tcheb vs scene_only (same test).
3. overall: fixed-split test of dipser_crop vs dipser_rich (13 test persons, 3 seeds).
Writes results/v6/phase7.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

R = Path(__file__).resolve().parent.parent / "results"
FAMILIES = ("hahn", "kraw", "kraw_eyes", "kraw_mouth")
N_BOOT = 2000


def person_of(s):
    parts = str(s).split("/")
    return f"{parts[0]}/{parts[2]}" if len(parts) == 3 else str(s)


def q(y, p):
    return cohen_kappa_score(y, p, weights="quadratic")


def boot_pair(y, persons, pa, pb, seed=0):
    """b - a: point, 95% CI, two-sided p; pa/pb = lists of prediction arrays (seeds)."""
    ids = np.unique(persons)
    rows_of = {i: np.flatnonzero(persons == i) for i in ids}
    rng = np.random.default_rng(seed)
    metric = lambda preds, r: (np.mean([q(y[r], p[r]) for p in preds]),
                               np.mean([accuracy_score(y[r], p[r]) for p in preds]))
    allr = np.arange(len(y))
    qa, aa = metric(pa, allr)
    qb, ab = metric(pb, allr)
    d = []
    for _ in range(N_BOOT):
        r = np.concatenate([rows_of[i] for i in rng.choice(ids, len(ids))])
        (x1, y1), (x2, y2) = metric(pa, r), metric(pb, r)
        d.append((x2 - x1, y2 - y1))
    d = np.array(d)
    p = float(min(1.0, 2 * min((d[:, 0] <= 0).mean(), (d[:, 0] >= 0).mean())))
    ci = lambda c: [round(float(np.percentile(d[:, c], 2.5)), 4), round(float(np.percentile(d[:, c], 97.5)), 4)]
    return {"a_qwk": round(qa, 4), "b_qwk": round(qb, 4), "diff_qwk": round(qb - qa, 4), "ci_qwk": ci(0),
            "p_qwk": p, "a_acc": round(aa, 4), "b_acc": round(ab, 4), "diff_acc": round(ab - aa, 4),
            "ci_acc": ci(1), "persons": int(len(ids))}


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


def main():
    report = {"family": {}, "crop_vs_scene": {}, "overall_test": {}}
    for run in ("acc", "qwk"):
        z = np.load(R / "v4" / f"dipser_crop_{run}" / "family_swap_oof.npz", allow_pickle=True)
        y, persons = z["y"], z["persons"].astype(str)
        base = [z["crop_only__tcheb"]]
        fam = {f: boot_pair(y, persons, base, [z[f"crop_only__{f}"]], seed=i) for i, f in enumerate(FAMILIES)}
        for f, adj in zip(FAMILIES, holm([fam[f]["p_qwk"] for f in FAMILIES])):
            fam[f]["p_holm"] = round(adj, 4)
            fam[f]["matters"] = adj < 0.05 and fam[f]["ci_qwk"][0] > 0
        report["family"][run] = fam
        report["crop_vs_scene"][run] = boot_pair(y, persons, [z["scene_only"]], base, seed=9)
        # 3. fixed-split test vs the rich-feature search (same 13 persons, 3 seeds each)
        load = lambda d: [json.load(open(R / "v4" / d / f"preds_test_seed{s}.json")) for s in range(3)]
        a, b = load(f"dipser_rich_{run}"), load(f"dipser_crop_{run}")
        assert a[0]["keys"] == b[0]["keys"]
        yt = np.array(a[0]["y"])
        pt = np.array([person_of(s) for s in a[0]["subject"]])
        report["overall_test"][run] = boot_pair(yt, pt, [np.array(s["pred"]) for s in a],
                                                [np.array(s["pred"]) for s in b], seed=11)
    report["family_matters_in_both_runs"] = {
        f: bool(report["family"]["acc"][f]["matters"] and report["family"]["qwk"][f]["matters"]) for f in FAMILIES}
    out = R / "v6" / "phase7.json"
    out.write_text(json.dumps(report, indent=1))
    for run in ("acc", "qwk"):
        print(f"== run {run}")
        for f in FAMILIES:
            r = report["family"][run][f]
            print(f"  {f:10s} vs tcheb: QWK {r['a_qwk']:.3f} -> {r['b_qwk']:.3f}  diff {r['diff_qwk']:+.3f} {r['ci_qwk']} "
                  f"p {r['p_qwk']:.3f} holm {r['p_holm']:.3f}  acc diff {r['diff_acc']:+.3f}")
        r = report["crop_vs_scene"][run]
        print(f"  crop(tcheb) vs scene: QWK {r['a_qwk']:.3f} -> {r['b_qwk']:.3f} diff {r['diff_qwk']:+.3f} {r['ci_qwk']} p {r['p_qwk']:.4f}; "
              f"acc {r['a_acc']:.3f} -> {r['b_acc']:.3f} {r['ci_acc']}")
        r = report["overall_test"][run]
        print(f"  test crop vs rich: QWK {r['a_qwk']:.3f} -> {r['b_qwk']:.3f} {r['ci_qwk']} p {r['p_qwk']:.3f}; "
              f"acc {r['a_acc']:.3f} -> {r['b_acc']:.3f} diff {r['diff_acc']:+.3f} {r['ci_acc']}")
    print("family matters in both runs:", report["family_matters_in_both_runs"])


if __name__ == "__main__":
    main()
