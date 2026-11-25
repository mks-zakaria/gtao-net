"""Phase 8 confirmatory analysis (rules pre-declared in context/PLAN_accuracy.md "Phase 8").

Input results/v8/oof.npz (kernel gtao-v8-dipser-time-context): out-of-fold predictions per
arm and seed, same 5 person-grouped folds. Paired person bootstrap (57 persons, 2000 draws)
of pooled QWK averaged over the 3 seeds; two-sided p.
1. gru vs none: context "helps" if the 95% CI excludes 0 and p < 0.05.
2. gru vs ema (1 comparison).
Exploratory: bigru vs none, ema vs none. Writes results/v8/phase8.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

R = Path(__file__).resolve().parent.parent / "results" / "v8"
N_BOOT = 2000


def metric(y, preds, rows):
    return (float(np.mean([cohen_kappa_score(y[rows], p[rows], weights="quadratic") for p in preds])),
            float(np.mean([accuracy_score(y[rows], p[rows]) for p in preds])))


def paired(y, persons, pa, pb, seed):
    ids = np.unique(persons)
    rows_of = {i: np.flatnonzero(persons == i) for i in ids}
    rng = np.random.default_rng(seed)
    everything = np.arange(len(y))
    (qa, aa), (qb, ab) = metric(y, pa, everything), metric(y, pb, everything)
    d = []
    for _ in range(N_BOOT):
        r = np.concatenate([rows_of[i] for i in rng.choice(ids, len(ids))])
        (x1, y1), (x2, y2) = metric(y, pa, r), metric(y, pb, r)
        d.append((x2 - x1, y2 - y1))
    d = np.array(d)
    ci = lambda c: [round(float(np.percentile(d[:, c], 2.5)), 4), round(float(np.percentile(d[:, c], 97.5)), 4)]
    p = float(min(1.0, 2 * min((d[:, 0] <= 0).mean(), (d[:, 0] >= 0).mean())))
    return {"a_qwk": round(qa, 4), "b_qwk": round(qb, 4), "diff_qwk": round(qb - qa, 4), "ci_qwk": ci(0),
            "p_qwk": p, "a_acc": round(aa, 4), "b_acc": round(ab, 4), "diff_acc": round(ab - aa, 4),
            "ci_acc": ci(1), "persons": int(len(ids))}


def main():
    z = np.load(R / "oof.npz", allow_pickle=True)
    y, persons = z["y"], z["persons"].astype(str)
    arm = lambda a: [z[f"{a}_seed{s}"] for s in range(3)]
    for a in ("none", "gru", "bigru", "ema"):
        assert all((p >= 0).all() for p in arm(a)), f"{a}: missing out-of-fold predictions"
    report = {"gru_vs_none": paired(y, persons, arm("none"), arm("gru"), 0),
              "gru_vs_ema": paired(y, persons, arm("ema"), arm("gru"), 1),
              "exploratory_bigru_vs_none": paired(y, persons, arm("none"), arm("bigru"), 2),
              "exploratory_ema_vs_none": paired(y, persons, arm("none"), arm("ema"), 3)}
    g = report["gru_vs_none"]
    report["context_helps"] = bool(g["p_qwk"] < 0.05 and g["ci_qwk"][0] > 0)
    e = report["gru_vs_ema"]
    report["gru_beats_ema"] = bool(e["p_qwk"] < 0.05 and e["ci_qwk"][0] > 0)
    (R / "phase8.json").write_text(json.dumps(report, indent=1))
    for name, r in report.items():
        if isinstance(r, dict):
            print(f"{name:28s} QWK {r['a_qwk']:.3f} -> {r['b_qwk']:.3f} diff {r['diff_qwk']:+.3f} {r['ci_qwk']} "
                  f"p {r['p_qwk']:.3f} | acc {r['a_acc']:.3f} -> {r['b_acc']:.3f} {r['ci_acc']}")
    print("context helps:", report["context_helps"], "| gru beats ema:", report["gru_beats_ema"])


if __name__ == "__main__":
    main()
