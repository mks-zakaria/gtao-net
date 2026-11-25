"""Paper vs rich Tchebichef features on the same fixed 13-person test split (v4 search).

Paired person bootstrap (persons resampled, same draw for both runs), metrics averaged
over the 3 final seeds. Also: what the search explored (best val per face_k / seq_t).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

R = Path(__file__).resolve().parent.parent / "results" / "v4"


def person_of(s):
    """Same rule as kaggle/v4_search/search.py: group/subject (a student across experiments)."""
    parts = str(s).split("/")
    return f"{parts[0]}/{parts[2]}" if len(parts) == 3 else str(s)


def load(run):
    seeds = [json.load(open(R / run / f"preds_test_seed{s}.json")) for s in range(3)]
    keys = np.array(seeds[0]["keys"])
    for s in seeds[1:]:
        assert list(s["keys"]) == list(keys)
    return (np.array(seeds[0]["y"]), np.array([person_of(x) for x in seeds[0]["subject"]]),
            [np.array(s["pred"]) for s in seeds], keys)


def metrics(y, preds, rows):
    acc = np.mean([accuracy_score(y[rows], p[rows]) for p in preds])
    qwk = np.mean([cohen_kappa_score(y[rows], p[rows], weights="quadratic") for p in preds])
    return acc, qwk


def paired(a, b, n_boot=2000, seed=0):
    ya, sa, pa, ka = load(a)
    yb, sb, pb, kb = load(b)
    assert list(ka) == list(kb) and (ya == yb).all()
    persons = np.unique(sa)
    rows_of = {p: np.where(sa == p)[0] for p in persons}
    rng = np.random.default_rng(seed)
    point = {a: metrics(ya, pa, slice(None)), b: metrics(yb, pb, slice(None))}
    draws = []
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[p] for p in rng.choice(persons, len(persons))])
        ma, mb = metrics(ya, pa, rows), metrics(yb, pb, rows)
        draws.append([ma[0], ma[1], mb[0], mb[1], mb[0] - ma[0], mb[1] - ma[1]])
    d = np.array(draws)
    ci = lambda c: [round(float(np.percentile(d[:, c], 2.5)), 4), round(float(np.percentile(d[:, c], 97.5)), 4)]
    return {"persons": len(persons), "windows": int(len(ya)),
            a: {"acc": round(point[a][0], 4), "acc_ci": ci(0), "qwk": round(point[a][1], 4), "qwk_ci": ci(1)},
            b: {"acc": round(point[b][0], 4), "acc_ci": ci(2), "qwk": round(point[b][1], 4), "qwk_ci": ci(3)},
            "diff_acc": round(point[b][0] - point[a][0], 4), "diff_acc_ci": ci(4),
            "p_acc_le0": float((d[:, 4] <= 0).mean()),
            "diff_qwk": round(point[b][1] - point[a][1], 4), "diff_qwk_ci": ci(5),
            "p_qwk_le0": float((d[:, 5] <= 0).mean())}


def explored(run):
    trials = [t for t in json.load(open(R / run / "study.json"))["trials"] if t.get("value") is not None]
    out = {}
    for name in ("face_k", "seq_t"):
        best = {}
        for t in trials:
            v = t["params"].get(name)
            if v is not None:
                best.setdefault(v, []).append(t["value"])
        out[name] = {str(k): {"n": len(v), "best_val": round(max(v), 4)} for k, v in sorted(best.items())}
    return out


if __name__ == "__main__":
    report = {}
    for obj in ("acc", "qwk"):
        a, b = f"dipser_{obj}", f"dipser_rich_{obj}"
        if not (R / a / "preds_test_seed0.json").exists():
            print(f"skip {obj}: {a} missing", file=sys.stderr)
            continue
        report[obj] = paired(a, b) | {"explored_rich": explored(b)}
    out = R / "rich_vs_paper.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
