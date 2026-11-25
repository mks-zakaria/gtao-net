"""Shared statistics for the v3 analysis (Phase 1 and Phase 1b of PLAN_accuracy.md).

* fast quadratic weighted kappa (numpy; identical to sklearn's, checked in __main__)
* who/when decomposition: between-person vs within-person agreement
* person-level bootstrap CIs (resample PERSONS, keep all their windows)
* matched-pair factorial contrasts with Wilcoxon + Holm correction

A run directory holds <config>/seed<k>/oof.json files with
oof = {y, pred, subject}. DIPSER subjects are student-sessions
("group/experiment/subject"); the person is "group/subject". Other datasets'
subject ids are used as persons directly.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

N_CLASSES_MAX = 4


def person_of(subject):
    parts = str(subject).split("/")
    return f"{parts[0]}/{parts[2]}" if len(parts) == 3 else str(subject)


def qwk(y, p, k=None):
    """Quadratic weighted kappa; 0 when undefined (constant predictions/labels)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=int)
    k = k or max(int(y.max()), int(p.max())) + 1
    observed = np.zeros((k, k))
    np.add.at(observed, (y, p), 1)
    n = observed.sum()
    expected = np.outer(observed.sum(1), observed.sum(0)) / n
    i, j = np.indices((k, k))
    weights = (i - j) ** 2 / (k - 1) ** 2
    denominator = (weights * expected).sum()
    if denominator == 0:
        return 0.0
    return float(1 - (weights * observed).sum() / denominator)


def _corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def decompose(y, p, groups):
    """(between_r, within_r): agreement of per-group means, and after removing them."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    groups = np.asarray(groups).astype(str)
    uniq, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    mean_y = np.bincount(inverse, y) / counts
    mean_p = np.bincount(inverse, p) / counts
    return _corr(mean_y, mean_p), _corr(y - mean_y[inverse], p - mean_p[inverse])


def label_between_share(y, groups):
    """Share of label variance explained by group (person/session) means."""
    y = np.asarray(y, float)
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    means = np.bincount(inverse, y) / np.bincount(inverse)
    return float(np.var(means[inverse]) / np.var(y)) if np.var(y) > 0 else 0.0


def balanced_accuracy(y, p):
    y, p = np.asarray(y), np.asarray(p)
    recalls = [float((p[y == k] == k).mean()) for k in np.unique(y)]
    return float(np.mean(recalls))


def load_runs(run_dir):
    """{config: [oof dict per seed]} for every <config>/seed*/oof.json under run_dir."""
    runs = {}
    for cfg_dir in sorted(Path(run_dir).iterdir()):
        if not cfg_dir.is_dir():
            continue
        seeds = []
        for f in sorted(cfg_dir.glob("seed*/oof.json")):
            o = json.loads(f.read_text())["oof"]
            seeds.append({"y": np.asarray(o["y"], int), "pred": np.asarray(o["pred"], int),
                          "subject": np.asarray(o["subject"]).astype(str),
                          "person": np.array([person_of(s) for s in o["subject"]])})
        if seeds:
            runs[cfg_dir.name] = seeds
    return runs


def summarise(seeds):
    """Seed-mean metrics of one configuration (who/when split by person)."""
    q = [qwk(s["y"], s["pred"]) for s in seeds]
    dec = [decompose(s["y"], s["pred"], s["person"]) for s in seeds]
    ba = [balanced_accuracy(s["y"], s["pred"]) for s in seeds]
    return {"qwk": float(np.mean(q)), "qwk_sd": float(np.std(q, ddof=1)) if len(q) > 1 else 0.0,
            "per_seed": q, "between": float(np.mean([d[0] for d in dec])),
            "within": float(np.mean([d[1] for d in dec])), "bacc": float(np.mean(ba))}


def person_bootstrap(seeds_a, seeds_b=None, n_boot=1000, seed=0):
    """95% CI of the seed-mean QWK and within-person r, resampling persons.

    With ``seeds_b`` it bootstraps the paired DIFFERENCE a - b of seed-mean QWK
    (both configs are scored on the same resampled persons).
    Returns {"qwk": (lo, hi), "within": (lo, hi)} or {"diff": (lo, hi)}.
    """
    rng = np.random.default_rng(seed)
    persons = np.unique(seeds_a[0]["person"])
    index = [{p: np.flatnonzero(s["person"] == p) for p in persons} for s in seeds_a]
    index_b = ([{p: np.flatnonzero(s["person"] == p) for p in persons} for s in seeds_b]
               if seeds_b else None)
    q_draws, w_draws, d_draws = [], [], []
    for _ in range(n_boot):
        sample = rng.choice(persons, size=len(persons), replace=True)
        qa, wa, qb = [], [], []
        for k, s in enumerate(seeds_a):
            rows = np.concatenate([index[k][p] for p in sample])
            # duplicated persons must stay distinct groups for the within split
            tags = np.concatenate([np.full(len(index[k][p]), j) for j, p in enumerate(sample)])
            qa.append(qwk(s["y"][rows], s["pred"][rows]))
            wa.append(decompose(s["y"][rows], s["pred"][rows], tags)[1])
            if index_b:
                sb = seeds_b[k]
                rows_b = np.concatenate([index_b[k][p] for p in sample])
                qb.append(qwk(sb["y"][rows_b], sb["pred"][rows_b]))
        q_draws.append(np.mean(qa))
        w_draws.append(np.mean(wa))
        if index_b:
            d_draws.append(np.mean(qa) - np.mean(qb))
    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    if index_b:
        return {"diff": ci(d_draws)}
    return {"qwk": ci(q_draws), "within": ci(w_draws)}


def _pairs(names, factor):
    """Matched pairs (a, b) of fusion configs that differ only in ``factor``."""
    def parse(name):
        parts = name.split("_")
        loss = parts[3] if len(parts) > 3 else "coral"
        return {"fusion": parts[0], "skip": parts[1], "drop": parts[2], "loss": loss}

    def build(cfg):
        base = f"{cfg['fusion']}_{cfg['skip']}_{cfg['drop']}"
        return base if cfg["loss"] == "coral" else f"{base}_{cfg['loss']}"

    fusion = [n for n in names if n.startswith(("gate_", "core_"))]
    levels = {"link": ("skip", "skip1", "skip0"), "gate_vs_core": ("fusion", "gate", "core"),
              "dropout": ("drop", "drop1", "drop0"), "soft_vs_coral": ("loss", "soft", "coral"),
              "soft_vs_hybrid": ("loss", "soft", "hybrid")}
    key, level_a, level_b = levels[factor]
    pairs = []
    for name in fusion:
        cfg = parse(name)
        if cfg[key] != level_a:
            continue
        other = dict(cfg)
        other[key] = level_b
        if build(other) in names:
            pairs.append((name, build(other)))
    return pairs


def contrasts(summary, factors=("link", "gate_vs_core", "dropout", "soft_vs_coral", "soft_vs_hybrid")):
    """Wilcoxon on config-level mean QWK differences per factor, Holm-corrected."""
    rows = []
    for factor in factors:
        pairs = _pairs(set(summary), factor)
        if len(pairs) < 2:
            continue
        diffs = np.array([summary[a]["qwk"] - summary[b]["qwk"] for a, b in pairs])
        p = 1.0 if np.allclose(diffs, 0) else float(wilcoxon(diffs).pvalue)
        rows.append({"factor": factor, "pairs": len(pairs), "wins": int((diffs > 0).sum()),
                     "median": float(np.median(diffs)), "p": p})
    order = np.argsort([r["p"] for r in rows])
    m, running = len(rows), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i]["p"]))
        rows[i]["p_holm"] = running
    return rows


if __name__ == "__main__":
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(1)
    for _ in range(200):
        y, p = rng.integers(0, 3, 300), rng.integers(0, 3, 300)
        assert abs(qwk(y, p) - cohen_kappa_score(y, p, weights="quadratic")) < 1e-9
    print("qwk matches sklearn on 200 random cases")
