# ---- phase10a.py (verbatim, without its main call) ----
"""Phase 10A: sparse-probe tracking (rules pre-declared in context/PLAN_accuracy.md "Phase 10A").

Usage: python analysis_v3/phase10a.py [corpus_full.npz] [--fast]   (on Kaggle the corpus is found under /kaggle/input)
Writes oof.npz and phase10a.json to results/v11/ (or /kaggle/working). Kernel: kaggle/v11_probe.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, log_loss
from sklearn.preprocessing import StandardScaler

KAGGLE = Path("/kaggle/working").exists()
OUT = Path("/kaggle/working") if KAGGLE else Path(__file__).resolve().parent.parent / "results" / "v11"
PS = (8, 15, 38)
SEEDS = (0, 1, 2)
N_BOOT = 2000
FAST = "--fast" in sys.argv
MAX_ITER = 40 if FAST else 300


def load(path):
    d = np.load(path, allow_pickle=True)
    s = d["subjects"].astype(str)
    t = np.array([float(k.split("#")[1]) for k in d["keys"]])
    order = np.lexsort((t, s))
    s, y = s[order], d["y"][order].astype(int)
    group = np.array([x.split("/")[0] for x in s])
    person = np.array([x.split("/")[0] + "/" + x.split("/")[2] for x in s])
    start = np.r_[True, s[1:] != s[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(len(s)), 0))
    idx = np.arange(len(s)) - first                      # index of the window in its session
    streams = {}
    for name in ("face", "facemesh", "head_pose", "physio"):
        x = d[f"feat_{name}"][order].astype(np.float32)
        ok = d[f"avail_{name}"][order].astype(bool) & np.isfinite(x).all(1)
        streams[name] = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), ok)
    return s, y, group, person, idx, first, streams


def sensors(streams, train):
    """Per-fold sensor matrix: PCA(32) for face and mesh (fit on training rows), raw pose and physio."""
    cols = []
    for name, (x, ok) in streams.items():
        if x.shape[1] > 64:
            fit = train & ok
            sc = StandardScaler().fit(x[fit])
            pca = PCA(32, svd_solver="randomized", random_state=0).fit(sc.transform(x[fit]))
            z = pca.transform(sc.transform(x)).astype(np.float32)
        else:
            z = x.copy()
        z[~ok] = np.nan
        cols.append(z)
    return np.hstack(cols)


def fit_predict(xtr, ytr, ptr, xte, seed):
    """Early stopping on 15% of the training persons, then refit on all training rows."""
    rng = np.random.default_rng(seed)
    ids = np.unique(ptr)
    val = np.isin(ptr, rng.choice(ids, max(1, int(round(0.15 * len(ids)))), replace=False))
    kw = dict(learning_rate=0.05, early_stopping=False, random_state=seed)
    m = HistGradientBoostingClassifier(max_iter=MAX_ITER, **kw).fit(xtr[~val], ytr[~val])
    losses = [log_loss(ytr[val], p, labels=m.classes_) for p in m.staged_predict_proba(xtr[val])]
    best = int(np.argmin(losses)) + 1
    m = HistGradientBoostingClassifier(max_iter=best, **kw).fit(xtr, ytr)
    return m.classes_[m.predict_proba(xte).argmax(1)], best


def metric(y, preds, rows):
    return (float(np.mean([cohen_kappa_score(y[rows], p[rows], weights="quadratic") for p in preds])),
            float(np.mean([accuracy_score(y[rows], p[rows]) for p in preds])))


def paired(y, persons, pa, pb, rows, seed):
    """b minus a on the scored rows; person bootstrap."""
    ids = np.unique(persons[rows])
    rows_of = {i: rows[persons[rows] == i] for i in ids}
    rng = np.random.default_rng(seed)
    (qa, aa), (qb, ab) = metric(y, pa, rows), metric(y, pb, rows)
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
            "ci_acc": ci(1)}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else str(next(Path("/kaggle/input").rglob("corpus_full.npz")))
    print("corpus:", path, flush=True)
    s, y, group, person, idx, first, streams = load(path)
    n = len(y)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = np.unique(group)
    oof = {}
    sens_fold = {}
    t0 = time.time()
    for g in groups:
        sens_fold[g] = sensors(streams, group != g)
        print(f"sensors fold {g}: {sens_fold[g].shape} ({time.time() - t0:.0f}s)", flush=True)

    # sensors_only: independent of P
    for seed in SEEDS:
        pred = np.full(n, -1)
        for g in groups:
            tr, te = group != g, group == g
            pred[te], best = fit_predict(sens_fold[g][tr], y[tr], person[tr], sens_fold[g][te], seed)
        oof[f"sensors_only_seed{seed}"] = pred
        print(f"sensors_only seed {seed} done, last best_iter {best} ({time.time() - t0:.0f}s)", flush=True)

    for P in PS:
        probe = first + (idx // P) * P                   # row of the last probe
        prev = np.where(idx // P >= 1, probe - P, -1)    # row of the probe before it
        lab = np.column_stack([y[probe], np.where(prev >= 0, y[np.maximum(prev, 0)], np.nan),
                               idx % P, idx // P]).astype(np.float32)
        scored = idx % P != 0
        oof[f"persist_P{P}"] = y[probe]
        for seed in SEEDS:
            p_lab, p_full = np.full(n, -1), np.full(n, -1)
            for g in groups:
                tr, te = (group != g) & scored, (group == g) & scored
                p_lab[te], b1 = fit_predict(lab[tr], y[tr], person[tr], lab[te], seed)
                sx = sens_fold[g]
                full = np.hstack([lab, sx, sx - sx[probe]])
                p_full[te], b2 = fit_predict(full[tr], y[tr], person[tr], full[te], seed)
            oof[f"label_only_P{P}_seed{seed}"], oof[f"full_P{P}_seed{seed}"] = p_lab, p_full
            print(f"P={P} seed {seed} done, best_iter {b1}/{b2} ({time.time() - t0:.0f}s)", flush=True)
        oof[f"scored_P{P}"] = scored
    np.savez_compressed(OUT / "oof.npz", y=y, persons=person, groups=group, subjects=s, **oof)

    report = {"n_windows": int(n), "persons": int(len(np.unique(person))), "fast": FAST,
              "majority_acc": round(float(np.bincount(y).max() / n), 4)}
    pvals = []
    for P in PS:
        rows = np.flatnonzero(oof[f"scored_P{P}"])
        arm = lambda a: [oof[f"{a}_P{P}_seed{k}"] for k in SEEDS]
        per = [oof[f"persist_P{P}"]]
        so = [oof[f"sensors_only_seed{k}"] for k in SEEDS]
        r = {"scored": int(len(rows)),
             "full_vs_label_only": paired(y, person, arm("label_only"), arm("full"), rows, P),
             "full_vs_persist": paired(y, person, per, arm("full"), rows, P + 1),
             "label_only_vs_persist": paired(y, person, per, arm("label_only"), rows, P + 2)}
        q, a = metric(y, so, rows)
        r["sensors_only"] = {"qwk": round(q, 4), "acc": round(a, 4)}
        best = max((r["full_vs_label_only"]["b_acc"], r["full_vs_label_only"]["b_qwk"], "full"),
                   (r["full_vs_label_only"]["a_acc"], r["full_vs_label_only"]["a_qwk"], "label_only"))
        r["target_reached"] = bool(best[0] >= 0.70 and best[1] >= 0.50)
        r["best_deployable_arm"] = best[2]
        pvals.append(r["full_vs_label_only"]["p_qwk"])
        report[f"P{P}"] = r
    # Holm over the three P
    order = np.argsort(pvals)
    adj, run = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    for P, a in zip(PS, adj):
        f = report[f"P{P}"]["full_vs_label_only"]
        f["p_holm"] = round(a, 4)
        report[f"P{P}"]["sensors_help"] = bool(a < 0.05 and f["ci_qwk"][0] > 0)
    (OUT / ("phase10a_fast.json" if FAST else "phase10a.json")).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))



# ---- phase10b.py ----
"""Phase 10B: learned face encoder (rules pre-declared in context/PLAN_accuracy.md "Phase 10B" + Amendment 1).

Usage: python analysis_v3/phase10b.py [corpus_full.npz face_embeddings.npz] [--fast]
(on Kaggle both files are found under /kaggle/input). Writes oof.npz and phase10b.json to
results/v12/ (or /kaggle/working). Kernel: kaggle/v12_face_encoder (phase10a.py inlined).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler


KAGGLE = Path("/kaggle/working").exists()
OUT = Path("/kaggle/working") if KAGGLE else Path(__file__).resolve().parent.parent / "results" / "v12"
PS = (8, 15, 38)
ENC_PCA = 64


def holm(pvals):
    order = np.argsort(pvals)
    adj, run = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    return adj


def collapse(v):
    """1..5 scores -> 0/1/2 ({1,2}, 3, {4,5}); 0 (no vote) -> -1."""
    return np.where(v <= 0, -1, np.where(v <= 2, 0, np.where(v == 3, 1, 2)))


def plurality(votes):
    """Plurality over the 3-class votes of one row (-1 = none); ties to the lower class."""
    counts = np.stack([(votes == c).sum(1) for c in range(3)], 1)
    return np.where(counts.sum(1) > 0, counts.argmax(1), -1)


def align(corpus_path, emb_path):
    """Embedding rows in the order of phase10a.load (sorted by subject, then window start)."""
    d = np.load(corpus_path, allow_pickle=True)
    s = d["subjects"].astype(str)
    t = np.array([float(k.split("#")[1]) for k in d["keys"]])
    keys = d["keys"].astype(str)[np.lexsort((t, s))]
    e = np.load(emb_path, allow_pickle=False)
    row = {k: i for i, k in enumerate(e["keys"].astype(str))}
    take = np.array([row.get(k, -1) for k in keys])
    matched = float((take >= 0).mean())
    print(f"embeddings matched: {matched:.4f} of {len(keys)} windows", flush=True)
    if matched < 0.99:
        sys.exit(f"STOP: only {matched:.2%} of the corpus windows have embeddings (rule: >= 99%)")
    ok = take >= 0
    take = np.maximum(take, 0)
    face = np.where(ok, e["face_frac"][take], 0.0).astype(np.float32)
    encs = {}
    for name in e.files:
        if name.startswith("emb_"):
            x = e[name][take].astype(np.float32)
            x[~ok] = 0.0
            bad = int((~np.isfinite(x).all(1)).sum())
            if bad:
                sys.exit(f"STOP: {name} has {bad} non-finite windows (re-embed with --fp32)")
            encs[name[4:]] = x
    names = list(e["rater_names"].astype(str))
    votes = collapse(e["votes"][take].astype(int))
    votes[~ok] = -1
    return encs, face, votes, names, matched


def encoder_block(x, face, train):
    """z-score + PCA(64) fit on training rows with a face; NaN where no face; + face fraction."""
    has = face > 0
    fit = train & has
    sc = StandardScaler().fit(x[fit])
    pca = PCA(ENC_PCA, svd_solver="randomized", random_state=0).fit(sc.transform(x[fit]))
    z = pca.transform(sc.transform(x)).astype(np.float32)
    z[~has] = np.nan
    return np.hstack([z, face[:, None]])


def probe_features(probe_val, first, idx, P):
    """Phase 10A probe block, with the probe value taken from probe_val (-1 = missing -> NaN)."""
    probe = first + (idx // P) * P
    prev = np.where(idx // P >= 1, probe - P, -1)
    pv = probe_val.astype(np.float32)
    pv[pv < 0] = np.nan
    lab = np.column_stack([pv[probe], np.where(prev >= 0, pv[np.maximum(prev, 0)], np.nan),
                           idx % P, idx // P]).astype(np.float32)
    return lab, probe


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        corpus_path, emb_path = args[0], args[1]
    else:
        corpus_path = str(next(Path("/kaggle/input").rglob("corpus_full.npz")))
        emb_path = str(next(Path("/kaggle/input").rglob("face_embeddings.npz")))
    print("corpus:", corpus_path, "| embeddings:", emb_path, flush=True)
    s, y, group, person, idx, first, streams = load(corpus_path)
    encs, face, votes, rater_names, matched = align(corpus_path, emb_path)
    n = len(y)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = np.unique(group)
    enc_names = sorted(encs)
    print(f"encoders: {enc_names}; raters: {rater_names}", flush=True)
    oof = {}
    t0 = time.time()

    # ---- primary: base vs base + encoder (leave-one-class-out, all windows) ----
    sens_fold, enc_fold = {}, {}
    for g in groups:
        tr = group != g
        sens_fold[g] = sensors(streams, tr)
        enc_fold[g] = {e: encoder_block(encs[e], face, tr) for e in enc_names}
        print(f"features fold {g}: sensors {sens_fold[g].shape} ({time.time() - t0:.0f}s)", flush=True)
    arms = {"base": lambda g: sens_fold[g]}
    for e in enc_names:
        arms[f"enc_{e}"] = lambda g, e=e: np.hstack([sens_fold[g], enc_fold[g][e]])
        arms[f"enc_only_{e}"] = lambda g, e=e: enc_fold[g][e]
    for arm, build in arms.items():
        for seed in SEEDS:
            pred = np.full(n, -1)
            for g in groups:
                tr, te = group != g, group == g
                x = build(g)
                pred[te], best = fit_predict(x[tr], y[tr], person[tr], x[te], seed)
            oof[f"{arm}_seed{seed}"] = pred
            print(f"{arm} seed {seed} done, last best_iter {best} ({time.time() - t0:.0f}s)", flush=True)

    allrows = np.arange(n)
    arm_preds = lambda a: [oof[f"{a}_seed{k}"] for k in SEEDS]
    report = {"n_windows": int(n), "persons": int(len(np.unique(person))), "fast": FAST,
              "matched": round(matched, 4), "encoders": enc_names,
              "majority_acc": round(float(np.bincount(y).max() / n), 4), "arms": {}, "tests": {}}
    for arm in arms:
        q, a = metric(y, arm_preds(arm), allrows)
        report["arms"][arm] = {"qwk": round(q, 4), "acc": round(a, 4)}
    pvals = []
    for i, e in enumerate(enc_names):
        r = paired(y, person, arm_preds("base"), arm_preds(f"enc_{e}"), allrows, 10 + i)
        report["tests"][f"enc_{e}_vs_base"] = r
        pvals.append(r["p_qwk"])
    for e, a in zip(enc_names, holm(pvals)):
        r = report["tests"][f"enc_{e}_vs_base"]
        r["p_holm"] = round(a, 4)
        r["helps"] = bool(a < 0.05 and r["ci_qwk"][0] > 0)

    # ---- secondary (descriptive): self-report probe, target = external plurality ----
    ext_cols = [i for i, r in enumerate(rater_names) if r.startswith("labeler_")]
    self_col = rater_names.index("self_labeling")
    target = plurality(votes[:, ext_cols])
    selfrep = votes[:, self_col]
    both = (target >= 0) & (selfrep >= 0)
    sec = {"external_raters": [rater_names[i] for i in ext_cols],
           "target_available": int((target >= 0).sum()), "self_available": int((selfrep >= 0).sum()),
           "self_vs_external": {"n": int(both.sum()),
                                "acc": round(float(accuracy_score(target[both], selfrep[both])), 4),
                                "qwk": round(float(cohen_kappa_score(target[both], selfrep[both],
                                                                     weights="quadratic")), 4)},
           "target_vs_consensus_y": round(float((target[target >= 0] == y[target >= 0]).mean()), 4)}
    for P in PS:
        lab, probe = probe_features(selfrep, first, idx, P)
        scored = (idx % P != 0) & (target >= 0) & (selfrep[probe] >= 0)
        rows = np.flatnonzero(scored)
        oof[f"self_persist_P{P}"] = selfrep[probe]
        oof[f"ext_persist_P{P}"] = target[probe]
        oof[f"self_scored_P{P}"] = scored
        for seed in SEEDS:
            p_lab, p_full = np.full(n, -1), np.full(n, -1)
            for g in groups:
                tr, te = (group != g) & scored, (group == g) & scored
                p_lab[te], b1 = fit_predict(lab[tr], target[tr], person[tr], lab[te], seed)
                sx = sens_fold[g]
                full = np.hstack([lab, sx, sx - sx[probe]])
                p_full[te], b2 = fit_predict(full[tr], target[tr], person[tr], full[te], seed)
            oof[f"self_label_only_P{P}_seed{seed}"], oof[f"self_full_P{P}_seed{seed}"] = p_lab, p_full
            print(f"self-probe P={P} seed {seed} done, best_iter {b1}/{b2} ({time.time() - t0:.0f}s)", flush=True)
        r = {"scored": int(len(rows))}
        for arm, preds in (("persist_self", [oof[f"self_persist_P{P}"]]),
                           ("persist_ext", [oof[f"ext_persist_P{P}"]]),
                           ("label_only_self", [oof[f"self_label_only_P{P}_seed{k}"] for k in SEEDS]),
                           ("full_self", [oof[f"self_full_P{P}_seed{k}"] for k in SEEDS])):
            q, a = metric(target, preds, rows)
            r[arm] = {"qwk": round(q, 4), "acc": round(a, 4)}
        sec[f"P{P}"] = r
    report["self_report_probe"] = sec

    np.savez_compressed(OUT / "oof.npz", y=y, persons=person, groups=group, subjects=s,
                        target_ext=target, self_report=selfrep, **oof)
    (OUT / ("phase10b_fast.json" if FAST else "phase10b.json")).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
