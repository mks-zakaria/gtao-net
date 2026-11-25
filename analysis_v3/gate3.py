"""Gate 3 verdict for the v3b full-corpus run (PLAN_accuracy.md, Phase 3b).

    python gate3.py --out <folder holding gtao_grid_v3b/, gtao_grid_v3b_arm_*/,
                     gtao_grid_v3b_leo/> [--boot 2000] [--json gate3.json]

Written 2026-11-18, before any v3b output existed. It applies the pre-declared
criteria mechanically:
 (a) >= 2 of {face, head pose, physio}: best single-stream config QWK >= 0.10, person-
     bootstrap CI lower bound > 0, and QWK minus its classmate control with CI > 0;
 (b) some single stream: within-SESSION r >= 0.10 with CI lower bound > 0;
 (c) best single stream: partial r(pred, label | R2) >= 0.10 with CI lower bound > 0
     (best fusion reported beside it);
 (d) H4: best fusion - best single > 0 with CI lower bound > 0 and bootstrap p < 0.05;
 (e) per fix (no_phys / no_lag / no_ctx / no_calib arms): selected - reverted over the
     8 single-stream configs; helps if Holm p < 0.05 (Wilcoxon over config x seed pairs)
     and the person-bootstrap CI of the mean difference excludes 0.
R2 = experiment prior: mean TRUE label of the windows of the other outer folds in the
same group/experiment (recomputed from each run's own folds).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from whowhen import qwk

STREAMS = ("face", "facemesh", "head_pose", "physio")
FIXES = ("no_phys", "no_lag", "no_ctx", "no_calib")


def person_of(session):
    parts = str(session).split("/")
    return f"{parts[0]}/{parts[2]}" if len(parts) == 3 else str(session)


def experiment_of(session):
    return "/".join(str(session).split("/")[:2])


def experiment_prior(y_true, sessions, folds):
    """Mean true label of other-fold windows in the same group/experiment."""
    experiments = np.array([experiment_of(s) for s in sessions])
    prior = np.full(len(y_true), np.nan)
    for e in np.unique(experiments):
        in_e = experiments == e
        for f in np.unique(folds[in_e]):
            train = in_e & (folds != f)
            if train.any():
                prior[in_e & (folds == f)] = y_true[train].mean()
    prior[np.isnan(prior)] = np.nanmean(y_true)
    return prior


def load_dir(run_dir):
    """{config: [per-seed dict of arrays]} for one grid folder."""
    runs = {}
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return runs
    for f in sorted(run_dir.glob("*/seed*/oof.json")):
        d = json.loads(f.read_text())
        o = d["oof"]
        rec = {k: np.asarray(o[k]) for k in ("keys", "y", "y_true", "pred", "subject", "fold")}
        rec["person"] = np.array([person_of(s) for s in rec["subject"]])
        rec["prior"] = experiment_prior(rec["y_true"].astype(float), rec["subject"], rec["fold"])
        runs.setdefault(f.parent.parent.name, []).append(rec)
    return runs


def _corr(a, b):
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def metrics(rec, rows=None, tags=None):
    """QWK, within-session r and partial r | R2 on rows (tags = session ids)."""
    if rows is None:
        rows = np.arange(len(rec["y"]))
        tags = rec["subject"]
    y = rec["y"][rows].astype(float)
    p = rec["pred"][rows].astype(float)
    out = {"qwk": qwk(rec["y"][rows].astype(int), rec["pred"][rows].astype(int))}
    _, inverse = np.unique(tags, return_inverse=True)
    counts = np.bincount(inverse)
    y_dev = y - (np.bincount(inverse, weights=y) / counts)[inverse]
    p_dev = p - (np.bincount(inverse, weights=p) / counts)[inverse]
    out["within_session"] = _corr(y_dev, p_dev)
    if rows is not None and len(rows) == len(rec["y"]) and tags is rec["subject"]:
        _, pinv = np.unique(rec["person"], return_inverse=True)
        pc = np.bincount(pinv)
        out["within_person"] = _corr(y - (np.bincount(pinv, weights=y) / pc)[pinv],
                                     p - (np.bincount(pinv, weights=p) / pc)[pinv])
    prior = rec["prior"][rows]
    design = np.column_stack([np.ones(len(prior)), prior])
    resid = lambda v: v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
    out["partial_r"] = _corr(resid(y), resid(p))
    return out


def seed_mean(recs, draw=None):
    vals = []
    for k, rec in enumerate(recs):
        if draw is None:
            vals.append(metrics(rec))
        else:
            rows, tags = draw[k]
            vals.append(metrics(rec, rows, tags))
    keys = set.intersection(*(set(v) for v in vals))
    return {m: float(np.mean([v[m] for v in vals])) for m in sorted(keys)}


def make_draws(recs, persons_sample):
    """Rows and session tags for one bootstrap draw, per seed (duplicates kept apart)."""
    out = []
    for rec in recs:
        if "_by_person" not in rec:
            rec["_by_person"] = {q: np.flatnonzero(rec["person"] == q)
                                 for q in np.unique(rec["person"])}
            _, rec["_session_id"] = np.unique(rec["subject"], return_inverse=True)
        n_sessions = rec["_session_id"].max() + 1
        rows, tags = [], []
        for j, person in enumerate(persons_sample):
            r = rec["_by_person"].get(person, np.zeros(0, dtype=int))
            rows.append(r)
            tags.append(rec["_session_id"][r] + j * n_sessions)
        out.append((np.concatenate(rows), np.concatenate(tags)))
    return out


def bootstrap(recs_a, recs_b=None, n_boot=2000, seed=0):
    """Person bootstrap: CIs of each metric (or of the a - b QWK difference)."""
    rng = np.random.default_rng(seed)
    persons = np.unique(np.concatenate([r["person"] for r in recs_a]))
    draws_a, draws_d = [], []
    for _ in range(n_boot):
        sample = rng.choice(persons, size=len(persons), replace=True)
        ma = seed_mean(recs_a, make_draws(recs_a, sample))
        draws_a.append(ma)
        if recs_b is not None:
            mb = seed_mean(recs_b, make_draws(recs_b, sample))
            draws_d.append(ma["qwk"] - mb["qwk"])
    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    result = {m: ci([d[m] for d in draws_a]) for m in draws_a[0]}
    if recs_b is not None:
        d = np.array(draws_d)
        result["diff"] = ci(d)
        result["diff_p"] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return result


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    root = Path(args.out)
    main_runs = load_dir(root / "gtao_grid_v3b")
    point = {c: seed_mean(r) for c, r in main_runs.items()}
    print(f"{'config':34s} {'QWK':>7s} {'within-s':>9s} {'partial':>8s} seeds")
    for c in sorted(point, key=lambda c: -point[c]["qwk"]):
        print(f"{c:34s} {point[c]['qwk']:+7.3f} {point[c]['within_session']:+9.3f} "
              f"{point[c]['partial_r']:+8.3f} {len(main_runs[c])}")

    singles = [c for c in point if c.startswith("unimodal_")]
    fusions = [c for c in point if c.startswith(("gate_", "core_"))]
    best_stream = {s: max([c for c in singles if c.startswith(f"unimodal_{s}_")],
                          key=lambda c: point[c]["qwk"]) for s in STREAMS
                   if any(c.startswith(f"unimodal_{s}_") for c in singles)}
    best_single = max(singles, key=lambda c: point[c]["qwk"])
    best_fusion = max(fusions, key=lambda c: point[c]["qwk"])
    ci = {c: bootstrap(main_runs[c], n_boot=args.boot) for c in set(best_stream.values())
          | {best_fusion}}
    result = {"point": point, "ci": ci, "best_stream": best_stream,
              "best_single": best_single, "best_fusion": best_fusion}

    # (a)
    recovered = []
    for s in ("face", "head_pose", "physio"):
        c = best_stream.get(s)
        mate = f"classmate_unimodal_{s}_cw"
        if not c or mate not in main_runs:
            continue
        vs_mate = bootstrap(main_runs[c], main_runs[mate], n_boot=args.boot)
        result.setdefault("vs_classmate", {})[s] = vs_mate
        if point[c]["qwk"] >= 0.10 and ci[c]["qwk"][0] > 0 and vs_mate["diff"][0] > 0:
            recovered.append(s)
    # (b)
    within_ok = [s for s, c in best_stream.items()
                 if point[c]["within_session"] >= 0.10 and ci[c]["within_session"][0] > 0]
    # (c)
    c_ok = (point[best_single]["partial_r"] >= 0.10 and ci[best_single]["partial_r"][0] > 0)
    # (d)
    d = bootstrap(main_runs[best_fusion], main_runs[best_single], n_boot=args.boot)
    above = sum(point[c]["qwk"] > point[best_single]["qwk"] for c in fusions)
    h4_flip = d["diff"][0] > 0 and d["diff_p"] < 0.05
    # (e)
    fix_rows = []
    for fix in FIXES:
        arm = load_dir(root / f"gtao_grid_v3b_arm_{fix}")
        if not arm:
            fix_rows.append({"fix": fix, "status": "never selected (arm identical, not run)",
                             "p": 1.0})
            continue
        pairs = []
        for c in singles:
            if c in arm:
                for a, b in zip(main_runs[c], arm[c]):
                    pairs.append(metrics(a)["qwk"] - metrics(b)["qwk"])
        stat_p = float(wilcoxon(pairs).pvalue) if np.any(np.abs(pairs) > 0) else 1.0
        fix_rows.append({"fix": fix, "mean_delta_qwk": float(np.mean(pairs)),
                         "n_pairs": len(pairs), "p": stat_p})
    for row, p_holm in zip(fix_rows, holm([r["p"] for r in fix_rows])):
        row["p_holm"] = p_holm
        row["helps"] = bool(p_holm < 0.05 and row.get("mean_delta_qwk", 0) > 0)
    base = load_dir(root / "gtao_grid_v3b_arm_base")
    if base:
        result["all_fixes_vs_base"] = {c: seed_mean(main_runs[c])["qwk"] - seed_mean(base[c])["qwk"]
                                       for c in singles if c in base}
    leo = load_dir(root / "gtao_grid_v3b_leo")
    result["leave_experiment_out"] = {c: seed_mean(r) for c, r in leo.items()}

    gate = {"a_recovered": recovered, "a_pass": len(recovered) >= 2,
            "b_streams": within_ok, "b_pass": bool(within_ok),
            "c_best_single": {"config": best_single, "partial_r": point[best_single]["partial_r"],
                              "ci": ci[best_single]["partial_r"]},
            "c_best_fusion": {"config": best_fusion, "partial_r": point[best_fusion]["partial_r"],
                              "ci": ci[best_fusion]["partial_r"]},
            "c_pass": bool(c_ok),
            "d_fusions_above_best_single": f"{above}/{len(fusions)}",
            "d_diff": point[best_fusion]["qwk"] - point[best_single]["qwk"],
            "d_ci": d["diff"], "d_p": d["diff_p"], "d_h4_flips": bool(h4_flip),
            "e_fixes": fix_rows}
    result["gate"] = gate
    print("\nGATE 3")
    print(f" (a) recovered (QWK>=0.10, CI>0, > classmate): {recovered or 'none'} -> "
          f"{'PASS' if gate['a_pass'] else 'FAIL'}")
    print(f" (b) within-session r >= 0.10, CI>0: {within_ok or 'none'} -> "
          f"{'PASS' if gate['b_pass'] else 'FAIL'}")
    print(f" (c) partial r | R2, best single {best_single}: {point[best_single]['partial_r']:+.3f} "
          f"CI [{ci[best_single]['partial_r'][0]:+.3f}, {ci[best_single]['partial_r'][1]:+.3f}] -> "
          f"{'PASS' if c_ok else 'FAIL'} (best fusion {point[best_fusion]['partial_r']:+.3f})")
    print(f" (d) best fusion - best single {gate['d_diff']:+.3f} CI [{d['diff'][0]:+.3f}, "
          f"{d['diff'][1]:+.3f}] p {d['diff_p']:.4f}; above: {gate['d_fusions_above_best_single']}"
          f" -> H4 {'FLIPS' if h4_flip else 'stays rejected'}")
    for r in fix_rows:
        print(f" (e) {r['fix']:9s} {r.get('mean_delta_qwk', 0):+.3f} p_holm {r['p_holm']:.4f} "
              f"{'HELPS' if r['helps'] else '-'} {r.get('status', '')}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
